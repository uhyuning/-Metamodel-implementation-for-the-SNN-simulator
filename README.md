# SNN Meta-Model: 파편화된 프레임워크에서 C++/SIMD 네이티브 커널까지

SNN(Spiking Neural Network) 연구 생태계는 snnTorch, SpikingJelly, BindsNET 등으로 파편화되어 있고, 각 프레임워크는 서로 다른 API와 뉴런 파라미터 표기를 가진다. 이 프로젝트는 **하나의 JSON 메타모델**에서 LIF 뉴런 다이내믹스를 표준화하고, 그 메타모델이 4개의 서로 다른 실행 백엔드(snnTorch / SpikingJelly / BindsNET / **C++·SIMD**)로 코드를 생성하도록 만든다. C++/SIMD 백엔드는 실제로 컴파일·실행되며, 동일 아키텍처 기준 Python 시뮬레이터 대비 지연시간을 실측으로 크게 줄인다.

## 한눈에 보는 결과

동일 모델(784 → 128 → 10, LIF, `tau_mem=10.0` · `v_threshold=1.0` · `v_reset=0.0`, batch=1, 단일 스레드, 워밍업 5회 + 30회 반복 측정 median 기준):

| 백엔드 | 지연시간 | vs C++/SIMD |
|---|---|---|
| snnTorch (Python, CPU) | 12.22 ~ 12.37 ms | — |
| SpikingJelly (Python, CPU) | 3.84 ~ 3.86 ms | — |
| **C++/SIMD (MSVC `/O2 /arch:AVX2`)** | **0.382 ~ 0.385 ms** | **96.8%↓ vs snnTorch · 90.0%↓ vs SpikingJelly** |

- MSVC 컴파일러 진단(`/Qvec-report:1`)으로 SIMD 벡터화가 실제로 일어났음을 확인 (`info C5001: 루프가 벡터화되었습니다`)
- 정적/실행 테스트 21개 전부 통과 (`pytest tests/`)
- 측정 방법과 한계는 [벤치마크 방법론](#벤치마크-방법론과-한계) 참고 — 한 대의 개발 머신에서 측정한 값이며, 실제 MCU/임베디드 보드 실측은 아직 범위 밖

## 왜 필요한가

- **파편화**: 동일한 SNN 모델을 snnTorch/SpikingJelly/BindsNET마다 매번 다시 구현해야 하고, 재현성이 떨어진다.
- **배포 공백**: 초저전력 온디바이스 배포 수요는 느는데, 연구용 Python 시뮬레이터에서 실배포 가능한 저지연 네이티브 커널로 이어지는 경로가 없다.
- **수작업 최적화**: 엣지 배포 시 필요한 LIF→IF 뉴런 단순화 같은 최적화를 연구자가 매번 손으로 처리한다.

## 아키텍처

```
JSON 메타모델
    │
    ▼
[src/parser.py] load_and_validate_metamodel()
    - 필수 키 검증
    - LIFNode/IFNode 레이어에 표준 LIF 다이내믹스(tau_mem, decay, v_threshold, v_reset, reset_mechanism) 적용
    │
    ▼
[src/codegen.py] generate_snn_code()
    - power_mode == "low_energy"면 LIFNode → IFNode 자동 교체 + decay 재계산
    - surrogate_function / step_mode를 캐노니컬 값 → 백엔드별 토큰으로 변환
    - CppSIMD 타깃이면 레이어를 버퍼 단위 연산 목록으로 변환
    │
    ▼
[Jinja2] 4개 템플릿 중 하나 렌더링
    ├─ snntorch_template.j2      → snn.Leaky / snn.Lapicque
    ├─ spikingjelly_template.j2  → neuron.LIFNode / neuron.IFNode
    ├─ bindsnet_template.j2      → LIFNodes / IFNodes
    └─ cpp_simd_template.j2      → __restrict 포인터 + branch-free C++ 커널
    │
    ▼
실행 가능한 코드 (.py × 3, .cpp × 1)
    │
    ▼ (C++만 해당)
scripts/build_cpp.py → MSVC(vcvars64) 컴파일 → 실행 → 지연시간 출력
```

같은 입력에서 4개의 산출물이 나오되, 물리적으로 의미 있는 값(뉴런의 누설 계수·임계값·리셋 방식)은 전부 `src/lif_dynamics.py`라는 단일 지점에서 계산된다.

## 핵심 기술

### 1. LIF 다이내믹스 표준화

LIF 뉴런의 막전위는 `tau_mem · dV/dt = -(V - V_reset) + I(t)`를 따르며, 오일러 이산화하면 `decay = exp(-dt / tau_mem)`이 된다. `src/lif_dynamics.py`의 `compute_decay()` 한 곳에서 이 값을 계산하고, 세 프레임워크(snnTorch의 `beta`, SpikingJelly의 `tau`, BindsNET의 `tc_decay`)가 전부 여기서 파생된다. `tests/test_codegen.py::test_lif_dynamics_are_identical_across_backends`가 동일한 `tau_mem`을 넣으면 세 백엔드에서 정확히 같은 수치가 나오는지 검증한다.

### 2. 하드웨어 인식 최적화 (Low-Energy 모드)

`power_mode: "low_energy"`가 감지되면 `codegen.py`가 `LIFNode`(누설 있음)를 `IFNode`(누설 없음, decay=1.0)로 자동 치환한다. IF는 지수 감쇠 연산이 없는 단순 임계값 비교라 연산량이 적다.

### 3. C++/SIMD 네이티브 백엔드

AVX2 인트린식을 직접 쓰는 대신 `__restrict` 포인터 + 분기 없는(branch-free) 뉴런 리셋 수식으로 작성해 컴파일러가 스스로 SIMD로 벡터화하도록 유도했다 (GCC/Clang/MSVC 이식성 확보). `src/codegen.py`의 `_build_cpp_ops()`가 레이어 시퀀스를 버퍼 이름·크기가 명시된 연산 목록으로 코드 생성 시점에 미리 계산해, C++에는 없는 "텐서 shape 추론"을 대신한다.

## 벤치마크 방법론과 한계

**통제한 변수**: 동일 아키텍처(784→128→10) · 동일 LIF 다이내믹스 · 동일 배치(1, 온디바이스 스트리밍 추론을 가정) · `torch.set_num_threads(1)`(PyTorch 멀티스레드 BLAS가 단일 스레드 C++ 대비 부당하게 유리해지지 않도록) · 워밍업 5회 + 30회 반복.

**한계**:
- 한 대의 개발 머신(x86-64, AVX2)에서 측정한 값 — 여러 하드웨어에 걸친 통계적 검증은 아직 아님
- 학습되지 않은 랜덤 가중치로 순전파만 수행 — 정확도가 아니라 연산 구조의 지연시간만 비교
- BindsNET은 아직 이 비교에 포함되지 않음
- 실제 MCU/임베디드 보드 실측이 아니라 개발 PC의 CPU 기준

## 사용법

### 코드 생성

```bash
pip install -r requirements.txt
python main.py   # examples/ 폴더의 JSON 메타모델 중 하나를 선택해 코드 생성
```

### 테스트 (정적 검증 + C++ 컴파일·실행 검증)

```bash
pip install pytest
pytest tests/ -v
```

MSVC(`vcvars64.bat`)가 설치돼 있으면 `tests/test_cpp_backend.py`가 C++ 코드를 실제로 컴파일·실행까지 검증하고, 없는 환경에서는 해당 테스트만 자동으로 skip된다.

### C++/SIMD 커널 직접 빌드

```bash
python scripts/build_cpp.py generated_code/CppSimdStandardModel_CppSIMD.cpp --vec-report
```

### Python vs C++ 벤치마크 재현

Python 시뮬레이터(snnTorch, SpikingJelly)는 무거운 의존성이라 별도 venv를 권장한다.

```bash
py -3.9 -m venv .venv_bench
.venv_bench\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cpu torch
.venv_bench\Scripts\python.exe -m pip install -r requirements-bench.txt
.venv_bench\Scripts\python.exe scripts\run_benchmark.py
```

## 프로젝트 구조

```
src/
  parser.py          # 메타모델 로드·검증 + LIF 다이내믹스 표준화
  codegen.py          # 하드웨어 인식 최적화 + 4개 백엔드 코드 생성
  lif_dynamics.py      # 시뮬레이터 독립적인 LIF 다이내믹스 단일 계산 지점
templates/
  snntorch_template.j2
  spikingjelly_template.j2
  bindsnet_template.j2
  cpp_simd_template.j2  # branch-free, auto-vectorizable C++ 커널
examples/              # 8종 메타모델 예제 (프레임워크 × 표준/저전력)
scripts/
  build_cpp.py          # MSVC 컴파일·실행 헬퍼
  run_benchmark.py       # Python vs C++ 지연시간 벤치마크
tests/
  test_codegen.py         # 정적 검증 + LIF/surrogate/step_mode 표준화 검증
  test_cpp_backend.py      # C++ 코드 생성 + 실제 컴파일·실행 검증
```

## 다음 단계

- BindsNET을 C++/SIMD 벤치마크 비교에 포함
- 실제 MCU/임베디드 타깃(ARM Cortex-M 등) 크로스컴파일 및 전력 실측
- 학습된 가중치 기반 정확도 검증 (현재는 순전파 지연시간만 비교)
- Conv/RNN 등 레이어 타입 확장

## License

[MIT](LICENSE)
