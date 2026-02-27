# 🚀 SNN Meta-Model 기반 자동 코드 생성기 (Hardware-aware)

본 프로젝트는 SNN(Spiking Neural Network) 연구의 파편화된 프레임워크 환경을 통합하고, 하드웨어 제약 조건에 최적화된 시뮬레이션 코드를 자동으로 생성하는 프레임워크입니다.

---

## ✨ 핵심 기술 (Core Technologies)

### 1. 🏗️ 메타모델 설계 및 추상화 인터페이스
SNN 모델 설계의 파편화를 해결하기 위해, 시뮬레이터에 독립적인 **중립적 메타모델(Neutral Metamodel)** 구조를 제안합니다.
* **Intermediate Representation**: `parser.py`를 통해 사용자의 모델 정의를 기계가 해석 가능한 중간 표현으로 변환합니다.
* **Pre-verification**: 레이어 간 피처 일치성 및 타임스텝 유효성을 사후 검증하여, 코드 생성 후 발생할 수 있는 런타임 에러를 설계 단계에서 최소화합니다.
* **Efficiency**: 반복적인 디버깅 시간을 단축시켜 연구 효율성을 극대화하는 기저 기술로 작동합니다.

### 2. ⚡ 하드웨어 제약 기반의 동적 모델 최적화
정적인 코드 변환을 넘어, 실행 환경의 자원 제약 조건에 대응하는 **적응형 최적화(Adaptive Optimization)** 로직을 내장하고 있습니다.
* **Low-Energy Mode**: 저전력 모드 활성화 시, LIF(Leaky Integrate-and-Fire) 노드를 연산 효율이 높은 IF(Integrate-and-Fire) 노드로 실시간 교체(Swapping)합니다.
* **Precision Tuning**: FP16 반정밀도(Half-precision) 캐스팅 코드를 자동 삽입하여 메모리 대역폭 점유율을 낮춥니다.
* **Hardware-aware**: 엣지 디바이스와 같은 극한 환경에서의 SNN 배포 및 연구 가능성을 시사합니다.

### 3. 🧩 Jinja2 기반의 확장형 템플릿 매핑 엔진
유지보수와 확장성을 고려하여 로직과 문법이 분리된 **디커플링(Decoupling)** 구조를 채택하였습니다.
* **Template-driven**: Jinja2 엔진을 활용하여 메인 로직의 수정 없이 템플릿 프로파일(`.j2`) 추가만으로 새로운 프레임워크(SpikingJelly, snnTorch, BindsNET 등)에 즉각 대응할 수 있습니다.
* **Auto-Mapping**: 네트워크 토폴로지 정의 방식을 각 프레임워크의 고유 API 규격으로 자동 사상(Mapping)합니다.
* **Cross-validation**: 단일 메타모델만으로 서로 다른 시뮬레이터 환경에서의 교차 검증을 손쉽게 수행할 수 있습니다.