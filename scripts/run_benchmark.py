"""
Python 시뮬레이터(snnTorch, SpikingJelly) vs C++/SIMD 백엔드 지연시간 비교 벤치마크.

동일한 메타모델(784->128->10, LIF, tau_mem=10/threshold=1.0/reset=0.0, batch=1)을
세 백엔드에 각각 생성해서, 같은 아키텍처·같은 다이내믹스·같은 배치 크기로 지연시간을
N회 반복 측정해 비교한다. 배치를 1로 고정한 이유는 이 프로젝트가 목표로 하는
"온디바이스/엣지" 추론이 보통 스트리밍 단건 추론이기 때문이다. torch 스레드도 1개로
고정해 C++(단일 스레드, 인트린식 없이 auto-vec만 사용)과 동일한 조건에서 비교한다.

사용 예:
    .venv_bench\\Scripts\\python.exe scripts\\run_benchmark.py
"""
import importlib.util
import json
import os
import statistics
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from src.parser import load_and_validate_metamodel
from src.codegen import generate_snn_code
from scripts.build_cpp import compile_cpp, run_exe

BENCH_DIR = os.path.join(ROOT_DIR, "benchmarks", "_generated")
TIME_STEPS = 100
N_RUNS = 30
N_WARMUP = 5

# surrogate_function/step_mode는 이제 codegen.py가 캐노니컬 값을 백엔드별 토큰으로
# 알아서 변환해주므로(SURROGATE_FUNCTION_MAP/STEP_MODE_MAP), 여기서는 시뮬레이터에
# 상관없이 같은 캐노니컬 값만 쓰면 된다.
def _make_spec(target_simulator, model_name):
    layers = [
        {"type": "linear", "in_features": 784, "out_features": 128},
        {"type": "LIFNode", "tau_mem": 10.0, "v_threshold": 1.0, "v_reset": 0.0, "surrogate_function": "atan"},
        {"type": "linear", "in_features": 128, "out_features": 10},
        {"type": "LIFNode", "tau_mem": 10.0, "v_threshold": 1.0, "v_reset": 0.0, "surrogate_function": "atan"},
    ]
    return {
        "model_name": model_name,
        "target_simulator": target_simulator,
        "time_steps": TIME_STEPS,
        "power_mode": "normal",
        "hardware_constraints": {},
        "layers": layers,
        "connections": [{"source": "input", "target": "layer_1"}],
        "backend": "torch",
        "step_mode": "multi_step",
    }


def _write_and_generate(target_simulator, model_name):
    os.makedirs(BENCH_DIR, exist_ok=True)
    spec_path = os.path.join(BENCH_DIR, f"{model_name}.json")
    spec = _make_spec(target_simulator, model_name)
    with open(spec_path, "w", encoding="utf-8") as f:
        json.dump(spec, f)
    loaded = load_and_validate_metamodel(spec_path)
    generate_snn_code(loaded, output_dir=os.path.relpath(BENCH_DIR, ROOT_DIR))
    return loaded


def _import_generated_module(py_path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, py_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def benchmark_snntorch():
    import torch

    torch.set_num_threads(1)
    model_name = "BenchSnnTorch"
    _write_and_generate("snnTorch", model_name)
    py_path = os.path.join(BENCH_DIR, f"{model_name}_snnTorch.py")
    module = _import_generated_module(py_path, model_name)
    model = getattr(module, model_name)().eval()

    x = torch.randn(TIME_STEPS, 1, 784)
    latencies = []
    with torch.no_grad():
        for i in range(N_WARMUP + N_RUNS):
            t0 = _now()
            model(x)
            dt = _now() - t0
            if i >= N_WARMUP:
                latencies.append(dt)
    return latencies


def benchmark_spikingjelly():
    import torch

    torch.set_num_threads(1)
    model_name = "BenchSpikingJelly"
    _write_and_generate("SpikingJelly", model_name)
    py_path = os.path.join(BENCH_DIR, f"{model_name}_SpikingJelly.py")
    module = _import_generated_module(py_path, model_name)
    model = getattr(module, model_name)().eval()

    x = torch.randn(TIME_STEPS, 1, 784)
    latencies = []
    with torch.no_grad():
        for i in range(N_WARMUP + N_RUNS):
            t0 = _now()
            model(x)
            dt = _now() - t0
            if i >= N_WARMUP:
                latencies.append(dt)
    return latencies


def benchmark_cpp():
    model_name = "BenchCppSimd"
    _write_and_generate("CppSIMD", model_name)
    cpp_path = os.path.join(BENCH_DIR, f"{model_name}_CppSIMD.cpp")
    exe_path = compile_cpp(cpp_path)

    latencies = []
    for i in range(N_WARMUP + N_RUNS):
        output = run_exe(exe_path)
        # "Latency: 0.4012 ms total | 0.004012 ms/step" 형태에서 total ms를 파싱
        for line in output.splitlines():
            if "Latency:" in line:
                ms = float(line.split("Latency:")[1].split("ms")[0].strip())
                if i >= N_WARMUP:
                    latencies.append(ms / 1000.0)  # 초 단위로 통일
                break
    return latencies


def _now():
    import time
    return time.perf_counter()


def _summarize(name, latencies_sec):
    ms = [v * 1000.0 for v in latencies_sec]
    return {
        "name": name,
        "mean_ms": statistics.mean(ms),
        "median_ms": statistics.median(ms),
        "stdev_ms": statistics.stdev(ms) if len(ms) > 1 else 0.0,
        "min_ms": min(ms),
        "max_ms": max(ms),
        "n": len(ms),
    }


def main():
    results = []
    print(f"[benchmark] time_steps={TIME_STEPS}, batch=1, warmup={N_WARMUP}, runs={N_RUNS}")

    print("[benchmark] snnTorch 측정 중...")
    results.append(_summarize("snnTorch (Python, CPU, 1 thread)", benchmark_snntorch()))

    print("[benchmark] SpikingJelly 측정 중...")
    results.append(_summarize("SpikingJelly (Python, CPU, 1 thread)", benchmark_spikingjelly()))

    print("[benchmark] C++/SIMD 측정 중...")
    results.append(_summarize("CppSIMD (MSVC /O2 /arch:AVX2, 1 thread)", benchmark_cpp()))

    print("\n" + "=" * 78)
    print(f"{'Backend':<40}{'mean(ms)':>10}{'median(ms)':>12}{'stdev(ms)':>10}")
    print("-" * 78)
    for r in results:
        print(f"{r['name']:<40}{r['mean_ms']:>10.4f}{r['median_ms']:>12.4f}{r['stdev_ms']:>10.4f}")
    print("=" * 78)

    cpp_median = next(r["median_ms"] for r in results if r["name"].startswith("CppSIMD"))
    for r in results:
        if not r["name"].startswith("CppSIMD"):
            reduction = (1 - cpp_median / r["median_ms"]) * 100.0
            print(f"C++/SIMD is {reduction:.1f}% faster than {r['name']} (median latency)")

    return results


if __name__ == "__main__":
    main()
