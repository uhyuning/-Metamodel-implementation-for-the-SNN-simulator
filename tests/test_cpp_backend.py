"""
C++/SIMD 백엔드 검증.

(1) 구조적 검증: 생성된 .cpp 소스에 표준 LIF 다이내믹스 수치가 정확히 반영됐는지
    (컴파일러 없이도 항상 실행됨).
(2) 실행 검증: MSVC(vcvars64.bat)가 있는 환경에서는 실제로 컴파일·실행까지 해서
    지연시간 출력이 나오는지 확인 (없는 환경에서는 자동으로 skip됨 — 이식성 확보).
"""
import os
import sys

import pytest

from src.codegen import generate_snn_code
from src.parser import load_and_validate_metamodel

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLES_DIR = os.path.join(ROOT_DIR, "examples")

CPP_EXAMPLES = ["cpp_simd_standard_config.json", "cpp_simd_low_power_config.json"]


def _find_vcvars64():
    sys.path.insert(0, ROOT_DIR)
    from scripts.build_cpp import find_vcvars64
    return find_vcvars64()


@pytest.fixture(params=CPP_EXAMPLES)
def generated_cpp(request, tmp_path):
    spec = load_and_validate_metamodel(os.path.join(EXAMPLES_DIR, request.param))
    generate_snn_code(spec, output_dir=str(tmp_path))
    output_path = tmp_path / f"{spec['model_name']}_CppSIMD.cpp"
    return {"path": output_path, "spec": spec, "config_name": request.param}


def test_cpp_output_reflects_standardized_dynamics(generated_cpp):
    source = generated_cpp["path"].read_text(encoding="utf-8")
    neuron_layers = [l for l in generated_cpp["spec"]["layers"] if l["type"] in ("LIFNode", "IFNode")]

    assert neuron_layers, f"{generated_cpp['config_name']}: no neuron layers in spec"
    for layer in neuron_layers:
        # 템플릿의 "%.10f"|format(...) 포맷과 동일한 방식으로 값을 재현해 비교한다.
        assert f"{layer['decay']:.10f}f" in source
        assert f"{layer['v_threshold']:.10f}f" in source


@pytest.mark.skipif(_find_vcvars64() is None, reason="MSVC(vcvars64.bat)가 이 환경에 없음")
def test_cpp_compiles_and_runs(generated_cpp):
    sys.path.insert(0, ROOT_DIR)
    from scripts.build_cpp import build_and_run

    output = build_and_run(str(generated_cpp["path"]))
    assert "Latency:" in output
    assert "Total spikes:" in output
