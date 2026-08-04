"""
생성된 코드의 구조적 정합성을 검증하는 스모크 테스트.

torch/snntorch/spikingjelly/bindsnet 등 무거운 프레임워크를 설치하지 않고도
템플릿 렌더링이 메타모델 스펙과 일치하는지(문법 유효성 + 뉴런 타입/개수)를
빠르게 확인하기 위한 정적(static) 검증. 과거 두 차례 실제로 발생했던 버그
(power_mode 필드 위치 불일치, snnTorch IFNode 분기 누락으로 뉴런 레이어가
통째로 사라지는 문제)를 재발 시 즉시 잡아내는 것이 목적.
"""
import ast
import copy
import json
import os
import re

import pytest

from src.codegen import generate_snn_code
from src.lif_dynamics import compute_decay
from src.parser import load_and_validate_metamodel

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLES_DIR = os.path.join(ROOT_DIR, "examples")
# 이 모듈은 Python 템플릿 백엔드(snnTorch/SpikingJelly/BindsNET)만 다룬다.
# C++/SIMD 백엔드는 출력 확장자·검증 방식이 달라 tests/test_cpp_backend.py에서 별도로 다룬다.
EXAMPLE_FILES = sorted(
    f for f in os.listdir(EXAMPLES_DIR)
    if f.endswith(".json") and not f.startswith("cpp_simd_")
)

# 시뮬레이터별 뉴런 클래스 생성자 이름: (표준 LIF 표기, low_energy IF 표기)
NEURON_PATTERNS = {
    "snnTorch": (r"snn\.Leaky\(", r"snn\.Lapicque\("),
    "SpikingJelly": (r"neuron\.LIFNode\(", r"neuron\.IFNode\("),
    "BindsNET": (r"\bLIFNodes\(", r"\bIFNodes\("),
}


@pytest.fixture(params=EXAMPLE_FILES)
def generated(request, tmp_path):
    config_name = request.param
    spec = load_and_validate_metamodel(os.path.join(EXAMPLES_DIR, config_name))
    simulator = spec["target_simulator"]
    power_mode = spec.get("power_mode", spec.get("hardware_constraints", {}).get("power_mode", "normal"))
    # 일부 예제(BindsNET/SpikingJelly의 low_power 설정)는 IFNode를 직접 명시하고,
    # 다른 예제(snnTorch)는 LIFNode를 명시하고 codegen이 low_energy 시 IFNode로 스왑한다.
    # 따라서 원본 스펙 시점에는 두 타입 모두 "뉴런 레이어"로 취급해 개수를 센다.
    neuron_layer_count = sum(1 for l in spec["layers"] if l["type"] in ("LIFNode", "IFNode"))

    generate_snn_code(spec, output_dir=str(tmp_path))

    output_filename = f"{spec['model_name']}_{simulator}.py"
    with open(tmp_path / output_filename, "r", encoding="utf-8") as f:
        source = f.read()

    return {
        "config_name": config_name,
        "source": source,
        "simulator": simulator,
        "power_mode": power_mode,
        "neuron_layer_count": neuron_layer_count,
    }


def test_generated_code_is_valid_python(generated):
    ast.parse(generated["source"])


def test_neuron_layer_count_and_variant_match_spec(generated):
    standard_pattern, low_power_pattern = NEURON_PATTERNS[generated["simulator"]]
    source = generated["source"]
    expected_count = generated["neuron_layer_count"]

    if generated["power_mode"] == "low_energy":
        active_pattern, inactive_pattern = low_power_pattern, standard_pattern
    else:
        active_pattern, inactive_pattern = standard_pattern, low_power_pattern

    active_count = len(re.findall(active_pattern, source))
    inactive_count = len(re.findall(inactive_pattern, source))

    assert active_count == expected_count, (
        f"{generated['config_name']}: expected {expected_count} "
        f"'{active_pattern}' neuron layer(s) for power_mode="
        f"'{generated['power_mode']}', found {active_count}"
    )
    assert inactive_count == 0, (
        f"{generated['config_name']}: found {inactive_count} unexpected "
        f"'{inactive_pattern}' neuron layer(s) for power_mode="
        f"'{generated['power_mode']}'"
    )


# 동일한 tau_mem/v_threshold/v_reset을 세 시뮬레이터에 각각 명시적으로 지정한 최소 스펙.
# 표준화 주장의 핵심: 이 값들이 각 백엔드의 고유 파라미터 이름(beta/tau/tc_decay)으로
# 옮겨지되, 실제 수치는 src/lif_dynamics.compute_decay() 하나에서 파생되어 일치해야 한다.
_BASE_SPEC = {
    "time_steps": 10,
    "power_mode": "normal",
    "hardware_constraints": {},
    "layers": [
        {"type": "linear", "in_features": 4, "out_features": 2},
        {
            "type": "LIFNode",
            "surrogate_function": "atan",
            "out_features": 2,  # BindsNET 템플릿이 n={{ l.out_features }}로 직접 참조함
            "tau_mem": 7.5,
            "v_threshold": 0.8,
            "v_reset": 0.1,
        },
    ],
    "connections": [{"source": "input", "target": "layer_1"}],
    "backend": "torch",
    "step_mode": "multi_step",
}

_SIMULATOR_CASES = {
    "snnTorch": ("SnnTorchDynamicsCheck", r"beta=([\d.]+),\s*threshold=([\d.]+)"),
    "SpikingJelly": ("SpikingJellyDynamicsCheck", r"tau=([\d.]+), v_threshold=([\d.]+)"),
    "BindsNET": ("BindsNetDynamicsCheck", r"thresh=([\d.]+),\s*rest=([\d.]+),\s*tc_decay=([\d.]+)"),
}


@pytest.mark.parametrize("simulator", sorted(_SIMULATOR_CASES))
def test_lif_dynamics_are_identical_across_backends(simulator, tmp_path):
    model_name, pattern = _SIMULATOR_CASES[simulator]
    spec = copy.deepcopy(_BASE_SPEC)
    spec["model_name"] = model_name
    spec["target_simulator"] = simulator
    if simulator == "BindsNET":
        # BindsNET 템플릿은 첫 레이어를 Input으로 취급하므로 linear 레이어가 필요 없다.
        spec["layers"] = [{"type": "input", "in_features": 4, "out_features": 4}, spec["layers"][1]]

    # 실제 파이프라인(main.py)과 동일하게 parser의 load_and_validate_metamodel을 거치게 해서
    # LIF 다이내믹스 정규화(normalize_lif_layer)가 codegen 이전에 적용되도록 한다.
    config_path = tmp_path / f"{model_name}.json"
    config_path.write_text(json.dumps(spec), encoding="utf-8")
    spec = load_and_validate_metamodel(str(config_path))

    generate_snn_code(spec, output_dir=str(tmp_path))

    output_filename = f"{model_name}_{simulator}.py"
    with open(tmp_path / output_filename, "r", encoding="utf-8") as f:
        source = f.read()

    expected_decay = compute_decay(tau_mem=7.5, dt=1.0)
    match = re.search(pattern, source)
    assert match, f"{simulator}: expected pattern '{pattern}' not found in generated code"

    if simulator == "BindsNET":
        thresh, rest, tc_decay = (float(g) for g in match.groups())
        assert thresh == pytest.approx(0.8)
        assert rest == pytest.approx(0.1)
        assert tc_decay == pytest.approx(7.5)
    else:
        decay_or_tau, threshold = (float(g) for g in match.groups())
        assert threshold == pytest.approx(0.8)
        if simulator == "snnTorch":
            assert decay_or_tau == pytest.approx(expected_decay)
        else:  # SpikingJelly는 decay 대신 tau_mem을 직접 받는다
            assert decay_or_tau == pytest.approx(7.5)


# 캐노니컬 surrogate_function="atan"/step_mode="multi_step" 하나로 snnTorch/SpikingJelly
# 양쪽 모두 올바른 프레임워크 고유 토큰(소문자 함수명 vs PascalCase 클래스명, 'm'/'s')으로
# 번역되는지 확인한다 (src/codegen.py의 SURROGATE_FUNCTION_MAP/STEP_MODE_MAP).
_TOKEN_TRANSLATION_CASES = {
    "snnTorch": ("SnnTorchTokenCheck", r"surrogate\.atan\(\)"),
    "SpikingJelly": ("SpikingJellyTokenCheck", r"surrogate\.ATan\(\).*?step_mode='m'"),
}


@pytest.mark.parametrize("simulator", sorted(_TOKEN_TRANSLATION_CASES))
def test_surrogate_and_step_mode_translate_from_canonical_value(simulator, tmp_path):
    model_name, pattern = _TOKEN_TRANSLATION_CASES[simulator]
    spec = copy.deepcopy(_BASE_SPEC)
    spec["model_name"] = model_name
    spec["target_simulator"] = simulator
    # _BASE_SPEC은 이미 캐노니컬 값("atan"/"multi_step")을 쓴다 — 시뮬레이터별로
    # 다른 값을 넣지 않아도 codegen이 알아서 올바른 토큰으로 번역해야 한다.

    config_path = tmp_path / f"{model_name}.json"
    config_path.write_text(json.dumps(spec), encoding="utf-8")
    loaded = load_and_validate_metamodel(str(config_path))
    generate_snn_code(loaded, output_dir=str(tmp_path))

    with open(tmp_path / f"{model_name}_{simulator}.py", "r", encoding="utf-8") as f:
        source = f.read()

    assert re.search(pattern, source, re.DOTALL), (
        f"{simulator}: canonical 'atan'/'multi_step' did not translate as expected "
        f"(pattern '{pattern}' not found)"
    )
