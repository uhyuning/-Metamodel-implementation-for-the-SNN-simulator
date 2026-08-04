"""
시뮬레이터 독립적인 LIF(Leaky Integrate-and-Fire) 다이내믹스 표준 정의.

이산 시간 LIF 막전위 갱신은 다음 미분방정식의 오일러 이산화로부터 나온다:

    tau_mem * dV/dt = -(V - V_reset) + I(t)
    => V[t+1] = decay * V[t] + (1 - decay) * V_reset + I[t],  decay = exp(-dt / tau_mem)

이 모듈은 그 decay 계수를 계산하는 단일 지점이다. snnTorch(beta), BindsNET(tc_decay),
SpikingJelly(tau) 등 프레임워크마다 이름이 다른 "누설 계수" 파라미터는 전부 여기서
계산한 동일한 tau_mem/decay 값에서 파생되며, 각 codegen 템플릿은 자기 프레임워크의
파라미터 이름으로만 값을 옮겨 담는다 — 다이내믹스 자체를 각자 다시 정의하지 않는다.

IFNode는 누설이 없는(decay=1.0) LIF의 극한 케이스로 취급한다.
"""
import math

DEFAULT_TAU_MEM = 10.0
DEFAULT_V_THRESHOLD = 1.0
DEFAULT_V_RESET = 0.0
DEFAULT_RESET_MECHANISM = "subtract"  # "subtract"(soft reset) 또는 "zero"(hard reset)


def compute_decay(tau_mem: float, dt: float = 1.0) -> float:
    """막전위 누설 계수 decay = exp(-dt / tau_mem)."""
    return math.exp(-dt / tau_mem)


def normalize_lif_layer(layer: dict, dt: float = 1.0) -> dict:
    """레이어 스펙에 표준 LIF 다이내믹스 필드(tau_mem, decay, v_threshold, v_reset,
    reset_mechanism)를 채워 넣는다. layer를 in-place로 수정하고 그대로 반환한다.

    - `type == 'IFNode'`: 누설 없음(decay=1.0)으로 강제.
    - 레거시 `beta` 필드만 있고 `tau_mem`이 없으면, tau_mem = -dt / ln(beta)로 역산해
      과거 예제 JSON이 표현하려던 값을 최대한 존중한다.
    """
    if layer.get("type") == "IFNode":
        layer["tau_mem"] = math.inf
        layer["decay"] = 1.0
    else:
        if "tau_mem" not in layer and "beta" in layer and 0 < layer["beta"] < 1:
            layer["tau_mem"] = -dt / math.log(layer["beta"])
        layer.setdefault("tau_mem", DEFAULT_TAU_MEM)
        layer["decay"] = compute_decay(layer["tau_mem"], dt)

    layer.setdefault("v_threshold", DEFAULT_V_THRESHOLD)
    layer.setdefault("v_reset", DEFAULT_V_RESET)
    layer.setdefault("reset_mechanism", DEFAULT_RESET_MECHANISM)
    return layer
