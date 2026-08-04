import os
import sys
from jinja2 import Environment, FileSystemLoader

from src.lif_dynamics import normalize_lif_layer

# 시뮬레이터 식별자 -> (템플릿 파일, 출력 확장자)
TEMPLATE_CONFIG = {
    'BindsNET': ('bindsnet_template.j2', 'py'),
    'snnTorch': ('snntorch_template.j2', 'py'),
    'SpikingJelly': ('spikingjelly_template.j2', 'py'),
    'CppSIMD': ('cpp_simd_template.j2', 'cpp'),
}

# 표준(캐노니컬) 서로게이트 함수 이름 -> 프레임워크별 실제 토큰.
# snnTorch는 소문자 함수명(atan), SpikingJelly는 PascalCase 클래스명(ATan)을 쓰는 등
# 이름 표기가 갈려서, LIF dynamics(src/lif_dynamics.py)처럼 여기서도 하나의 캐노니컬
# 이름을 기준으로 각 백엔드 토큰을 파생시킨다.
SURROGATE_FUNCTION_MAP = {
    'atan': {'snnTorch': 'atan', 'SpikingJelly': 'ATan'},
    'sigmoid': {'snnTorch': 'sigmoid', 'SpikingJelly': 'Sigmoid'},
    'fast_sigmoid': {'snnTorch': 'fast_sigmoid', 'SpikingJelly': 'Sigmoid'},
}

# 표준 step_mode -> 프레임워크별 토큰. SpikingJelly는 'm'/'s' 리터럴만 허용.
STEP_MODE_MAP = {
    'multi_step': {'snnTorch': 'multi_step', 'SpikingJelly': 'm'},
    'single_step': {'snnTorch': 'single_step', 'SpikingJelly': 's'},
}
# 레거시 값(이미 프레임워크 고유 토큰으로 쓰인 예제)을 캐노니컬로 역매핑
_STEP_MODE_ALIASES = {'m': 'multi_step', 's': 'single_step'}


def _resolve_token(value, target_simulator, canonical_map, aliases=None):
    """캐노니컬 값(또는 이미 프레임워크 토큰인 레거시 값)을 target_simulator에 맞는
    실제 토큰으로 변환한다. 매핑에 없는 값은 그대로 통과시켜(레거시 호환) 기존
    예제가 계속 동작하게 한다."""
    if value is None:
        return value
    canonical = aliases.get(value, value) if aliases else value
    entry = canonical_map.get(canonical)
    if entry and target_simulator in entry:
        return entry[target_simulator]
    return value


def _build_cpp_ops(layers):
    """C++ 백엔드용: linear/LIFNode/IFNode 레이어 시퀀스를,
    각 연산이 읽고 쓰는 버퍼 이름과 크기가 명시된 연산 목록으로 변환한다.

    snnTorch/SpikingJelly 템플릿은 파이썬 텐서가 알아서 shape을 실어 나르지만,
    C++에서는 각 버퍼의 크기를 코드 생성 시점에 직접 계산해 둬야 하므로 이 전처리가 필요하다.
    """
    ops = []
    prev_buf = "x_in"
    current_width = None
    buf_idx = 0

    for layer in layers:
        if layer['type'] == 'linear':
            out_buf = f"buf{buf_idx}"
            buf_idx += 1
            ops.append({
                "kind": "linear",
                "in_buf": prev_buf,
                "out_buf": out_buf,
                "in_features": layer['in_features'],
                "out_features": layer['out_features'],
                "weight_name": f"W{len(ops)}",
                "bias_name": f"b{len(ops)}",
            })
            current_width = layer['out_features']
            prev_buf = out_buf
        elif layer['type'] in ('LIFNode', 'IFNode'):
            out_buf = f"buf{buf_idx}"
            v_buf = f"v{buf_idx}"
            buf_idx += 1
            ops.append({
                "kind": "neuron",
                "in_buf": prev_buf,
                "out_buf": out_buf,
                "v_buf": v_buf,
                "width": current_width,
                "decay": layer['decay'],
                "v_threshold": layer['v_threshold'],
                "v_reset": layer['v_reset'],
                "reset_mechanism": layer['reset_mechanism'],
            })
            prev_buf = out_buf

    return ops


def generate_snn_code(model_spec: dict, output_dir: str = 'generated_code'):
    """
    Metamodel 명세를 바탕으로 SNN 시뮬레이터 코드를 생성합니다.
    하드웨어 제약 조건(Hardware-aware)에 따른 최적화 로직이 포함되어 있습니다.
    """
    
    # 1. 경로 설정 (기존 코드 유지)
    current_file_path = os.path.abspath(__file__)
    src_dir = os.path.dirname(current_file_path)
    base_dir = os.path.dirname(src_dir)
    template_dir = os.path.join(base_dir, 'templates')
    
    print(f"[DEBUG] Base Directory: {base_dir}")
    print(f"[DEBUG] Template Directory: {template_dir}")

    # 2. Jinja2 환경 설정 (기존 코드 유지)
    if not os.path.exists(template_dir):
        raise FileNotFoundError(f"Template directory not found at: {template_dir}")

    file_loader = FileSystemLoader(template_dir, encoding='utf-8') 
    env = Environment(loader=file_loader)
    
    # 3. 사용할 템플릿 파일 결정
    simulator = model_spec.get('target_simulator')

    if simulator not in TEMPLATE_CONFIG:
        raise ValueError(f"Unsupported simulator: {simulator}")
    template_name, output_ext = TEMPLATE_CONFIG[simulator]

    try:
        template = env.get_template(template_name)
    except Exception as e:
        print(f"\n[ERROR] Template Load Failed: Could not find '{template_name}' in {template_dir}.")
        raise e

    # --- [신규 추가] 3.5 하드웨어 제약 조건에 따른 모델 전처리 (Hardware-aware Optimization) ---
    hw_constraints = model_spec.get('hardware_constraints', {})
    power_mode = model_spec.get('power_mode', hw_constraints.get('power_mode', 'normal'))

    if power_mode == "low_energy":
        print(f"⚠️ [Optimization] Low Energy mode detected. Reconfiguring layers for efficiency...")
        for layer in model_spec['layers']:
            # [입문자 가이드] 연산량이 많은 LIFNode를 가장 단순한 IFNode로 자동 교체합니다.
            if layer['type'] == "LIFNode":
                layer['type'] = "IFNode"
                # 템플릿에서 주석으로 출력될 수 있도록 설명을 덧붙입니다.
                layer['hw_note'] = "Optimized for Low Energy (Switched from LIF to IF)"
                # 타입이 바뀌었으므로 표준 LIF 다이내믹스도 IF(무누설)에 맞게 재계산합니다.
                normalize_lif_layer(layer)
    # ---------------------------------------------------------------------------------

    # --- 서로게이트 함수/step_mode를 캐노니컬 값에서 백엔드별 토큰으로 변환 ---
    # (LIF dynamics와 같은 이유: 사용자는 시뮬레이터별 API 표기를 몰라도 되게 함)
    for layer in model_spec['layers']:
        if 'surrogate_function' in layer:
            layer['surrogate_function'] = _resolve_token(
                layer['surrogate_function'], simulator, SURROGATE_FUNCTION_MAP
            )
    step_mode = _resolve_token(
        model_spec.get('step_mode', 'multi_step'), simulator, STEP_MODE_MAP, _STEP_MODE_ALIASES
    )

    # 4. 템플릿 렌더링
    backend = model_spec.get('backend', 'torch')
    render_kwargs = dict(
        model_name=model_spec['model_name'],
        time_steps=model_spec['time_steps'],
        layers=model_spec['layers'],
        connections=model_spec['connections'],
        backend=backend,
        backend_type=backend,
        step_mode=step_mode,
        # 하드웨어 정보도 필요시 템플릿에서 쓸 수 있게 넘겨줍니다.
        hardware_constraints=hw_constraints,
        power_mode=power_mode
    )

    if simulator == 'CppSIMD':
        # C++ 백엔드는 텐서가 shape을 실어 나르지 않으므로, 버퍼 이름/크기를 코드 생성 시점에 미리 계산해 넘긴다.
        ops = _build_cpp_ops(model_spec['layers'])
        first_linear = next(l for l in model_spec['layers'] if l['type'] == 'linear')
        render_kwargs.update(
            ops=ops,
            input_features=first_linear['in_features'],
            output_width=ops[-1]['width'] if ops[-1]['kind'] == 'neuron' else ops[-1]['out_features'],
        )

    output_code = template.render(**render_kwargs)

    # 5. 출력 디렉토리 생성 및 파일 저장
    output_path = os.path.join(base_dir, output_dir)
    output_filename = f"{model_spec['model_name']}_{simulator}.{output_ext}"
    output_filepath = os.path.join(output_path, output_filename)

    os.makedirs(output_path, exist_ok=True) 
    
    with open(output_filepath, 'w', encoding='utf-8') as f:
        f.write(output_code)

    print(f"\n✅ [{simulator}] Code generation successful!")
    print(f"   Saved to: {output_filepath}")