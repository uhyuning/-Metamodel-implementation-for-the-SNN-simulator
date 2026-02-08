import os
import sys
from jinja2 import Environment, FileSystemLoader

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
    
    # 3. 사용할 템플릿 파일 결정 (기존 코드 유지)
    simulator = model_spec.get('target_simulator')
    
    if simulator == 'BindsNET':
        template_name = 'bindsnet_template.j2'
    elif simulator == 'snnTorch':
        template_name = 'snntorch_template.j2'
    elif simulator == 'SpikingJelly':
        template_name = 'spikingjelly_template.j2'
    else:
        raise ValueError(f"Unsupported simulator: {simulator}")

    try:
        template = env.get_template(template_name)
    except Exception as e:
        print(f"\n[ERROR] Template Load Failed: Could not find '{template_name}' in {template_dir}.")
        raise e

    # --- [신규 추가] 3.5 하드웨어 제약 조건에 따른 모델 전처리 (Hardware-aware Optimization) ---
    hw_constraints = model_spec.get('hardware_constraints', {})
    power_mode = hw_constraints.get('power_mode', 'normal')

    if power_mode == "low_energy":
        print(f"⚠️ [Optimization] Low Energy mode detected. Reconfiguring layers for efficiency...")
        for layer in model_spec['layers']:
            # [입문자 가이드] 연산량이 많은 LIFNode를 가장 단순한 IFNode로 자동 교체합니다.
            if layer['type'] == "LIFNode":
                layer['type'] = "IFNode"
                # 템플릿에서 주석으로 출력될 수 있도록 설명을 덧붙입니다.
                layer['hw_note'] = "Optimized for Low Energy (Switched from LIF to IF)"
    # ---------------------------------------------------------------------------------
        
    # 4. 템플릿 렌더링 (기존 파라미터 유지)
    output_code = template.render(
        model_name=model_spec['model_name'],
        time_steps=model_spec['time_steps'],
        layers=model_spec['layers'],
        connections=model_spec['connections'],
        backend=model_spec.get('backend', 'torch'),
        step_mode=model_spec.get('step_mode', 'multi_step'),
        # 하드웨어 정보도 필요시 템플릿에서 쓸 수 있게 넘겨줍니다.
        hardware_constraints=hw_constraints, 
        power_mode=power_mode
    )
    
    # 5. 출력 디렉토리 생성 및 파일 저장 (기존 코드 유지)
    output_path = os.path.join(base_dir, output_dir)
    output_filename = f"{model_spec['model_name']}_{simulator}.py"
    output_filepath = os.path.join(output_path, output_filename)

    os.makedirs(output_path, exist_ok=True) 
    
    with open(output_filepath, 'w', encoding='utf-8') as f:
        f.write(output_code)

    print(f"\n✅ [{simulator}] Code generation successful!")
    print(f"   Saved to: {output_filepath}")