import os
import sys
from jinja2 import Environment, FileSystemLoader

def generate_snn_code(model_spec: dict, output_dir: str = 'generated_code'):
    """
    Metamodel 명세를 바탕으로 SNN 시뮬레이터 코드를 생성합니다.
    """
    
    # 1. 경로 설정 (더 안전한 방식)
    # 현재 파일(codegen.py)의 절대 경로를 기준으로 프로젝트 루트를 잡습니다.
    current_file_path = os.path.abspath(__file__)
    src_dir = os.path.dirname(current_file_path)
    base_dir = os.path.dirname(src_dir)  # 프로젝트 최상위 루트 (SNN_MetaModel_Project)
    
    template_dir = os.path.join(base_dir, 'templates')
    
    # 디버깅용 경로 출력 (문제가 생기면 이 경로를 확인하세요)
    print(f"[DEBUG] Base Directory: {base_dir}")
    print(f"[DEBUG] Template Directory: {template_dir}")

    # 2. Jinja2 환경 설정
    if not os.path.exists(template_dir):
        raise FileNotFoundError(f"Template directory not found at: {template_dir}")

    file_loader = FileSystemLoader(template_dir, encoding='utf-8') 
    env = Environment(loader=file_loader)
    
    # 3. 사용할 템플릿 파일 결정
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
        
    # 4. 템플릿 렌더링 (SpikingJelly 파라미터 포함)
    output_code = template.render(
        model_name=model_spec['model_name'],
        time_steps=model_spec['time_steps'],
        layers=model_spec['layers'],
        connections=model_spec['connections'],
        backend=model_spec.get('backend', 'torch'),
        step_mode=model_spec.get('step_mode', 'multi_step')
    )
    
    # 5. 출력 디렉토리 생성 및 파일 저장
    output_path = os.path.join(base_dir, output_dir)
    output_filename = f"{model_spec['model_name']}_{simulator}.py"
    output_filepath = os.path.join(output_path, output_filename)

    # 폴더 생성
    os.makedirs(output_path, exist_ok=True) 
    
    # 파일 저장
    with open(output_filepath, 'w', encoding='utf-8') as f:
        f.write(output_code)

    print(f"\n✅ [{simulator}] Code generation successful!")
    print(f"   Saved to: {output_filepath}")

if __name__ == "__main__":
    # 단독 테스트용 (필요시 사용)
    pass