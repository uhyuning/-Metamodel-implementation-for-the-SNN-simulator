import os
from jinja2 import Environment, FileSystemLoader

def generate_snn_code(model_spec: dict, output_dir: str = 'generated_code'):
    """
    Metamodel 명세를 바탕으로 SNN 시뮬레이터 코드를 생성합니다.
    """
    
    # 1. Jinja2 환경 설정
    # __file__의 부모 디렉토리의 부모 디렉토리 (프로젝트 루트)에서 templates 폴더를 찾습니다.
    template_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'templates')
    file_loader = FileSystemLoader(template_dir) 
    env = Environment(loader=file_loader)
    
    # 2. 사용할 템플릿 파일 결정
    simulator = model_spec.get('target_simulator')
    
    if simulator == 'BindsNET':
        template_name = 'bindsnet_template.j2'
    elif simulator == 'snnTorch':
        template_name = 'snntorch_template.j2'
    else:
        # 이전에 파서에서 걸러지지 않은 경우를 대비한 예외 처리
        raise ValueError(f"Unsupported simulator: {simulator}")

    try:
        template = env.get_template(template_name)
    except Exception as e:
        print(f"\n[ERROR] Template Load Failed: Could not find or load '{template_name}'.")
        raise e
        
    # 3. 템플릿 렌더링 (코드 생성)
    output_code = template.render(
        model_name=model_spec['model_name'],
        time_steps=model_spec['time_steps'],
        layers=model_spec['layers'],
        connections=model_spec['connections']
    )
    
    # 4. 출력 디렉토리 생성 및 파일 저장
    
    # 4-1. 출력 경로 구성
    output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), output_dir)
    output_filename = f"{model_spec['model_name']}_{simulator}.py"
    output_filepath = os.path.join(output_path, output_filename)

    # <<<<<<<< 디버깅용 코드: 경로 확인 >>>>>>>>
    print(f"\n[DEBUG] Target Output Dir (Absolute): {os.path.abspath(output_path)}")
    print(f"[DEBUG] Target Filepath (Absolute): {os.path.abspath(output_filepath)}")
    # <<<<<<<< 디버깅용 코드 끝 >>>>>>>>
    
    try:
        # 디렉토리 생성 시도 (이미 있다면 무시)
        os.makedirs(output_path, exist_ok=True) 
        print(f"[DEBUG] Directory creation attempted: {output_path}")
    except OSError as e:
        # 권한 문제 등 OS 레벨 오류가 발생했을 때
        print(f"\n[FATAL ERROR] Failed to create directory. Check OS permissions or path validity.")
        raise e
        
    try:
        # 파일 저장 시도
        with open(output_filepath, 'w', encoding='utf-8') as f:
            f.write(output_code)
    except Exception as e:
        print(f"\n[FATAL ERROR] Failed to write file: {output_filepath}. Check file path/name.")
        raise e

    print(f"\n✅ Code generation successful!")
    print(f"   Generated file saved to: {output_filepath}")


if __name__ == "__main__":
    # 이 파일 단독 실행 시 테스트 로직을 추가할 수 있습니다. (현재는 main.py를 통해 실행)
    pass