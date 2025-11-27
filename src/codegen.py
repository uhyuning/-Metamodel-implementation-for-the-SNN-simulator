import os
import sys
from jinja2 import Environment, FileSystemLoader

def generate_snn_code(model_spec: dict, output_dir: str = 'generated_code'):
    """
    Metamodel 명세를 바탕으로 SNN 시뮬레이터 코드를 생성합니다.
    (Generates SNN simulator code based on the Metamodel specification.)
    """
    
    # 1. Jinja2 환경 설정
    # Get the project root directory
    base_dir = os.path.dirname(os.path.dirname(__file__))
    template_dir = os.path.join(base_dir, 'templates')
    
    # 
    # Explicitly set encoding to 'utf-8' for the file loader to handle non-ASCII paths/templates.
    file_loader = FileSystemLoader(template_dir, encoding='utf-8') 
    env = Environment(loader=file_loader)
    
    # 2. 사용할 템플릿 파일 결정
    simulator = model_spec.get('target_simulator')
    
    if simulator == 'BindsNET':
        template_name = 'bindsnet_template.j2'
    elif simulator == 'snnTorch':
        template_name = 'snntorch_template.j2'
    else:
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
    output_path = os.path.join(base_dir, output_dir)
    output_filename = f"{model_spec['model_name']}_{simulator}.py"
    output_filepath = os.path.join(output_path, output_filename)

    # <<<<<<<< DEBUG: 렌더링된 코드 내용 확인 >>>>>>>>
    # Check if the output_code is empty or corrupted.
    code_length = len(output_code)
    print(f"[DEBUG] Rendered Code Length: {code_length} characters")
    
    if code_length > 0:
        # Print first 200 characters to visually check for corrupted characters
        print(f"[DEBUG] RENDERED OUTPUT PREVIEW (First 200 chars):\n---START---\n{output_code[:200].strip()}\n---END---")
    else:
        # Critical warning if the code is empty.
        print("\n[CRITICAL DEBUG] RENDERED CODE IS EMPTY. Check template content or variables.")
    # <<<<<<<< DEBUG END >>>>>>>>
    
    # 4-2. 디렉토리 생성
    try:
        os.makedirs(output_path, exist_ok=True) 
        print(f"[DEBUG] Directory creation attempted: {output_path}")
    except OSError as e:
        print(f"\n[FATAL ERROR] Failed to create directory. Check OS permissions or path validity.")
        raise e
        
    # 4-3. 파일 저장
    try:
        # Strong UTF-8 encoding is specified to resolve potential Windows encoding conflicts.
        with open(output_filepath, 'w', encoding='utf-8') as f:
            f.write(output_code)
            
    except Exception as e:
        print(f"\n[FATAL ERROR] Failed to write file: {output_filepath}. Details: {e}")
        raise e

    print(f"\n✅ Code generation successful!")
    print(f"   Generated file saved to: {output_filepath}")


if __name__ == "__main__":
    pass