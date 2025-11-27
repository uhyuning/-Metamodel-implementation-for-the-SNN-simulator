# main.py 파일 내용
import os
import sys

# src 디렉토리를 Python 경로에 추가하여 모듈을 임포트할 수 있게 함
sys.path.append(os.path.join(os.path.dirname(__file__), 'src')) 

from src.parser import load_and_validate_metamodel
from src.codegen import generate_snn_code

def main():
    print("--- SNN Metamodel Code Generator Started ---")
    
    # 1. Metamodel JSON 파일 경로 설정 (일단 예시 파일을 사용)
    # 실제 구현 시에는 커맨드 라인 인수를 통해 경로를 받아야 합니다.
    base_dir = os.path.dirname(__file__)
    json_path = os.path.join(base_dir, 'examples', 'minimal_lif_model.json')
    
    # 2. JSON 파일 로드 및 파싱
    try:
        model_spec = load_and_validate_metamodel(json_path)
        
        # 3. 파싱된 데이터를 사용하여 다음 단계 (코드 생성) 진행
        # print(f"Target Simulator: {model_spec['target_simulator']}")
        
        generate_snn_code(model_spec)
        
    except Exception as e:
        print(f"\n[FATAL ERROR] Project execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
    main()
