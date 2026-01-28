import os
import sys

# 1. 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

try:
    from src.parser import load_and_validate_metamodel
    from src.codegen import generate_snn_code
except ImportError as e:
    print(f"[ERROR] Import failure: {e}")
    sys.exit(1)

def main():
    print("\n" + "="*40)
    print("   SNN Metamodel Code Generator")
    print("="*40)
    
    # 2. examples 폴더 내의 json 파일 목록 가져오기
    json_dir = os.path.join(BASE_DIR, 'examples')
    
    if not os.path.exists(json_dir):
        print(f"[ERROR] '{json_dir}' 폴더가 존재하지 않습니다.")
        return

    # .json 확장자만 골라내기
    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]

    if not json_files:
        print("[INFO] 'examples' 폴더에 사용할 수 있는 JSON 파일이 없습니다.")
        return

    # 3. 파일 목록 보여주기
    print("\n사용 가능한 모델 명세 목록:")
    for idx, filename in enumerate(json_files, 1):
        print(f" [{idx}] {filename}")
    
    # 4. 사용자 입력 받기
    try:
        choice = input(f"\n생성할 파일 번호를 입력하세요 (1-{len(json_files)}): ").strip()
        
        # 입력값이 숫자인지 확인
        if not choice.isdigit():
            raise ValueError("숫자만 입력 가능합니다.")
        
        selected_idx = int(choice) - 1
        
        if 0 <= selected_idx < len(json_files):
            selected_file = json_files[selected_idx]
            json_path = os.path.join(json_dir, selected_file)
        else:
            raise IndexError("범위를 벗어난 번호입니다.")

    except (ValueError, IndexError) as e:
        print(f"[ERROR] 잘못된 입력입니다: {e}")
        return

    # 5. 코드 생성 진행
    try:
        print(f"\n[*] '{selected_file}' 파일을 읽어오는 중...")
        model_spec = load_and_validate_metamodel(json_path)
        generate_snn_code(model_spec)
        print("\n" + "-"*40)
        print("✅ 모든 작업이 완료되었습니다!")
        print("-"*40)
        
    except Exception as e:
        print(f"\n[FATAL ERROR]: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()