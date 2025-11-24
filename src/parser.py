import json
import os
from pprint import pprint

def load_and_validate_metamodel(filepath: str) -> dict:
    """
    JSON 파일을 로드하고, 기본적인 구조 유효성을 검사합니다.
    
    Args:
        filepath: 로드할 JSON 파일의 경로.
    Returns:
        파싱된 Metamodel 딕셔너리.
    Raises:
        FileNotFoundError, json.JSONDecodeError, ValueError
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Metamodel file not found at: {filepath}")

    with open(filepath, 'r', encoding='utf-8') as f:
        try:
            metamodel_spec = json.load(f)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"JSON decoding error: {e}", e.doc, e.pos)
    
    # --- 1차 필수 필드 검증 ---
    required_keys = ['model_name', 'target_simulator', 'time_steps', 'layers', 'connections']
    if not all(key in metamodel_spec for key in required_keys):
        missing_keys = [key for key in required_keys if key not in metamodel_spec]
        raise ValueError(f"Metamodel JSON is missing required keys: {', '.join(missing_keys)}")
        
    # --- 2차 로직 검증 (TODO) ---
    # 실제 구현 단계에서는 layers와 connections 배열의 내용을 상세히 검사해야 합니다.
    
    print(f"Successfully loaded and validated model: {metamodel_spec.get('model_name')}")
    return metamodel_spec

if __name__ == "__main__":
    # 테스트를 위해 examples 폴더의 JSON 파일을 로드
    example_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'examples', 'minimal_lif_model.json')
    try:
        model_data = load_and_validate_metamodel(example_path)
        pprint(model_data)
    except Exception as e:
        print(f"Error during parsing: {e}")
