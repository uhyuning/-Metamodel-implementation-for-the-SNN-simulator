import torch
from bindsnet.network import Network
from bindsnet.network.nodes import LIFNodes, IFNodes, Input
from bindsnet.network.topology import Connection
from bindsnet.learning import PostPre, NoOp
from bindsnet.network.monitors import Monitor

"""
======================================================================
SNN Meta-Model Framework (BindsNET Expert Ver. 2.0)
[Hardware-Aware Configuration]
- Mode: low_energy | Target: edge_neuromorphic_chip
- Precision: FP16 (Half)
======================================================================
"""

def create_BindsNetLowPowerModel(dt=1.0, is_learning=True):
    network = Network(dt=dt)
    power_mode = "low_energy"
    
    # [Innovation] 학습 상태에 따른 가중치 업데이트 규칙 동적 할당
    learning_rule = PostPre if is_learning else NoOp

    # 1. Input Layer 정의 (SNN Spike Encoding 준비)
    input_layer = Input(n=784, shape=(784,), sum_input=True)
    network.add_layer(input_layer, name="input_layer")

    prev_name = "input_layer"
    
    
    
    curr_name = "layer_1"
    
    # 2. [Hardware-Aware] 전력 모드에 따른 뉴런 모델 최적화 선택
    
    # Low-Energy: 정수 기반 연산에 유리한 IF 모델 (No Leakage)
    curr_layer = IFNodes(n=128, sum_input=True)
    
    
    network.add_layer(curr_layer, name=curr_name)

    # 3. Connection 정의 (시냅스 연결 및 학습 규칙 적용)
    network.add_connection(
        Connection(
            source=network.layers[prev_name], 
            target=network.layers[curr_name], 
            update_rule=learning_rule
        ),
        source=prev_name, target=curr_name
    )
    
    # 4. [Innovation] 연구용 스파이크 분석 모니터 추가
    network.add_monitor(
        Monitor(network.layers[curr_name], state_vars=("s", "v"), time=50),
        name=f"{curr_name}_monitor"
    )

    prev_name = curr_name
    
    
    
    curr_name = "layer_2"
    
    # 2. [Hardware-Aware] 전력 모드에 따른 뉴런 모델 최적화 선택
    
    # Low-Energy: 정수 기반 연산에 유리한 IF 모델 (No Leakage)
    curr_layer = IFNodes(n=10, sum_input=True)
    
    
    network.add_layer(curr_layer, name=curr_name)

    # 3. Connection 정의 (시냅스 연결 및 학습 규칙 적용)
    network.add_connection(
        Connection(
            source=network.layers[prev_name], 
            target=network.layers[curr_name], 
            update_rule=learning_rule
        ),
        source=prev_name, target=curr_name
    )
    
    # 4. [Innovation] 연구용 스파이크 분석 모니터 추가
    network.add_monitor(
        Monitor(network.layers[curr_name], state_vars=("s", "v"), time=50),
        name=f"{curr_name}_monitor"
    )

    prev_name = curr_name
    
    

    return network

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_BindsNetLowPowerModel(is_learning=True)
    model.to(device)

    # 5. [Optimization] 저전력 가동을 위한 FP16 양자화 전략
    
    # BindsNET 텐서 내부 최적화 (Memory-Efficient Inference)
    for p in model.parameters():
        p.data = p.data.half()
    for layer in model.layers.values():
        if hasattr(layer, 'v'): layer.v = layer.v.half()
    print(">>> [System] FP16 Quantization Applied for Energy Efficiency.")
    

    # 시뮬레이션 데이터 준비 (Time, Batch, Features)
    data = torch.randn(50, 1, 784).to(device)
    data = data.half()

    # 시뮬레이션 실행 (Research Execution)
    model.run(inputs={"input_layer": data}, time=50)

    # 결과 요약 출력
    print(f"\n==================================================")
    print(f"BindsNET Experiment Summary: 'BindsNetLowPowerModel'")
    print(f" - Active Mode: low_energy")
    print(f" - Computed Layers: {list(model.layers.keys())}")
    print(f" - Last Layer Spikes: {torch.sum(model.monitors[f'{prev_name}_monitor'].get('s')).item()}")
    print(f"==================================================")