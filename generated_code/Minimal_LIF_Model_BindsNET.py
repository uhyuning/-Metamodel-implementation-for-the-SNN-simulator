# 생성된 파일: Minimal_LIF_Model.py
# 대상 시뮬레이터: BindsNET
import torch
from bindsnet.network import Network
from bindsnet.network.nodes import LIF, Input 
from bindsnet.network.topology import Connection
import numpy as np

# ----------------- 1. 네트워크 및 시간 설정 -----------------
network = Network(dt=100)

# ----------------- 2. 레이어 생성 -----------------


# Python 코드:
L1_Input = Input(n=784)

network.add_layer(L1_Input, name='L1_Input')


# Python 코드:
L2_LIF = LIF(n=100, v_th=1.0, tau_mem=10.0)

network.add_layer(L2_LIF, name='L2_LIF')


# ----------------- 3. 연결 설정 -----------------

connection_L1_Input_L2_LIF = Connection(
    source=L1_Input, 
    target=L2_LIF, 
    nu=[1e-4, 1e-2] # 임시 학습률
)
network.add_connection(connection_L1_Input_L2_LIF, source=L1_Input, target=L2_LIF)


# ----------------- 4. 시뮬레이션 예시 -----------------
if __name__ == '__main__':
    # 임의의 입력 데이터 (784 차원) 생성
    input_data = torch.from_numpy(np.random.rand(1, 784)).float() 
    
    inputs = {'L1_Input': input_data}
    print(f"Running BindsNET simulation for 100 steps...")
    network.run(inputs=inputs, time=100)
    print("Simulation finished.")