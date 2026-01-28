import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, layer, surrogate, functional

class MySpikingModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 메타모델의 layers 정보를 바탕으로 구성
        self.network = nn.Sequential(
            
            
            layer.Linear(784, 128),
            neuron.LIFNode(surrogate_function=surrogate.Sigmoid()),
            
            
            
            layer.Linear(128, 10),
            neuron.LIFNode(surrogate_function=surrogate.Sigmoid()),
            
            
        )

    def forward(self, x):
        # SpikingJelly의 MultiStep 모드: (T, N, C, H, W) 형태의 입력을 기대함
        
        functional.reset_net(self.network)
        out = []
        for t in range(50):
            out.append(self.network(x[t]))
        return torch.stack(out)
        

# 시뮬레이션 설정
time_steps = 50
model = MySpikingModel()
print(f"SpikingJelly 모델 'MySpikingModel' 생성 완료")