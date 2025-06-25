from thop import profile
import torch
from ConvMamba_arch import MConvMamba as net

# 确定是否可以使用 CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化模型
model = net()

# 将模型发送到设备（GPU或CPU）
model.to(device)

# 创建输入数据并发送到相同的设备
input = torch.randn(1, 3, 320, 180).to(device)

# 使用 thop 计算 FLOPs 和参数
macs, params = profile(model, inputs=(input, ))

# 输出结果
print("Multi-adds[G]:", macs / 1e9)
print("Parameters [K]:", params / 1e3)
