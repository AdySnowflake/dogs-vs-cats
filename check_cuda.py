import torch

print("是否可用 CUDA:", torch.cuda.is_available())
print("当前 CUDA 设备数量:", torch.cuda.device_count())
print("当前使用的设备:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "无")
