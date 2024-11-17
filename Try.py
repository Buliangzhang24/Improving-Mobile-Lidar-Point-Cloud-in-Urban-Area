
import torch
print(torch.cuda.is_available())  # 检查是否有可用的 CUDA 设备
print(torch.cuda.current_device())  # 获取当前使用的设备索引
print(torch.cuda.get_device_name(0))  # 获取 GPU 设备名称
