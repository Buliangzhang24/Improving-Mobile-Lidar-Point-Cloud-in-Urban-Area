import torch

# 检查是否有可用的 GPU
print(torch.cuda.is_available())

# 获取当前设备的 ID
print(torch.cuda.current_device())

# 获取当前 GPU 的名称
print(torch.cuda.get_device_name(torch.cuda.current_device()))
