import torch

# 查询每个 GPU 的详细信息
for i in range(torch.cuda.device_count()):
    device = torch.device("cuda:{}".format(i))
    print("Device {}: {}".format(i, torch.cuda.get_device_name(device)))
    print("Device {}: Compute capability: {}".format(i, torch.cuda.get_device_capability(device)))
    print("Device {}: Total memory: {:.2f} GB".format(i, torch.cuda.get_device_properties(device).total_memory / 1e9))
    print()