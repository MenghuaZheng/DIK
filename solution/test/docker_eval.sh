#!/bin/bash
# check_environment.sh

echo "=== 检查Docker环境 ==="
echo "1. 检查GPU可见性:"
rocminfo | grep -E "Device Type|Marketing Name" || echo "rocminfo不可用"

echo -e "\n2. 检查PyTorch和ROCm:"
python -c "
import torch
print(f'PyTorch版本: {torch.__version__}')
print(f'ROCm可用: {torch.cuda.is_available()}')
print(f'GPU数量: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
"

echo -e "\n3. 检查NCCL:"
python -c "
import torch
print(f'NCCL可用: {torch.distributed.is_nccl_available()}')
"

echo -e "\n4. 检查设备权限:"
ls -la /dev/dri/ /dev/kfd

echo -e "\n5. 检查用户组:"
groups