#!/bin/bash
# run_fixed_test.sh

# 清除可能冲突的环境变量
unset MASTER_ADDR
unset MASTER_PORT
unset WORLD_SIZE
unset RANK
unset LOCAL_RANK

# 设置关键环境变量
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=^lo,docker0
export NCCL_IB_DISABLE=1
export HSA_FORCE_FINE_GRAIN_PCIE=1
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 禁用NUMA平衡（根据NCCL警告）
echo "Disabling NUMA balancing..."
sysctl kernel.numa_balancing=0 || echo "Warning: Could not disable NUMA balancing"

# 使用torchrun启动
echo "Starting distributed test..."
torchrun \
    --nproc_per_node=8 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=12355 \
    test_simple_communication.py