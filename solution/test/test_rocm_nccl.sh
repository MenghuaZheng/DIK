#!/bin/bash
# run_rocm_test.sh

# 设置NCCL环境变量
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=^lo,docker0
export NCCL_IB_DISABLE=1
export HSA_FORCE_FINE_GRAIN_PCIE=1
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 设置PyTorch相关环境变量
export MASTER_ADDR=localhost
export MASTER_PORT=12355
export WORLD_SIZE=8

# 使用torchrun启动
torchrun \
    --nproc_per_node=8 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=12355 \
    test_rocm_nccl.py