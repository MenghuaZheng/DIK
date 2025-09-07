#!/bin/bash

# 设置环境变量
export MASTER_ADDR=localhost
export MASTER_PORT=12355
export WORLD_SIZE=8

# 使用torchrun启动分布式训练
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port=12355 test_rocm_distributed.py