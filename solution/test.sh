# 设置必要的环境变量
export NCCL_DEBUG=INFO
export NCCL_CUMEM_ENABLE=0
export NCCL_SOCKET_IFNAME=enp196s0f3

# 运行诊断脚本
python test_env.py