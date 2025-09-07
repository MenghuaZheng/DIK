export POPCORN_FD=1
export POPCORN_SEED=42
export POPCORN_GPUS=8
export NCCL_CUMEM_ENABLE=0
export NCCL_DEBUG=INFO
export NCCL_SHM_DISABLE=0
export NCCL_P2P_DISABLE=1
python amd_distributed/eval.py leaderboard test_cases.txt
# torchrun --nproc_per_node=2 amd_distributed/eval.py leaderboard test_cases.txt
# torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port=12356 amd_distributed/eval.py leaderboard test_cases.txt
# rocm-smi --showtopo