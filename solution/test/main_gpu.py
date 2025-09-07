import os
from typing import Callable

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def do_all_reduce(rank: int, size: int):
    # 设置当前GPU设备
    torch.cuda.set_device(rank)
    # 创建GPU张量
    device = torch.device(f"cuda:{rank}")
    tensor = torch.ones(1, device=device)
    
    # create a group with all processors
    group = dist.new_group(list(range(size)))
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
    # can be dist.ReduceOp.PRODUCT, dist.ReduceOp.MAX, dist.ReduceOp.MIN
    # will output 4 for all ranks
    print(f"[{rank}] data = {tensor[0]}")


def init_process(rank: int, size: int, fn: Callable[[int, int], None], backend="gloo"):
    """Initialize the distributed environment."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29501"

    # os.environ["NCCL_DEBUG"] = "INFO"  # 获取详细日志，调试完毕后可考虑设置为 WARN
    os.environ["NCCL_DEBUG"] = "WARN"  # 获取详细日志，调试完毕后可考虑设置为 WARN
    # os.environ["NCCL_SOCKET_IFNAME"] = "lo"  # 强制使用回环接口通信，对于单机多卡有时很有效
    # os.environ["NCCL_P2P_DISABLE"] = "1"     # 禁用 Peer-to-Peer 通信，解决一些 NVLink/PCIe 问题:cite[6]
    # os.environ["NCCL_IB_DISABLE"] = "1"      # 禁用 Infiniband，强制使用 Socket
    # 对于 ROCm，还可以尝试设置
    # os.environ["HSA_FORCE_FINE_GRAIN_PCIE"] = "1" # 可能改善 PCIe 通信

    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)
    # 添加清理操作
    dist.destroy_process_group()


if __name__ == "__main__":
    size = 4
    processes = []
    backend = "nccl"  # 现在可以使用nccl后端了
    print("If nccl is available: ", torch.distributed.is_nccl_available())
    mp.set_start_method("spawn")
    os.environ["HIP_VISIBLE_DEVICES"] = "0,1,2,3"
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, do_all_reduce, backend))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()