import os, multiprocessing as mp

def run(rank, world_size):
    # 1. 子进程里先写死可见设备，保证只能看到一张卡
    os.environ["HIP_VISIBLE_DEVICES"] = str(rank)
    # 2. 再 import torch，此时 PyTorch 只会把这张卡映射成 cuda:0
    import torch, torch.distributed as dist

    torch.cuda.set_device(0)          # 一定是 0
    dist.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:29500",
        rank=rank,
        world_size=world_size)

    x = torch.ones(1, device="cuda:0")
    dist.all_reduce(x)
    print(f"[rank {rank}] result = {x.item()}")
    dist.destroy_process_group()

def launch():
    world_size = 4
    mp.set_start_method("spawn", force=True)
    procs = []
    for r in range(world_size):
        p = mp.Process(target=run, args=(r, world_size))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()

if __name__ == "__main__":
    import torch
    # ROCm 版本
    print("ROCm version :", torch.version.hip)

    # RCCL 版本（NCCL 接口）
    print("RCCL version :", torch.cuda.nccl.version() if torch.cuda.is_available() else "CUDA/RCCL not available")
    launch()