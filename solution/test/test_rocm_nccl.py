# test_rocm_nccl.py
import torch
import torch.distributed as dist
import os
import time
import datetime

def setup():
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    
    # 设置当前GPU
    torch.cuda.set_device(local_rank)
    
    # 初始化进程组
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank,
        timeout=datetime.timedelta(seconds=60)  # 增加超时时间
    )
    
    print(f"Rank {rank}/{world_size-1} on GPU {local_rank} initialized")
    return rank, local_rank, world_size

def test_nccl_communication():
    """测试NCCL通信"""
    rank, local_rank, world_size = setup()
    
    try:
        # 测试1: 简单的点对点通信
        if rank == 0:
            send_tensor = torch.ones(100, device=f'cuda:{local_rank}')
            dist.send(send_tensor, dst=1)
            print(f"Rank {rank}: Sent tensor to rank 1")
        elif rank == 1:
            recv_tensor = torch.zeros(100, device=f'cuda:{local_rank}')
            dist.recv(recv_tensor, src=0)
            print(f"Rank {rank}: Received tensor from rank 0: {recv_tensor.sum().item()}")
        
        # 同步所有进程
        dist.barrier()
        
        # 测试2: 简单的all_reduce（使用非常小的张量）
        small_tensor = torch.tensor([1.0], device=f'cuda:{local_rank}')
        dist.all_reduce(small_tensor, op=dist.ReduceOp.SUM)
        
        print(f"Rank {rank}: all_reduce result: {small_tensor.item()}")
        
        # 验证结果
        expected = world_size
        if abs(small_tensor.item() - expected) < 1e-6:
            print(f"Rank {rank}: ✓ Test PASSED")
            return True
        else:
            print(f"Rank {rank}: ✗ Test FAILED")
            return False
            
    except Exception as e:
        print(f"Rank {rank}: Error occurred: {str(e)}")
        return False
    finally:
        dist.destroy_process_group()

if __name__ == "__main__":
    success = test_nccl_communication()
    exit(0 if success else 1)