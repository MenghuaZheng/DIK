import torch
import torch.distributed as dist
import os
import time

def setup():
    """初始化分布式环境"""
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    
    # 设置当前GPU - 使用更安全的方式
    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(device)
    
    # 初始化进程组
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    
    print(f"Rank {rank}/{world_size-1} on GPU {local_rank} initialized")
    return rank, local_rank, world_size, device

def test_basic_operations():
    """测试基本操作"""
    rank, local_rank, world_size, device = setup()
    
    try:
        # 测试1: 简单的张量创建和计算
        print(f"Rank {rank}: Testing basic tensor operations...")
        tensor = torch.ones(10, device=device) * (rank + 1)
        result = tensor.sum()
        print(f"Rank {rank}: Basic tensor operation result: {result.item()}")
        
        # 测试2: 简单的all_reduce（使用更小的张量）
        print(f"Rank {rank}: Testing all_reduce...")
        small_tensor = torch.tensor([1.0], device=device)
        dist.all_reduce(small_tensor, op=dist.ReduceOp.SUM)
        print(f"Rank {rank}: all_reduce result: {small_tensor.item()}")
        
        # 测试3: 验证结果
        expected = world_size
        if abs(small_tensor.item() - expected) < 1e-6:
            print(f"Rank {rank}: ✓ Test PASSED")
            return True
        else:
            print(f"Rank {rank}: ✗ Test FAILED - expected {expected}, got {small_tensor.item()}")
            return False
            
    except Exception as e:
        print(f"Rank {rank}: Error occurred: {str(e)}")
        return False
    finally:
        dist.destroy_process_group()

if __name__ == "__main__":
    success = test_basic_operations()
    exit(0 if success else 1)