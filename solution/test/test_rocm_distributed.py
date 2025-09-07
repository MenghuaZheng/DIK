import torch
import torch.distributed as dist
import os
import time
import argparse

def setup(rank, world_size):
    """初始化分布式环境"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # 初始化进程组
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    print(f"Rank {rank}/{world_size-1} initialized successfully on GPU {torch.cuda.current_device()}")

def cleanup():
    """清理分布式环境"""
    dist.destroy_process_group()

def get_global_rank():
    """获取全局rank"""
    return int(os.environ['RANK'])

def get_local_rank():
    """获取本地rank"""
    return int(os.environ['LOCAL_RANK'])

def get_world_size():
    """获取world size"""
    return int(os.environ['WORLD_SIZE'])

def test_all_to_all(rank, world_size, tensor_size=1024):
    """测试All-to-All通信"""
    print("Start to Test all2all!")

    # 创建输入张量
    input_tensor = torch.ones(tensor_size, dtype=torch.float32, device=f'cuda:{rank}') * (rank + 1)
    
    # 创建输出张量列表
    output_tensors = [torch.zeros(tensor_size, dtype=torch.float32, device=f'cuda:{rank}') 
                     for _ in range(world_size)]
    
    # 同步所有进程
    dist.barrier()
    
    if rank == 0:
        print(f"\nTesting All-to-All communication with tensor size {tensor_size}")
        print("Input tensors:")
        for i in range(world_size):
            print(f"  Rank {i}: tensor filled with value {i+1}")
    
    # 执行All-to-All操作
    start_time = time.time()
    dist.all_to_all(output_tensors, input_tensor)
    end_time = time.time()
    
    # 同步并验证结果
    dist.barrier()
    
    # 验证结果
    success = True
    for i, output_tensor in enumerate(output_tensors):
        expected_value = i + 1  # 应该收到来自rank i的值
        if not torch.allclose(output_tensor, torch.tensor(expected_value, dtype=torch.float32, device=f'cuda:{rank}')):
            print(f"Rank {rank} ERROR: Received {output_tensor[0].item()} from rank {i}, expected {expected_value}")
            success = False
    
    if success and rank == 0:
        print("All-to-All test PASSED!")
        print(f"Communication time: {(end_time - start_time) * 1000:.2f} ms")
    
    return success

def test_broadcast(rank, world_size, tensor_size=1024):
    """测试广播通信"""
    print("Start to Test broadcast!")
    if rank == 0:
        data = torch.ones(tensor_size, dtype=torch.float32, device=f'cuda:{rank}') * 42.0
    else:
        data = torch.zeros(tensor_size, dtype=torch.float32, device=f'cuda:{rank}')
    
    dist.broadcast(data, src=0)
    
    # 验证广播结果
    if not torch.allclose(data, torch.tensor(42.0, dtype=torch.float32, device=f'cuda:{rank}')):
        print(f"Rank {rank} Broadcast test FAILED")
        return False
    
    if rank == 0:
        print("Broadcast test PASSED!")
    return True

def test_all_reduce(rank, world_size, tensor_size=1024):
    """测试All-Reduce通信"""
    print("Start to Test allreduce!")

    data = torch.ones(tensor_size, dtype=torch.float32, device=f'cuda:{rank}') * (rank + 1)
    
    dist.all_reduce(data, op=dist.ReduceOp.SUM)
    
    # 验证结果
    expected_sum = sum(range(1, world_size + 1))
    if not torch.allclose(data, torch.tensor(expected_sum, dtype=torch.float32, device=f'cuda:{rank}')):
        print(f"Rank {rank} All-Reduce test FAILED: got {data[0].item()}, expected {expected_sum}")
        return False
    
    if rank == 0:
        print(f"All-Reduce test PASSED! Sum: {data[0].item()}")
    return True

def main():
    # 从环境变量获取rank信息
    global_rank = get_global_rank()
    local_rank = get_local_rank()
    world_size = get_world_size()
    
    # 设置当前GPU
    torch.cuda.set_device(local_rank)
    
    print(f"Global Rank {global_rank}/{world_size-1}, Local Rank {local_rank}")
    print(f"GPU Name: {torch.cuda.get_device_name(local_rank)}")
    print(f"ROCm available: {torch.cuda.is_available()}")
    print(f"NCCL available: {dist.is_nccl_available()}")
    
    # 初始化分布式环境
    setup(global_rank, world_size)
    
    try:
        # 运行各种通信测试
        broadcast_success = test_broadcast(global_rank, world_size)
        all_reduce_success = test_all_reduce(global_rank, world_size)
        all_to_all_success = test_all_to_all(global_rank, world_size)
        
        # 汇总结果
        dist.barrier()
        if global_rank == 0:
            print("\n" + "="*50)
            print("DISTRIBUTED TRAINING TEST SUMMARY")
            print("="*50)
            print(f"Broadcast test: {'PASSED' if broadcast_success else 'FAILED'}")
            print(f"All-Reduce test: {'PASSED' if all_reduce_success else 'FAILED'}")
            print(f"All-to-All test: {'PASSED' if all_to_all_success else 'FAILED'}")
            print(f"Total GPUs tested: {world_size}")
            
            if broadcast_success and all_reduce_success and all_to_all_success:
                print("\n🎉 ALL TESTS PASSED! ROCm multi-GPU communication is working correctly!")
            else:
                print("\n❌ SOME TESTS FAILED! Please check your ROCm installation.")
                
    finally:
        cleanup()

if __name__ == "__main__":
    main()