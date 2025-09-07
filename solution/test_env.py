import os
import torch
import torch.distributed as dist
import time
import socket
import multiprocessing as mp

def find_free_port():
    """找到一个空闲端口"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

def test_rocm_nccl_simple(rank, world_size, master_port):
    """ROCm专用的简单NCCL测试 - 修复连接问题"""
    try:
        print(f"Rank {rank}: 启动测试，使用主端口 {master_port}")
        
        # 设置环境变量 - 所有进程必须使用相同的值
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = str(master_port)  # 所有进程使用相同端口！
        os.environ['RANK'] = str(rank)
        os.environ['WORLD_SIZE'] = str(world_size)
        
        print(f"Rank {rank}: 环境变量设置完成")
        
        # 设置设备
        if torch.cuda.is_available():
            torch.cuda.set_device(rank)
            print(f"Rank {rank}: 使用GPU {torch.cuda.current_device()}")
        
        # 初始化进程组
        print(f"Rank {rank}: 开始初始化NCCL...")
        
        # 使用更简单的初始化方式
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size
        )
        
        print(f"Rank {rank}: NCCL初始化成功")
        
        # 简单测试
        tensor = torch.ones(100, device=f'cuda:{rank}') * (rank + 1)
        print(f"Rank {rank}: 执行all_reduce...")
        dist.all_reduce(tensor)
        
        # 验证结果
        expected_sum = sum(range(1, world_size + 1))
        expected = torch.ones(100, device=f'cuda:{rank}') * expected_sum
        
        if torch.allclose(tensor, expected, atol=1e-6):
            print(f"✅ Rank {rank}: NCCL测试通过")
            success = True
        else:
            print(f"❌ Rank {rank}: NCCL测试失败")
            print(f"  期望: {expected_sum}, 实际: {tensor[0].item()}")
            success = False
            
        # 清理
        dist.destroy_process_group()
        return success
            
    except Exception as e:
        print(f"❌ Rank {rank}: NCCL测试异常 - {e}")
        import traceback
        traceback.print_exc()
        return False

def run_single_test(world_size, test_name):
    """运行单个测试 - 修复版本"""
    print(f"\n{test_name}:")
    
    # 首先找到一个空闲端口
    master_port = find_free_port()
    print(f"使用主端口: {master_port}")
    
    ctx = mp.get_context('spawn')
    processes = []
    results = []
    
    try:
        # 先启动所有进程，但延迟执行
        for rank in range(world_size):
            p = ctx.Process(
                target=test_rocm_nccl_simple,
                args=(rank, world_size, master_port)
            )
            processes.append(p)
        
        # 快速连续启动所有进程
        for p in processes:
            p.start()
            time.sleep(0.5)  # 很短延迟，避免竞争但确保快速启动
        
        # 等待结果
        for i, p in enumerate(processes):
            p.join(timeout=60)
            if p.exitcode == 0:
                print(f"✅ GPU {i} 测试成功")
                results.append(True)
            else:
                print(f"❌ GPU {i} 测试失败，退出码: {p.exitcode}")
                results.append(False)
        
        return all(results)
        
    except Exception as e:
        print(f"测试异常: {e}")
        return False
    finally:
        for p in processes:
            if p.is_alive():
                p.terminate()
                p.join()

def run_rocm_diagnostic():
    """运行ROCm专用的诊断"""
    print("=" * 60)
    print("AMD ROCm NCCL通信诊断 - 修复连接问题")
    print("=" * 60)
    
    # 检查环境
    print(f"PyTorch版本: {torch.__version__}")
    print(f"GPU数量: {torch.cuda.device_count()}")
    
    # 逐步测试
    test_cases = [
        (1, "单卡NCCL初始化测试"),
        (2, "2个GPU通信测试"),
    ]
    
    results = {}
    
    for world_size, test_name in test_cases:
        if world_size > torch.cuda.device_count():
            print(f"\n跳过 {test_name}: 可用GPU不足")
            continue
            
        success = run_single_test(world_size, test_name)
        results[test_name] = success
        
        if not success:
            print(f"❌ {test_name} 失败，停止后续测试")
            break
    
    # 输出结果
    print("\n" + "=" * 60)
    print("测试结果汇总:")
    print("=" * 60)
    
    for test_name, success in results.items():
        status = "✅ 通过" if success else "❌ 失败"
        print(f"{test_name}: {status}")
    
    return all(results.values())

if __name__ == "__main__":
    # 设置ROCm特定的环境变量
    os.environ['NCCL_DEBUG'] = 'INFO'
    os.environ['NCCL_DEBUG_SUBSYS'] = 'INIT,ENV,NET'
    os.environ['NCCL_CUMEM_ENABLE'] = '0'
    os.environ['HIP_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
    os.environ['NCCL_SOCKET_IFNAME'] = 'enp196s0f3'
    
    # ROCm优化设置
    os.environ['HSA_FORCE_FINE_GRAIN_PCI'] = '1'
    os.environ['NCCL_PROTO'] = 'simple'
    
    print(f"设置网络接口: {os.environ['NCCL_SOCKET_IFNAME']}")
    
    success = run_rocm_diagnostic()
    
    if success:
        print("\n🎉 所有ROCm NCCL测试通过！")
    else:
        print("\n❌ 测试失败，请尝试以下解决方案:")

# 额外的调试函数
def debug_connection_issue():
    """调试连接问题"""
    print("\n=== 连接问题调试 ===")
    
    # 测试端口是否可用
    test_port = find_free_port()
    print(f"测试端口 {test_port} 是否可用...")
    
    try:
        # 尝试绑定端口
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('127.0.0.1', test_port))
            print(f"✅ 端口 {test_port} 可用")
    except Exception as e:
        print(f"❌ 端口 {test_port} 不可用: {e}")
    
    # 检查防火墙
    print("\n检查防火墙状态...")
    try:
        result = os.system('which ufw > /dev/null 2>&1 && ufw status | grep -q "Status: active" && echo "防火墙启用" || echo "防火墙未启用"')
    except:
        print("无法检查防火墙状态")

# 运行连接调试
debug_connection_issue()