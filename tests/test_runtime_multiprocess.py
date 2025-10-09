#!/usr/bin/env python3
"""
多进程测试脚本 - 测试PCCL runtime.cc功能
使用torchrun启动8个进程，每个进程设置runtime config并执行
"""

import os
import sys
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from typing import List, Dict, Any
import json

# 尝试导入PCCL模块
try:
    import pccl.cccl as cccl
    PCCL_AVAILABLE = True
except ImportError:
    print("警告: pccl.cccl模块不可用，将使用模拟模式")
    PCCL_AVAILABLE = False

class RuntimeTester:
    """Runtime测试器"""
    
    def __init__(self, rank: int, world_size: int):
        self.rank = rank
        self.world_size = world_size
        self.runtime_configs = []
        self.local_config = None
        
    def create_runtime_configs(self) -> List[Dict[str, Any]]:
        """创建所有进程的runtime配置"""
        configs = []
        
        for i in range(self.world_size):
            config = {
                "rank": i,
                "world_size": self.world_size,
                "buffer_nums": {
                    cccl.ExecutorType.CPU: 2,   # CPU buffers
                    cccl.ExecutorType.CUDA: 2   # CUDA buffers
                },
                "buffer_sizes": {
                    cccl.ExecutorType.CPU: 1024 * 1024,   # 1MB for CPU
                    cccl.ExecutorType.CUDA: 1024 * 1024   # 1MB for CUDA
                },
                "endpoint_configs": {
                    "pccl.runtime.rank": str(i),
                    "pccl.runtime.world_size": str(self.world_size),
                    "pccl.oob.ip": "29.119.98.121",
                    "pccl.oob.port": str(31250 + i),
                    "pccl.runtime.host_sign": str("oxksk"),
                    "pccl.runtime.use_roce": "true",
                    "pccl.roce.port_num": "1",
                    "pccl.roce.gid_index": "3",
                    "pccl.roce.lid": "0",
                    "pccl.roce.device_name": f"mlx5_bond_{i+1}",
                    "pccl.runtime.use_tcp": "false",
                    "pccl.tcp.local_ip": "127.0.0.1",
                    "pccl.tcp.local_port": str(31280 + i),
                }
            }
            configs.append(config)
            
        return configs
    
    def initialize_runtime(self):
        """初始化runtime"""
        if not PCCL_AVAILABLE:
            print(f"Rank {self.rank}: 模拟初始化runtime")
            return True
            
        try:
            # 创建runtime配置
            self.runtime_configs = self.create_runtime_configs()
            
            # 转换为RuntimeConfig对象
            runtime_config_objs = []
            for config in self.runtime_configs:
                runtime_config = cccl.RuntimeConfig()
                runtime_config.rank = config["rank"]
                runtime_config.world_size = config["world_size"]
                runtime_config.buffer_nums = config["buffer_nums"]
                runtime_config.buffer_sizes = config["buffer_sizes"]
                runtime_config.endpoint_configs = config["endpoint_configs"]
                runtime_config_objs.append(runtime_config)
            
            # 初始化runtime
            success = cccl.initializeRuntime(
                runtime_config_objs, 
                self.rank, 
                self.world_size
            )
            
            if success:
                print(f"Rank {self.rank}: Runtime初始化成功")
            else:
                print(f"Rank {self.rank}: Runtime初始化失败")
                
            return success
            
        except Exception as e:
            print(f"Rank {self.rank}: Runtime初始化异常: {e}")
            return False
    
    def test_basic_operations(self):
        """测试基本操作"""
        if not PCCL_AVAILABLE:
            print(f"Rank {self.rank}: 模拟测试基本操作")
            return True
            
        try:
            # 测试生成操作符ID
            op_id = cccl.generateOperatorId()
            print(f"Rank {self.rank}: 生成的operator ID: {op_id}")
            
            # 测试创建PrimitiveGraph
            graph = cccl.PrimitiveGraph(self.rank)
            
            # 添加buffer
            graph.addBuffer(0, cccl.DataType.F32, 1024)
            graph.addBuffer(1, cccl.DataType.F32, 1024)
            
            print(f"Rank {self.rank}: 基本操作测试完成")
            return True
            
        except Exception as e:
            print(f"Rank {self.rank}: 基本操作测试异常: {e}")
            return False
    
    def test_communication(self):
        """测试通信功能"""
        if not PCCL_AVAILABLE:
            print(f"Rank {self.rank}: 模拟测试通信功能")
            return True
            
        try:
            # 创建简单的计算图进行通信测试
            graph = cccl.PrimitiveGraph(self.rank)
            
            # 添加buffer
            graph.addBuffer(0, cccl.DataType.F32, 1024)  # 输入buffer
            graph.addBuffer(1, cccl.DataType.F32, 1024)  # 输出buffer
            
            # 创建操作符配置
            op_config = cccl.PrimitiveConfig()
            op_config.type = cccl.PrimitiveType.COPY
            op_config.dtype = cccl.DataType.F32
            op_config.src_buffer_idx = 0
            op_config.dst_buffer_idx = 1
            op_config.data_size = 1024
            op_config.executor_type = cccl.ExecutorType.CPU
            
            # 添加操作符
            graph.addOperator(op_config)
            
            print(f"Rank {self.rank}: 通信测试完成")
            return True
            
        except Exception as e:
            print(f"Rank {self.rank}: 通信测试异常: {e}")
            return False
    
    def shutdown_runtime(self):
        """关闭runtime"""
        if not PCCL_AVAILABLE:
            print(f"Rank {self.rank}: 模拟关闭runtime")
            return
            
        try:
            cccl.shutdownRuntime()
            print(f"Rank {self.rank}: Runtime关闭成功")
        except Exception as e:
            print(f"Rank {self.rank}: Runtime关闭异常: {e}")
    
    def run_all_tests(self):
        """运行所有测试"""
        print(f"Rank {self.rank}: 开始测试")
        
        # 测试1: 初始化runtime
        if not self.initialize_runtime():
            return False
        
        # 等待所有进程完成初始化
        if dist.is_initialized():
            dist.barrier()
        
        # 测试2: 基本操作
        if not self.test_basic_operations():
            return False
            
        # 测试3: 通信功能
        if not self.test_communication():
            return False
            
        # 等待所有进程完成测试
        if dist.is_initialized():
            dist.barrier()
        
        # 关闭runtime
        self.shutdown_runtime()
        
        print(f"Rank {self.rank}: 所有测试完成")
        return True


def setup_distributed(rank: int, world_size: int):
    
    # 初始化进程组
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    print(f"Rank {rank}: 分布式环境初始化完成")


def cleanup_distributed():
    """清理分布式环境"""
    dist.destroy_process_group()


def run_test(rank: int, world_size: int):
    """运行测试的主函数"""
    try:
        # 设置分布式环境
        setup_distributed(rank, world_size)
        print("setuped")
        # 创建测试器并运行测试
        tester = RuntimeTester(rank, world_size)
        success = tester.run_all_tests()
        print("tested ok")
        # 清理分布式环境
        cleanup_distributed()
        
        return success
        
    except Exception as e:
        print(f"Rank {rank}: 测试过程中发生异常: {e}")
        cleanup_distributed()
        return False


def main():
    """主函数"""
    # 检查是否在torchrun环境中
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        # torchrun模式 - 直接运行测试
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        print(f"检测到torchrun环境: rank={rank}, world_size={world_size}")
        return run_test(rank, world_size)
    
    # 非torchrun模式
    world_size = 8
    
    print("=" * 50)
    print("PCCL Runtime 多进程测试")
    print(f"使用 torchrun 启动 {world_size} 个进程")
    print("=" * 50)
    
    if PCCL_AVAILABLE:
        print("PCCL模块可用，将进行真实测试")
        print(f"PCCL版本: {cccl.get_version()}")
    else:
        print("PCCL模块不可用，将进行模拟测试")
    
    # 使用torchrun启动多进程测试
    print("\n启动命令:")
    print(f"torchrun --nproc_per_node={world_size} {__file__}")
    
    # 如果直接运行此脚本，则启动多进程
    if __name__ == "__main__":
        if len(sys.argv) > 1 and sys.argv[1] == "--standalone":
            # 独立模式 - 使用multiprocessing启动
            print("\n使用multiprocessing启动多进程测试...")
            processes = []
            
            for rank in range(world_size):
                p = mp.Process(target=run_test, args=(rank, world_size))
                p.start()
                processes.append(p)
            
            # 等待所有进程完成
            for p in processes:
                p.join()
                
            print("\n多进程测试完成")
        else:
            print("\n请使用以下命令运行测试:")
            print(f"torchrun --nproc_per_node={world_size} {__file__}")
            print("或使用独立模式:")
            print(f"python {__file__} --standalone")


if __name__ == "__main__":
    main()
