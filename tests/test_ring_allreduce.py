import torch
import torch.distributed as dist
import os
import sys
import numpy as np

# 导入PCCL模块
sys.path.append('/root/pccl2/pccl')
import pccl

def setup_distributed():
    """设置分布式环境"""
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl')
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    if world_size != 8:
        if rank == 0:
            print(f"Error: This test requires 8 GPUs, but found {world_size}")
        return False, rank, world_size
    
    return True, rank, world_size

def create_runtime_configs(world_size):
    """创建运行时配置"""
    runtime_configs = []
    
    for i in range(world_size):
        config = pccl.RuntimeConfig()
        config.rank = i
        config.world_size = world_size
        
        # 缓冲区配置
        config.buffer_nums = {
            pccl.ExecutorType.CPU: 20,   # 信号缓冲区
            pccl.ExecutorType.CUDA: 10   # 数据缓冲区
        }
        
        config.buffer_sizes = {
            pccl.ExecutorType.CPU: 4 * 1024,        # 4KB信号缓冲区
            pccl.ExecutorType.CUDA: 128 * 1024 * 1024  # 128MB数据缓冲区
        }
        
        # 网络配置
        config.endpoint_configs = {
            "pccl.runtime.rank": str(i),
            "pccl.runtime.world_size": str(world_size),
            "pccl.runtime.use_tcp": "true",
            "pccl.runtime.host_sign": f"host_{i}",
            
            # OOB配置
            "pccl.oob.ip": "127.0.0.1",
            "pccl.oob.port": str(10000 + i),
            
            # TCP配置
            "pccl.tcp.local_ip": "127.0.0.1",
            "pccl.tcp.remote_ip": "127.0.0.1",
            "pccl.tcp.remote_port": str(10000 + ((i + 1) % world_size)),
        }
        
        runtime_configs.append(config)
    
    return runtime_configs

def build_ring_allreduce_graph(rank, world_size, data_size, dtype):
    """构建ring allreduce图"""
    graph = pccl.PrimitiveGraph(rank)
    
    # 计算chunk大小
    chunk_size = data_size // world_size
    if dtype == pccl.DataType.F32:
        type_size = 4
    elif dtype == pccl.DataType.F16:
        type_size = 2
    else:
        type_size = 2  # BF16
    
    chunk_bytes = chunk_size * type_size
    total_bytes = data_size * type_size
    
    # 添加缓冲区
    # 0: 输入缓冲区
    # 1: 输出缓冲区
    # 2: 发送缓冲区 (当前chunk)
    # 3: 接收缓冲区
    # 4: reduce临时缓冲区
    # 5-19: 信号缓冲区
    
    graph.addBuffer(0, dtype, total_bytes)    # 输入
    graph.addBuffer(1, dtype, total_bytes)    # 输出
    graph.addBuffer(2, dtype, chunk_bytes)    # 发送chunk
    graph.addBuffer(3, dtype, chunk_bytes)    # 接收chunk  
    graph.addBuffer(4, dtype, chunk_bytes)    # reduce缓冲区
    
    # 信号缓冲区
    for i in range(5, 20):
        graph.addBuffer(i, pccl.DataType.F32, 4)
    
    # 第一步: 将输入数据分块到发送缓冲区
    copy_config = pccl.PrimitiveConfig()
    copy_config.type = pccl.PrimitiveType.COPY
    copy_config.dtype = dtype
    copy_config.src_buffer_idx = 0
    copy_config.dst_buffer_idx = 2
    copy_config.data_size = chunk_bytes
    copy_config.executor_type = pccl.ExecutorType.CUDA
    copy_config.num_executors = 1
    
    copy_op = graph.addOperator(copy_config)
    last_op = copy_op
    
    # 初始化信号 - 通知下一个rank我们准备好了
    signal_config = pccl.PrimitiveConfig()
    signal_config.type = pccl.PrimitiveType.SIGNAL
    signal_config.signal_value = 1000 + rank
    signal_config.target_rank = (rank + 1) % world_size
    signal_config.dst_buffer_idx = 5
    signal_config.executor_type = pccl.ExecutorType.CPU
    signal_config.num_executors = 1
    
    signal_op = graph.addOperator(signal_config)
    graph.addDependency(last_op, signal_op)
    last_op = signal_op
    
    # Reduce-Scatter阶段 (7步循环)
    for step in range(world_size - 1):
        # 计算当前步骤的发送和接收rank
        recv_from = (rank - step - 1) % world_size
        send_to = (rank + 1) % world_size
        
        # 等待前一个rank的信号
        wait_config = pccl.PrimitiveConfig()
        wait_config.type = pccl.PrimitiveType.WAITSIGNAL
        wait_config.signal_value = 1000 + recv_from
        wait_config.src_buffer_idx = 6 + step  # 不同的信号缓冲区
        wait_config.executor_type = pccl.ExecutorType.CPU
        wait_config.num_executors = 1
        
        wait_op = graph.addOperator(wait_config)
        graph.addDependency(last_op, wait_op)
        last_op = wait_op
        
        # 接收数据
        recv_config = pccl.PrimitiveConfig()
        recv_config.type = pccl.PrimitiveType.WRITE
        recv_config.dtype = dtype
        recv_config.target_rank = rank
        recv_config.src_buffer_idx = 2  # 从发送者的发送缓冲区
        recv_config.dst_buffer_idx = 3  # 到本地的接收缓冲区
        recv_config.data_size = chunk_bytes
        recv_config.executor_type = pccl.ExecutorType.CUDA
        recv_config.num_executors = 1
        
        recv_op = graph.addOperator(recv_config)
        graph.addDependency(last_op, recv_op)
        last_op = recv_op
        
        # 本地reduce操作
        if step == 0:
            # 第一步: 将接收的数据拷贝到reduce缓冲区
            reduce_config = pccl.PrimitiveConfig()
            reduce_config.type = pccl.PrimitiveType.COPY
            reduce_config.dtype = dtype
            reduce_config.src_buffer_idx = 3
            reduce_config.dst_buffer_idx = 4
            reduce_config.data_size = chunk_bytes
            reduce_config.executor_type = pccl.ExecutorType.CUDA
            reduce_config.num_executors = 1
        else:
            # 后续步骤: 累加到reduce缓冲区
            reduce_config = pccl.PrimitiveConfig()
            reduce_config.type = pccl.PrimitiveType.COMPUTE
            reduce_config.dtype = dtype
            reduce_config.compute_op = pccl.ComputeType.SUM
            reduce_config.src_buffer_idx = 3  # 新接收的数据
            reduce_config.dst_buffer_idx = 4  # 累加到reduce缓冲区
            reduce_config.data_size = chunk_bytes
            reduce_config.executor_type = pccl.ExecutorType.CUDA
            reduce_config.num_executors = 1
        
        reduce_op = graph.addOperator(reduce_config)
        graph.addDependency(last_op, reduce_op)
        last_op = reduce_op
        
        # 发送信号给下一个rank
        next_signal_config = pccl.PrimitiveConfig()
        next_signal_config.type = pccl.PrimitiveType.SIGNAL
        next_signal_config.signal_value = 1000 + rank
        next_signal_config.target_rank = send_to
        next_signal_config.dst_buffer_idx = 7 + step
        next_signal_config.executor_type = pccl.ExecutorType.CPU
        next_signal_config.num_executors = 1
        
        signal_op = graph.addOperator(next_signal_config)
        graph.addDependency(last_op, signal_op)
        last_op = signal_op
    
    # Allgather阶段开始: 将reduce后的chunk放到输出缓冲区
    scatter_config = pccl.PrimitiveConfig()
    scatter_config.type = pccl.PrimitiveType.COPY
    scatter_config.dtype = dtype
    scatter_config.src_buffer_idx = 4  # reduce后的chunk
    scatter_config.dst_buffer_idx = 1  # 输出缓冲区
    scatter_config.data_size = chunk_bytes
    scatter_config.executor_type = pccl.ExecutorType.CUDA
    scatter_config.num_executors = 1
    
    scatter_op = graph.addOperator(scatter_config)
    graph.addDependency(last_op, scatter_op)
    last_op = scatter_op
    
    # Allgather阶段 (7步循环)
    for step in range(world_size - 1):
        # 计算当前步骤的发送和接收rank
        send_to = (rank + 1) % world_size
        recv_from = (rank - step - 1) % world_size
        
        # 等待信号
        wait_config = pccl.PrimitiveConfig()
        wait_config.type = pccl.PrimitiveType.WAITSIGNAL
        wait_config.signal_value = 2000 + recv_from
        wait_config.src_buffer_idx = 12 + step
        wait_config.executor_type = pccl.ExecutorType.CPU
        wait_config.num_executors = 1
        
        wait_op = graph.addOperator(wait_config)
        graph.addDependency(last_op, wait_op)
        last_op = wait_op
        
        # 发送reduce后的chunk给下一个rank
        send_config = pccl.PrimitiveConfig()
        send_config.type = pccl.PrimitiveType.WRITE
        send_config.dtype = dtype
        send_config.target_rank = send_to
        send_config.src_buffer_idx = 4  # reduce后的chunk
        send_config.dst_buffer_idx = 1  # 目标rank的输出缓冲区
        send_config.data_size = chunk_bytes
        send_config.executor_type = pccl.ExecutorType.CUDA
        send_config.num_executors = 1
        
        send_op = graph.addOperator(send_config)
        graph.addDependency(last_op, send_op)
        last_op = send_op
        
        # 发送完成信号
        signal_config = pccl.PrimitiveConfig()
        signal_config.type = pccl.PrimitiveType.SIGNAL
        signal_config.signal_value = 2000 + rank
        signal_config.target_rank = send_to
        signal_config.dst_buffer_idx = 13 + step
        signal_config.executor_type = pccl.ExecutorType.CPU
        signal_config.num_executors = 1
        
        signal_op = graph.addOperator(signal_config)
        graph.addDependency(last_op, signal_op)
        last_op = signal_op
    
    return graph

def main():
    """主函数：执行8卡ring allreduce测试"""
    
    # 设置分布式环境
    success, rank, world_size = setup_distributed()
    if not success:
        return
    
    torch.cuda.set_device(rank)
    
    # 测试参数
    data_size = 1024 * 1024  # 1M元素
    dtype = torch.float32
    pccl_dtype = pccl.DataType.F32
    
    # 创建测试数据
    torch.manual_seed(42 + rank)
    input_tensor = torch.randn(data_size, device=f'cuda:{rank}', dtype=dtype)
    expected_output = torch.zeros_like(input_tensor)
    
    # 使用PyTorch计算预期结果
    dist.all_reduce(input_tensor.clone(), op=dist.ReduceOp.SUM, out=expected_output)
    
    # 重置输入数据
    input_tensor = torch.randn(data_size, device=f'cuda:{rank}', dtype=dtype)
    output_tensor = torch.zeros_like(input_tensor)
    
    if rank == 0:
        print("Initializing PCCL runtime...")
    
    # 创建运行时配置
    runtime_configs = create_runtime_configs(world_size)
    
    # 初始化PCCL
    success = pccl.initializeRuntime(runtime_configs, rank, world_size)
    if not success:
        if rank == 0:
            print("Failed to initialize PCCL runtime")
        return
    
    if rank == 0:
        print("Building ring allreduce graph...")
    
    # 构建图
    graph = build_ring_allreduce_graph(rank, world_size, data_size, pccl_dtype)
    
    if rank == 0:
        print("Executing graph...")
    
    # 执行图
    participants = list(range(world_size))
    
    try:
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        success = pccl.executeGraph(graph, participants, input_tensor, output_tensor)
        end_time.record()
        
        torch.cuda.synchronize()
        execution_time = start_time.elapsed_time(end_time)
        
        if success:
            # 验证结果
            if torch.allclose(output_tensor, expected_output, rtol=1e-4, atol=1e-6):
                if rank == 0:
                    print(f"✓ Ring allreduce test PASSED")
                    print(f"Execution time: {execution_time:.2f} ms")
                    print(f"Data size: {data_size} elements ({data_size * 4 / 1024 / 1024:.2f} MB)")
            else:
                max_diff = (output_tensor - expected_output).abs().max().item()
                if rank == 0:
                    print(f"✗ Ring allreduce test FAILED")
                    print(f"Max difference: {max_diff}")
        else:
            if rank == 0:
                print("✗ Graph execution failed")
                
    except Exception as e:
        if rank == 0:
            print(f"✗ Exception during execution: {e}")
    
    # 清理
    pccl.shutdownRuntime()
    
    if rank == 0:
        print("Test completed")

if __name__ == "__main__":
    main()
