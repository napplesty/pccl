# PCCL Runtime 实现说明

PCCL是一个高性能的并行集合通信库，设计用于支持分布式深度学习和高性能计算应用。它提供了类似于NCCL的功能，但具有更灵活的架构和更多的传输协议支持。

## 主要特性

- **多种传输协议支持**
  - Host IPC：用于主机内存之间的高速通信
  - CUDA IPC：用于同节点GPU之间的直接内存访问
  - InfiniBand (IB)：用于跨节点的高速RDMA通信
  - Ethernet：用于标准以太网环境
  - NVLS (NVIDIA Link Switch)：用于最新GPU的高速互连

- **灵活的内存管理**
  - 预定义的内存缓冲区（LIB、HOST、DEVICE）
  - 动态工作空间分配
  - 支持主机和设备内存的统一管理

- **丰富的集合通信操作**
  - AllReduce：所有进程的归约操作
  - Broadcast：广播操作
  - AllGather：全收集操作
  - 可扩展的操作符框架

- **高级特性**
  - 网络拓扑感知的优化
  - 动态网络配置切换
  - 性能分析支持

## 构建和安装

### 依赖项

- CMake >= 3.18
- C++17兼容的编译器
- CUDA >= 11.0（可选，用于GPU支持）
- nlohmann/json
- pybind11（可选，用于Python绑定）

### 构建步骤

```bash
mkdir build && cd build
cmake .. -DUSE_CUDA=ON -DBUILD_EXAMPLES=ON
make -j$(nproc)
sudo make install
```

### 配置选项

- `USE_CUDA`：启用CUDA支持（默认：ON）
- `USE_HIP`：启用AMD HIP支持（默认：OFF）
- `BUILD_EXAMPLES`：构建示例程序（默认：ON）
- `BUILD_TESTS`：构建测试（默认：ON）

## 使用示例

### 基本的AllReduce操作

```cpp
#include <pccl/runtime.h>
#include <vector>

using namespace pccl;

int main() {
    // 创建通信器
    auto comm = std::make_shared<Communicator>();
    
    // 注册操作符
    auto allreduce = comm->registerOperator("allreduce_ring.op");
    
    // 准备数据
    std::vector<float> data(1024, 1.0f);
    
    // 执行AllReduce
    Event event;
    auto result = allreduce->execute(
        rank, data.data(), data.data(),
        DataType::FP32, data.size() * sizeof(float),
        data.size() * sizeof(float), event, true, 0
    );
    
    // 等待完成
    result.wait();
    
    return 0;
}
```

### 环境变量配置

PCCL使用环境变量进行配置：

- `PCCL_RANK`：当前进程的rank
- `PCCL_WORLD_SIZE`：总进程数
- `PCCL_LOCAL_RANK`：节点内的本地rank
- `PCCL_SOCKET_ADDR`：通信地址
- `PCCL_SOCKET_PORT`：通信端口
- `PCCL_IB_DEVICE0`：InfiniBand设备0
- `PCCL_IB_DEVICE1`：InfiniBand设备1
- `PCCL_ENABLE_TRANSPORT_LIST`：启用的传输协议列表

### 运行多进程示例

```bash
# 设置环境变量
export PCCL_WORLD_SIZE=4

# 在不同终端运行
PCCL_RANK=0 ./build/examples/simple_allreduce
PCCL_RANK=1 ./build/examples/simple_allreduce
PCCL_RANK=2 ./build/examples/simple_allreduce
PCCL_RANK=3 ./build/examples/simple_allreduce
```

## 架构概述

PCCL采用模块化设计，主要组件包括：

1. **Communicator**：核心通信管理器
2. **MemoryContext**：内存管理上下文
3. **ConnectionContext**：连接管理上下文
4. **ClusterContext**：集群拓扑管理
5. **Operator**：集合通信操作执行器
6. **Connection**：底层传输连接抽象

## 性能优化

- 使用Ring算法优化AllReduce操作
- 支持NVLS多播加速
- 自动选择最优传输协议
- 内存池管理减少分配开销

## 贡献指南

欢迎贡献代码！请遵循以下步骤：

1. Fork项目
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 创建Pull Request

## 许可证

### 1. 可编程网络拓扑
- 支持通过NetConf协议动态配置网络设备
- 支持通信阶段的切换和网络重配置
- 集成光交换机支持高性能光网络

### 2. 多传输支持
- 统一抽象不同传输类型（IB、Ethernet、CUDA_IPC、NVLS）
- 支持传输类型的组合和自动选择
- 透明的跨传输类型通信

### 3. 内存管理
- 统一管理主机和设备内存
- 支持跨进程的内存注册和共享
- 协调的工作空间分配和同步

### 4. 异步执行
- Event机制支持异步操作
- 分离数据传输和同步控制
- 支持性能分析和调试

## 实现状态

当前所有方法都提供了占位实现，包含详细的功能说明和TODO注释，为后续的具体实现提供了清晰的指导。每个方法都基于函数命名和上下文分析推测了其预期功能。 