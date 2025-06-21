#pragma once

#include <bitset>
#include <cstdint>
#include <nlohmann/json.hpp>

namespace pccl {

enum class FunctionType : int {
  Context,
  Stream,
  Memory,
  MemReg,
  Connection,
  PhaseSwitch,
  FunctionTypeEnd,
};

enum class BasicOperationType : int {
  Notify,
  GetNotified,
  Put,
  Get,
  Flush,
  Sync,
  ElemwiseCompute,
  Load,
  Store,
  WriteAndCall,
  PhaseChange,
  Commit,
  Record,
  BasicOperationTypeEnd,
};

enum class BufferType : int {
  Control,
  Data,
  TemporalControl,
  TemporalData,
  BufferTypeEnd,
};

// 内存分配和计算
enum class ComponentType : int {
  Cpu,
  CpuIpc,
  Cuda,
  CudaIpc,
  Nvls,
  Hip,
  HipIpc,
  Port,
  ComponentTypeEnd,
};

// 外围设备插件
enum class PluginType : int {
  Ib,
  Tcp,
  Ovs,
  Cisco,
  PluginTypeEnd,
};

enum class ElemwiseComputeType : int {
  Add,
  Sub,
  Mul,
  Div,
  Max,
  Min,
  ElemwiseComputeTypeEnd,
};

enum class CacheLevel {
  Register,
  Shared,
  GlobalMemory,
  CacheLevelEnd,
};

using FunctionTypeFlags = std::bitset<(size_t)FunctionType::FunctionTypeEnd>;
using OperationTypeFlags = std::bitset<(size_t)BasicOperationType::BasicOperationTypeEnd>;
using ComponentTypeFlags = std::bitset<(size_t)ComponentType::ComponentTypeEnd>;
using PluginTypeFlags = std::bitset<(size_t)PluginType::PluginTypeEnd>;

using ProxyId = uint32_t;
using OperatorId = uint32_t;
using OperationId = uint32_t;
using TagId = int;
using HandleType = nlohmann::json;

struct alignas(16) ProxyTrigger {
  ProxyId proxy_id;
  OperatorId operator_id;
  bool has_value;
  OperationId operation_id;
};

} // namespace pccl