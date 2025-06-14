#pragma once

#include <cstdint>
#include <bitset>

namespace pccl {

enum class FunctionType : int {
  Context,
  Stream,
  Memory,
  MemReg,
  FunctionTypeEnd,
};

enum class BasicOperationType : int {
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

enum class ComponentType : int {
  Cpu,
  Cuda,
  Hip,
  Port,
  ComponentTypeEnd,
};

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

struct alignas(16) ProxyTrigger {
  ProxyId proxy_id;
  OperatorId operator_id;
  union {
    struct {
      bool has_value : 1;
      uint64_t operation_id : 63;
    };
    uint64_t value;
  };
};

} // namespace pccl