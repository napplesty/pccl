#pragma once

#include "registered_memory.h"
#include "types.h"
#include "communicator.h"
#include <cstdint>
#include <map>
#include <memory>
#include <vector>

namespace pccl {

struct BasicOperation {
  BasicOperationType type;
  union {
    struct {
      ProxyId proxy_id;
      OperationId op_id;
    } notify_op;
    struct {
      BufferType src_buffer_type;
      BufferType dst_buffer_type;
      int src;
      int dst;
      uint64_t src_offset;
      uint64_t dst_offset;
      uint64_t size;
      TagId tag;
    } net_op;
    struct {
      ElemwiseComputeType op_type;
      BufferType src_buffer_type;
      BufferType dst_buffer_type;
      int src;
      int dst;
      uint64_t src_offset;
      uint64_t dst_offset;
      TagId tag;
      bool with_epilogue;
    } elemwise_compute_op;
    struct {
      BufferType src_buffer_type;
      BufferType dst_buffer_type;
      int src;
      int dst;
      uint64_t src_offset;
      uint64_t dst_offset;
      TagId tag;
      bool with_epilogue;
    } load_store_op;
    struct {
      int phase;
      bool commit;
    } sw_op;
    struct {
      uint64_t event_id;
    } profile_op;
    struct {
      uint64_t granularity;
      uint64_t offset;
      uintptr_t func_ptr;
      uint32_t write_level;
    } epilogue_op;
  };
};

struct Workspace {
  OperatorId operator_id;
  RegisteredMemory lib;
  RegisteredMemory buffer;
  RegisteredMemory user_input;
  RegisteredMemory user_output;
  std::map<ComponentTypeFlags, std::vector<std::vector<BasicOperation>>>
      operations;
};

class Operator {
public:
  Operator() = default;
  ~Operator() = default;

  static Operator load(std::string_view path);
  void execute(Communicator &communicator);

private:
  class Impl;
  std::shared_ptr<Impl> impl_;
};

} // namespace pccl