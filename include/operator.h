#pragma once

#include "types.h"
#include <cstdint>
#include <memory>

namespace pccl {

struct BasicOperation {
  BasicOperationType type;
  union {
    struct {
      BufferType src_buffer_type;
      BufferType dst_buffer_type;
      int src;
      int dst;
      uint64_t src_offset;
      uint64_t dst_offset;
      uint64_t size;
      uint64_t tag;
    } net_op;
    struct {
      ElemwiseComputeType op_type;
      BufferType src_buffer_type;
      BufferType dst_buffer_type;
      int src_src;
      int dst_src;
      uint64_t src_offset;
      uint64_t dst_offset;
      uint64_t tag;
      bool with_epilogue;
    } elemwise_compute_op;
    struct {
      BufferType src_buffer_type;
      BufferType dst_buffer_type;
      int src_src;
      int dst_src;
      uint64_t src_offset;
      uint64_t dst_offset;
      uint64_t tag;
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

class Operator {
public:
  Operator() = default;
  ~Operator() = default;

  static Operator create(std::string_view path);
private:
  class Impl;
  std::shared_ptr<Impl> impl_;
};

} // namespace pccl