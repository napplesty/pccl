#pragma once

#include <cstdint>
#include <memory>
#include <map>
#include <runtime/api/repr.h>

namespace pccl::engine {

using runtime::PrimitiveType;
using runtime::DataType;
using runtime::ComputeType;
using runtime::ExecutorType;

struct PrimitiveConfig {
  PrimitiveType type;
  DataType dtype;
  int target_rank;
  void* src_buffer;
  void* dst_buffer; 
  unsigned long long data_size;
  ComputeType compute_op;
  int signal_value;
};

enum class ReadyQueueType {
  CPU_CUDA,
  CUDA_CPU,
  CPU_CPU,
  CUDA_CUDA,
  LAST
};

struct OperatorLayout {
  int uid;
  int required_executors;
  int remaining_executors;
  int dependency_count;
  OperatorLayout* next_operators[8];
  ExecutorType executor_type;
  int num_next;
  PrimitiveConfig primitive_config;
};

struct ReadyQueueLayout {
  void* buffer;
  unsigned long long capacity;
  unsigned long long head;
  unsigned long long tail;
  ExecutorType producer_type;
  ExecutorType consumer_type;
};

struct GraphBufferLayout {
  OperatorLayout* operators;
  uint64_t num_operators;
  ReadyQueueLayout* ready_queues;
  uint64_t num_queues;
};

class GraphExecutor {
public:
  GraphExecutor();
  ~GraphExecutor();
  bool initialize(GraphBufferLayout* graph_layout, const std::map<ExecutorType, int>& executor_config);
  void start();
  void stop();
  void wait();
  void initialize_ready_queues();
private:
  GraphBufferLayout* graph_layout_;
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}
