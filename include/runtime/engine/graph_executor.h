#pragma once
#include <cstdint>
#include <memory>
#include <map>
#include <runtime/api/repr.h>
#include <runtime/engine/memory_manager.h>
#include <runtime/communicator/channel.h>

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
  uint32_t src_lkey;
  uint32_t src_rkey;
  uint32_t dst_lkey;
  uint32_t dst_rkey;
  unsigned long long data_size;
  ComputeType compute_op;
  int signal_value;
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
  uint64_t *completed_operator;
  uint64_t *total_operator;
  OperatorLayout* operators;
  uint64_t num_operators;
  ReadyQueueLayout* ready_queues;
  uint64_t num_queues;
};

class GraphExecutor {
public:
  GraphExecutor();
  ~GraphExecutor();
  bool initialize(
    runtime::PrimitiveGrpah& graph,
    const engine::WorkspaceHandle& workspace_handle,
    const std::map<runtime::ExecutorType, int>& executor_config,
    std::shared_ptr<communicator::ChannelManager> channel_manager,
    std::shared_ptr<MemoryManager> memory_manager
  );
  void issue();
  void wait();
  void initialize_ready_queues();
private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace pccl::engine
