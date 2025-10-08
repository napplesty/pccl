#include "runtime/engine/graph_executor.h"
#include "plugins/ahost/executor.h"
#include "plugins/acu/executor.h"
#include "utils/allocator.h"
#include "utils/logging.h"
#include <cuda_runtime.h>
#include <cstring>

namespace pccl::engine {

class GraphExecutor::Impl {
public:
  bool initialize(
    runtime::PrimitiveGrpah& graph,
    const engine::WorkspaceHandle& workspace_handle,
    const std::map<runtime::ExecutorType, int>& executor_config,
    std::shared_ptr<communicator::ChannelManager> channel_manager,
    std::shared_ptr<MemoryManager> memory_manager
  ) {
    channel_manager_ = channel_manager;
    memory_manager_ = memory_manager;
    workspace_handle_ = workspace_handle;

    if (!convert_graph_to_layout(graph, workspace_handle)) {
      PCCL_LOG_ERROR("Failed to convert graph to layout");
      return false;
    }

    if (!allocate_unified_memory()) {
      PCCL_LOG_ERROR("Failed to allocate unified memory");
      return false;
    }

    if (!initialize_executors(executor_config)) {
      PCCL_LOG_ERROR("Failed to initialize executors");
      return false;
    }

    initialize_ready_queues();
    PCCL_LOG_INFO("GraphExecutor initialized successfully");
    return true;
  }

  void issue() {
    if (host_executor_) {
      host_executor_->launch();
    }
    if (cuda_executor_) {
      cuda_executor_->launch();
    }
    issued_ = true;
    PCCL_LOG_INFO("Graph execution started");
  }

  void wait() {
    if (!issued_) return;

    PCCL_LOG_INFO("Waiting for graph execution to complete");
    if (host_executor_) {
      host_executor_->wait();
    }
    if (cuda_executor_) {
      cudaStreamSynchronize(cuda_executor_->current_stream_);
    }

    cleanup();
    issued_ = false;
    PCCL_LOG_INFO("Graph execution completed");
  }

  void initialize_ready_queues() {
    if (!graph_layout_) return;

    for (uint64_t i = 0; i < graph_layout_->num_queues; ++i) {
      auto& queue = graph_layout_->ready_queues[i];
      queue.head = 0;
      queue.tail = 0;
    }

    int initial_count = 0;
    for (uint64_t i = 0; i < graph_layout_->num_operators; ++i) {
      auto& op = graph_layout_->operators[i];
      if (op.dependency_count == 0) {
        enqueue_operator(&op);
        initial_count++;
      }
    }
    PCCL_LOG_DEBUG("Enqueued {} initial operators", initial_count);
  }

private:
  bool convert_graph_to_layout(runtime::PrimitiveGrpah& graph, const engine::WorkspaceHandle& workspace_handle) {
    auto operators = graph.getOperators();
    
    graph_layout_ = new GraphBufferLayout();
    graph_layout_->num_operators = operators.size();
    graph_layout_->num_queues = 4;

    size_t op_size = graph_layout_->num_operators * sizeof(OperatorLayout);
    size_t queue_size = graph_layout_->num_queues * sizeof(ReadyQueueLayout);

    graph_layout_->operators = static_cast<OperatorLayout*>(utils::allocate(runtime::ExecutorType::CPU, op_size));
    graph_layout_->ready_queues = static_cast<ReadyQueueLayout*>(utils::allocate(runtime::ExecutorType::CPU, queue_size));

    if (!graph_layout_->operators || !graph_layout_->ready_queues) {
      PCCL_LOG_ERROR("Failed to allocate graph layout memory");
      return false;
    }

    memset(graph_layout_->operators, 0, op_size);
    memset(graph_layout_->ready_queues, 0, queue_size);

    int self_rank = workspace_handle.participant_ranks.empty() ? 0 : workspace_handle.participant_ranks[0];

    for (size_t i = 0; i < operators.size(); ++i) {
      auto& src_op = operators[i];
      auto& dst_op = graph_layout_->operators[i];

      dst_op.uid = i;
      dst_op.required_executors = src_op.num_executors;
      dst_op.remaining_executors = src_op.num_executors;
      dst_op.dependency_count = src_op.num_dependencies;
      dst_op.executor_type = src_op.executor_type;
      dst_op.num_next = src_op.num_followers;

      PrimitiveConfig engine_config;
      engine_config.type = src_op.type;
      engine_config.dtype = src_op.dtype;
      engine_config.target_rank = src_op.target_rank;
      engine_config.data_size = src_op.data_size;
      engine_config.compute_op = src_op.compute_op;
      engine_config.signal_value = src_op.signal_value;

      engine_config.src_buffer = resolve_buffer_address(workspace_handle, src_op.src_buffer_idx, self_rank, src_op.executor_type);
      engine_config.dst_buffer = resolve_buffer_address(workspace_handle, src_op.dst_buffer_idx, self_rank, src_op.executor_type);

      memcpy(&dst_op.primitive_config, &engine_config, sizeof(PrimitiveConfig));

      for (int j = 0; j < dst_op.num_next && j < 8; ++j) {
        int follower_id = src_op.followers[j];
        if (follower_id >= 0 && follower_id < static_cast<int>(operators.size())) {
          dst_op.next_operators[j] = &graph_layout_->operators[follower_id];
        }
      }
    }

    PCCL_LOG_DEBUG("Converted graph with {} operators and {} queues", 
                  graph_layout_->num_operators, graph_layout_->num_queues);
    return true;
  }

  void* resolve_buffer_address(const engine::WorkspaceHandle& workspace_handle, int buffer_idx, int rank, runtime::ExecutorType executor_type) {
    if (buffer_idx < 0) return nullptr;

    auto rank_buffers_it = workspace_handle.buffers.find(rank);
    if (rank_buffers_it == workspace_handle.buffers.end()) {
      PCCL_LOG_ERROR("No buffers found for rank {}", rank);
      return nullptr;
    }

    const auto& buffers = rank_buffers_it->second;
    if (buffer_idx >= static_cast<int>(buffers.size())) {
      PCCL_LOG_ERROR("Buffer index {} out of range for rank {}", buffer_idx, rank);
      return nullptr;
    }

    const auto& buffer_id = buffers[buffer_idx];
    
    if (executor_type == runtime::ExecutorType::CUDA && buffer_id.ipc_addr) {
      return buffer_id.ipc_addr;
    }
    
    return buffer_id.addr;
  }

  bool allocate_unified_memory() {
    if (!graph_layout_) return false;

    for (uint64_t i = 0; i < graph_layout_->num_queues; ++i) {
      auto& queue = graph_layout_->ready_queues[i];
      
      size_t buffer_size = 1024 * sizeof(OperatorLayout*);
      queue.buffer = utils::allocate(runtime::ExecutorType::CPU, buffer_size);
      if (!queue.buffer) {
        PCCL_LOG_ERROR("Failed to allocate unified memory for queue");
        return false;
      }

      queue.producer_type = static_cast<runtime::ExecutorType>(i / 2);
      queue.consumer_type = static_cast<runtime::ExecutorType>(i % 2);
      queue.head = 0;
      queue.tail = 0;
      queue.capacity = 1024;
    }

    PCCL_LOG_DEBUG("Allocated unified memory for {} queues", graph_layout_->num_queues);
    return true;
  }

  bool initialize_executors(const std::map<runtime::ExecutorType, int>& config) {
    if (!graph_layout_) return false;

    auto cpu_count = config.find(runtime::ExecutorType::CPU);
    auto cuda_count = config.find(runtime::ExecutorType::CUDA);

    if (cpu_count != config.end() && cpu_count->second > 0) {
      host_executor_ = std::make_unique<host::HostExecutorManager>(
        graph_layout_, cpu_count->second, graph_layout_->ready_queues, graph_layout_->num_queues,
        channel_manager_, memory_manager_
      );
      if (!host_executor_->initialize()) {
        PCCL_LOG_ERROR("Failed to initialize host executor");
        return false;
      }
      PCCL_LOG_DEBUG("Initialized CPU executor with {} threads", cpu_count->second);
    }

    if (cuda_count != config.end() && cuda_count->second > 0) {
      cuda_executor_ = std::make_unique<cuda::CUDAExecutorManager>(
        graph_layout_, cuda_count->second, graph_layout_->ready_queues, graph_layout_->num_queues
      );
      if (!cuda_executor_->initialize()) {
        PCCL_LOG_ERROR("Failed to initialize CUDA executor");
        return false;
      }
      PCCL_LOG_DEBUG("Initialized CUDA executor with {} SMs", cuda_count->second);
    }

    return true;
  }

  void enqueue_operator(OperatorLayout* op) {
    if (!op || !graph_layout_) return;

    int queue_index = static_cast<int>(runtime::ExecutorType::CPU) * 2 + static_cast<int>(op->executor_type);
    if (queue_index >= static_cast<int>(graph_layout_->num_queues)) {
      queue_index = 0;
    }

    auto& queue = graph_layout_->ready_queues[queue_index];
    uint64_t current_tail = queue.tail;
    uint64_t next_tail = (current_tail + 1) % queue.capacity;

    if (next_tail == queue.head) {
      PCCL_LOG_ERROR("Queue overflow for operator {}", op->uid);
      return;
    }

    OperatorLayout** buffer = static_cast<OperatorLayout**>(queue.buffer);
    buffer[current_tail] = op;
    queue.tail = next_tail;
  }

  void cleanup() {
    if (graph_layout_) {
      for (uint64_t i = 0; i < graph_layout_->num_queues; ++i) {
        auto& queue = graph_layout_->ready_queues[i];
        if (queue.buffer) {
          utils::close_shareable_handle(runtime::ExecutorType::CPU, queue.buffer);
          queue.buffer = nullptr;
        }
      }
      
      if (graph_layout_->operators) {
        utils::close_shareable_handle(runtime::ExecutorType::CPU, graph_layout_->operators);
        graph_layout_->operators = nullptr;
      }
      
      if (graph_layout_->ready_queues) {
        utils::close_shareable_handle(runtime::ExecutorType::CPU, graph_layout_->ready_queues);
        graph_layout_->ready_queues = nullptr;
      }
      
      delete graph_layout_;
      graph_layout_ = nullptr;
    }
    
    host_executor_.reset();
    cuda_executor_.reset();
  }

  std::shared_ptr<communicator::ChannelManager> channel_manager_;
  std::shared_ptr<MemoryManager> memory_manager_;
  engine::WorkspaceHandle workspace_handle_;
  std::unique_ptr<host::HostExecutorManager> host_executor_;
  std::unique_ptr<cuda::CUDAExecutorManager> cuda_executor_;
  GraphBufferLayout* graph_layout_ = nullptr;
  bool issued_ = false;
};

GraphExecutor::GraphExecutor() : impl_(new Impl()) {}
GraphExecutor::~GraphExecutor() = default;

bool GraphExecutor::initialize(
  runtime::PrimitiveGrpah& graph,
  const engine::WorkspaceHandle& workspace_handle,
  const std::map<runtime::ExecutorType, int>& executor_config,
  std::shared_ptr<communicator::ChannelManager> channel_manager,
  std::shared_ptr<MemoryManager> memory_manager
) {
  return impl_->initialize(graph, workspace_handle, executor_config, channel_manager, memory_manager);
}

void GraphExecutor::issue() {
  impl_->issue();
}

void GraphExecutor::wait() {
  impl_->wait();
}

void GraphExecutor::initialize_ready_queues() {
  impl_->initialize_ready_queues();
}

} // namespace pccl::engine
