#include "plugins/ahost/executor.h"
#include "utils/logging.h"
#include "utils/allocator.h"
#include <atomic>
#include <thread>
#include <cstring>

namespace pccl::engine::host {

HostExecutorManager::HostExecutorManager(GraphBufferLayout* graph_layout, 
                                        int num_threads, 
                                        ReadyQueueLayout* ready_queues,
                                        int num_queues,
                                        std::shared_ptr<communicator::ChannelManager> channel_manager,
                                        std::shared_ptr<MemoryManager> memory_manager)
  : graph_layout_(graph_layout),
    num_threads_(num_threads),
    ready_queues_(ready_queues),
    num_queues_(num_queues),
    channel_manager_(channel_manager),
    memory_manager_(memory_manager),
    running_(false) {
}

HostExecutorManager::~HostExecutorManager() {
  stop();
}

bool HostExecutorManager::initialize() {
  if (num_threads_ <= 0) {
    PCCL_LOG_ERROR("Invalid number of threads: {}", num_threads_);
    return false;
  }
  PCCL_LOG_DEBUG("Initialized host executor with {} threads", num_threads_);
  return true;
}

void HostExecutorManager::launch() {
  if (running_) return;

  running_ = true;
  executors_.clear();
  
  for (int i = 0; i < num_threads_; ++i) {
    executors_.emplace_back([this, i]() { worker_thread(i); });
  }
  PCCL_LOG_DEBUG("Launched {} host executor threads", num_threads_);
}

void HostExecutorManager::wait() {
  for (auto& thread : executors_) {
    if (thread.joinable()) {
      thread.join();
    }
  }
  executors_.clear();
}

void HostExecutorManager::stop() {
  running_ = false;
  wait();
}

void HostExecutorManager::worker_thread(int thread_id) {
  PCCL_LOG_DEBUG("Host executor thread {} started", thread_id);
  
  while (running_) {
    bool work_done = false;
    
    for (int queue_idx = 0; queue_idx < num_queues_; ++queue_idx) {
      ReadyQueueLayout* queue = &ready_queues_[queue_idx];
      if (queue->consumer_type != runtime::ExecutorType::CPU) continue;

      OperatorLayout* primitive = nullptr;
      int execute_index = -1;
      
      if (try_pop_primitive(queue, &primitive, &execute_index)) {
        execute_primitive(primitive, execute_index);
        update_dependencies(primitive);
        work_done = true;
        break;
      }
    }

    if (!work_done) {
      std::this_thread::yield();
    }
  }
  PCCL_LOG_DEBUG("Host executor thread {} stopped", thread_id);
}

bool HostExecutorManager::try_pop_primitive(ReadyQueueLayout* queue, OperatorLayout** primitive, int* execute_index) {
  uint64_t current_head = queue->head;
  uint64_t current_tail = queue->tail;
  
  if (current_head == current_tail) return false;
  
  OperatorLayout** buffer = static_cast<OperatorLayout**>(queue->buffer);
  *primitive = buffer[current_head];
  
  int required_executor = std::atomic_fetch_sub(reinterpret_cast<std::atomic<int>*>(&(*primitive)->required_executors), 1);
  if (required_executor <= 0) return false;
  
  *execute_index = required_executor - 1;
  
  uint64_t new_head = (current_head + 1) % queue->capacity;
  queue->head = new_head;
  return true;
}

void HostExecutorManager::execute_primitive(OperatorLayout* primitive, int execute_index) {
  const PrimitiveConfig& config = *reinterpret_cast<const PrimitiveConfig*>(&primitive->primitive_config);
  
  switch (config.type) {
    case runtime::PrimitiveType::WRITE:
      execute_write_primitive(primitive, execute_index, config);
      break;
    case runtime::PrimitiveType::SIGNAL:
      execute_notify_primitive(primitive, execute_index, config);
      break;
    case runtime::PrimitiveType::WAITSIGNAL:
      execute_get_notify_primitive(primitive, execute_index, config);
      break;
    default:
      PCCL_LOG_WARN("Unsupported primitive type for CPU executor: {}", static_cast<int>(config.type));
      break;
  }
}

void HostExecutorManager::execute_write_primitive(OperatorLayout* primitive, int execute_index, const PrimitiveConfig& config) {
  PCCL_LOG_DEBUG("Executing WRITE primitive {} on executor {}", primitive->uid, execute_index);
  
  if (!config.src_buffer || config.data_size == 0) {
    PCCL_LOG_ERROR("Invalid WRITE primitive parameters");
    return;
  }

  if (channel_manager_) {
    auto target_endpoint = create_endpoint_for_rank(config.target_rank);
    auto channel = channel_manager_->getChannel(target_endpoint);
    if (channel) {
      communicator::MemRegion src_region;
      src_region.ptr_ = config.src_buffer;
      src_region.size_ = config.data_size;
      
      communicator::MemRegion dst_region;
      dst_region.ptr_ = config.dst_buffer;
      dst_region.size_ = config.data_size;
      
      uint64_t tx_id = channel->prepSend(dst_region, src_region);
      if (tx_id != 0) {
        channel->postSend();
        if (channel->waitTx(tx_id)) {
          PCCL_LOG_DEBUG("WRITE primitive {} completed successfully", primitive->uid);
        } else {
          PCCL_LOG_ERROR("WRITE primitive {} failed", primitive->uid);
        }
      }
    }
  }
}

void HostExecutorManager::execute_notify_primitive(OperatorLayout* primitive, int execute_index, const PrimitiveConfig& config) {
  PCCL_LOG_DEBUG("Executing NOTIFY primitive {} on executor {}", primitive->uid, execute_index);
  
  if (config.target_rank < 0 || config.signal_value == 0) {
    PCCL_LOG_ERROR("Invalid NOTIFY primitive parameters");
    return;
  }

  void* signal_buffer = primitive->primitive_config.dst_buffer;
  if (!signal_buffer) {
    PCCL_LOG_ERROR("Failed to get signal buffer for NOTIFY primitive {}", primitive->uid);
    return;
  }

  if (channel_manager_) {
    auto target_endpoint = create_endpoint_for_rank(config.target_rank);
    auto channel = channel_manager_->getChannel(target_endpoint);
    if (channel) {
      communicator::MemRegion src_region;
      src_region.ptr_ = const_cast<int *>(&config.signal_value);
      src_region.size_ = sizeof(config.signal_value);
      
      communicator::MemRegion dst_region;
      dst_region.ptr_ = signal_buffer;
      dst_region.size_ = sizeof(config.signal_value);
      
      uint64_t tx_id = channel->prepSend(dst_region, src_region);
      if (tx_id != 0) {
        channel->postSend();
        if (channel->waitTx(tx_id)) {
          PCCL_LOG_DEBUG("NOTIFY primitive {} completed successfully", primitive->uid);
        } else {
          PCCL_LOG_ERROR("NOTIFY primitive {} failed", primitive->uid);
        }
      }
    }
  }
}

void HostExecutorManager::execute_get_notify_primitive(OperatorLayout* primitive, int execute_index, const PrimitiveConfig& config) {
  PCCL_LOG_DEBUG("Executing GET_NOTIFY primitive {} on executor {}", primitive->uid, execute_index);
  
  if (config.signal_value == 0) {
    PCCL_LOG_ERROR("Invalid GET_NOTIFY primitive parameters");
    return;
  }

  void* signal_buffer = primitive->primitive_config.src_buffer;
  if (!signal_buffer) {
    PCCL_LOG_ERROR("Failed to get signal buffer for GET_NOTIFY primitive {}", primitive->uid);
    return;
  }

  int* signal_ptr = static_cast<int*>(signal_buffer);
  const int max_retries = 100000;
  int retry_count = 0;
  
  while (retry_count < max_retries) {
    if (*signal_ptr == config.signal_value) {
      PCCL_LOG_DEBUG("GET_NOTIFY primitive {} completed after {} retries", primitive->uid, retry_count);
      return;
    }
    retry_count++;
    std::this_thread::yield();
  }
  
  PCCL_LOG_ERROR("GET_NOTIFY primitive {} timeout after {} retries", primitive->uid, max_retries);
}

void* HostExecutorManager::get_signal_buffer_address(int buffer_idx, int rank) {
  if (!memory_manager_ || buffer_idx < 0) return nullptr;
  
  auto buffer = memory_manager_->getBuffer(rank, runtime::ExecutorType::CPU, buffer_idx);
  return buffer.addr;
}

communicator::Endpoint HostExecutorManager::create_endpoint_for_rank(int rank) {
  communicator::Endpoint endpoint;
  endpoint.attributes_["pccl.runtime.rank"] = std::to_string(rank);
  return endpoint;
}

void HostExecutorManager::update_dependencies(OperatorLayout* primitive) {
  int old_remaining = std::atomic_fetch_sub(reinterpret_cast<std::atomic<int>*>(&primitive->remaining_executors), 1);
  if (old_remaining != 1) return;

  for (int j = 0; j < primitive->num_next; ++j) {
    OperatorLayout* next_primitive = primitive->next_operators[j];
    if (!next_primitive) continue;

    int old_dependency = std::atomic_fetch_sub(reinterpret_cast<std::atomic<int>*>(&next_primitive->dependency_count), 1);
    if (old_dependency == 1) {
      enqueue_primitive(next_primitive);
    }
  }
}

void HostExecutorManager::enqueue_primitive(OperatorLayout* primitive) {
  for (uint64_t k = 0; k < num_queues_; ++k) {
    ReadyQueueLayout* queue = &ready_queues_[k];
    if (queue->producer_type == runtime::ExecutorType::CPU && 
        queue->consumer_type == primitive->executor_type) {
      if (try_push_primitive(queue, primitive)) {
        PCCL_LOG_DEBUG("Enqueued primitive {} to queue {}", primitive->uid, k);
        break;
      }
    }
  }
}

bool HostExecutorManager::try_push_primitive(ReadyQueueLayout* queue, OperatorLayout* primitive) {
  uint64_t current_tail = queue->tail;
  uint64_t current_head = queue->head;
  
  uint64_t next_tail = (current_tail + 1) % queue->capacity;
  if (next_tail == current_head) return false;
  
  OperatorLayout** buffer = static_cast<OperatorLayout**>(queue->buffer);
  buffer[current_tail] = primitive;
  queue->tail = next_tail;
  return true;
}

} // namespace pccl::engine::host
