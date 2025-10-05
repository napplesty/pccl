#include "plugins/ahost/executor.h"
#include "runtime/engine/graph_executor.h"
#include "utils/logging.h"
#include <thread>
#include <atomic>
#include <vector>

namespace pccl::engine::host {

class HostExecutorManager {
public:
  HostExecutorManager(GraphBufferLayout* graph_layout, int num_threads)
    : graph_layout_(graph_layout), num_threads_(num_threads), running_(false) {}
  
  bool initialize() {
    if (!graph_layout_) {
      PCCL_LOG_ERROR("Invalid graph layout for host executor");
      return false;
    }
    
    threads_.reserve(num_threads_);
    PCCL_LOG_INFO("Host executor manager initialized with {} threads", num_threads_);
    return true;
  }
  
  void start() {
    if (running_) return;
    
    running_ = true;
    for (int i = 0; i < num_threads_; ++i) {
      threads_.emplace_back([this, i]() { worker_loop(i); });
    }
    PCCL_LOG_INFO("Host executor manager started");
  }
  
  void stop() {
    running_ = false;
    for (auto& thread : threads_) {
      if (thread.joinable()) {
        thread.join();
      }
    }
    threads_.clear();
    PCCL_LOG_INFO("Host executor manager stopped");
  }
  
  void wait() {
    for (auto& thread : threads_) {
      if (thread.joinable()) {
        thread.join();
      }
    }
  }

private:
  void worker_loop(int thread_id) {
    while (running_) {
      bool work_done = false;
      
      for (uint64_t i = 0; i < graph_layout_->num_queues; ++i) {
        ReadyQueueLayout* queue = &graph_layout_->ready_queues[i];
        if (queue->consumer_type != ExecutorType::CPU) continue;
        
        OperatorLayout* primitive = nullptr;
        int execute_index = -1;
        
        if (try_pop_primitive(queue, &primitive, &execute_index)) {
          execute_primitive(primitive, execute_index);
          work_done = true;
          
          if (--primitive->remaining_executors == 0) {
            for (int j = 0; j < primitive->num_next; ++j) {
              OperatorLayout* next_primitive = primitive->next_operators[j];
              if (--next_primitive->dependency_count == 0) {
                for (uint64_t k = 0; k < graph_layout_->num_queues; ++k) {
                  ReadyQueueLayout* next_queue = &graph_layout_->ready_queues[k];
                  if (next_queue->producer_type == ExecutorType::CPU && 
                      next_queue->consumer_type == next_primitive->executor_type) {
                    try_push_primitive(next_queue, next_primitive);
                    break;
                  }
                }
              }
            }
          }
        }
      }
      
      if (!work_done) {
        std::this_thread::sleep_for(std::chrono::microseconds(100));
      }
    }
  }
  
  bool try_pop_primitive(ReadyQueueLayout* queue, OperatorLayout** primitive, int* execute_index) {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    
    if (queue->head == queue->tail) return false;
    
    OperatorLayout** buffer = static_cast<OperatorLayout**>(queue->buffer);
    *primitive = buffer[queue->head];
    
    int old_remaining = (*primitive)->remaining_executors;
    if (old_remaining <= 0) return false;
    
    *execute_index = old_remaining - 1;
    queue->head = (queue->head + 1) % queue->capacity;
    
    return true;
  }
  
  bool try_push_primitive(ReadyQueueLayout* queue, OperatorLayout* primitive) {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    
    uint64_t next_tail = (queue->tail + 1) % queue->capacity;
    if (next_tail == queue->head) return false;
    
    OperatorLayout** buffer = static_cast<OperatorLayout**>(queue->buffer);
    buffer[queue->tail] = primitive;
    queue->tail = next_tail;
    
    return true;
  }
  
  void execute_primitive(OperatorLayout* primitive, int execute_index) {
    const PrimitiveConfig& config = primitive->primitive_config;
    int total_executors = primitive->required_executors;
    
    switch (config.type) {
      case PrimitiveType::COPY:
        if (config.src_buffer && config.dst_buffer && config.data_size > 0) {
          std::memcpy(config.dst_buffer, config.src_buffer, config.data_size);
        }
        break;
        
      case PrimitiveType::WRITE:
        if (config.dst_buffer && config.data_size > 0) {
          std::memset(config.dst_buffer, 0, config.data_size);
        }
        break;
        
      case PrimitiveType::COMPUTE:
        if (config.src_buffer && config.dst_buffer && config.data_size > 0) {
          execute_compute_operation(primitive, execute_index, total_executors);
        }
        break;
        
      default:
        break;
    }
  }
  
  void execute_compute_operation(OperatorLayout* primitive, int execute_index, int total_executors) {
    const PrimitiveConfig& config = primitive->primitive_config;
    size_t data_size = config.data_size;
    
    switch (config.dtype) {
      case DataType::F32:
        execute_compute_float<float>(primitive, execute_index, total_executors);
        break;
      case DataType::F16:
        execute_compute_float<uint16_t>(primitive, execute_index, total_executors);
        break;
      case DataType::BF16:
        execute_compute_float<uint16_t>(primitive, execute_index, total_executors);
        break;
      default:
        break;
    }
  }
  
  template<typename T>
  void execute_compute_float(OperatorLayout* primitive, int execute_index, int total_executors) {
    const PrimitiveConfig& config = primitive->primitive_config;
    const T* src = static_cast<const T*>(config.src_buffer);
    T* dst = static_cast<T*>(config.dst_buffer);
    size_t num_elements = config.data_size / sizeof(T);
    
    size_t elements_per_executor = (num_elements + total_executors - 1) / total_executors;
    size_t start_idx = execute_index * elements_per_executor;
    size_t end_idx = std::min(start_idx + elements_per_executor, num_elements);
    
    switch (config.compute_op) {
      case ComputeType::SUM:
        for (size_t i = start_idx; i < end_idx; ++i) {
          dst[i] += src[i];
        }
        break;
      case ComputeType::PROD:
        for (size_t i = start_idx; i < end_idx; ++i) {
          dst[i] *= src[i];
        }
        break;
      case ComputeType::MAX:
        for (size_t i = start_idx; i < end_idx; ++i) {
          if (src[i] > dst[i]) dst[i] = src[i];
        }
        break;
      case ComputeType::MIN:
        for (size_t i = start_idx; i < end_idx; ++i) {
          if (src[i] < dst[i]) dst[i] = src[i];
        }
        break;
      default:
        break;
    }
  }
  
  GraphBufferLayout* graph_layout_;
  int num_threads_;
  std::atomic<bool> running_;
  std::vector<std::thread> threads_;
  std::mutex queue_mutex_;
};

HostExecutorManager* create_host_executor_manager(GraphBufferLayout* graph_layout, int num_threads) {
  return new HostExecutorManager(graph_layout, num_threads);
}

void destroy_host_executor_manager(HostExecutorManager* manager) {
  if (manager) {
    manager->stop();
    delete manager;
  }
}

bool initialize_host_executor(HostExecutorManager* manager) {
  return manager ? manager->initialize() : false;
}

void start_host_executor(HostExecutorManager* manager) {
  if (manager) manager->start();
}

void stop_host_executor(HostExecutorManager* manager) {
  if (manager) manager->stop();
}

void wait_host_executor(HostExecutorManager* manager) {
  if (manager) manager->wait();
}

} // namespace pccl::engine::host
