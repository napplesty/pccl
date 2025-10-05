#include "runtime/engine/graph_executor.h"
#include "utils/logging.h"
#include <thread>
#include <atomic>

namespace pccl::engine {

class GraphExecutor::Impl {
public:
  Impl(GraphBufferLayout* graph_layout) : graph_layout_(graph_layout), running_(false) {}
  
  bool initialize(const std::map<ExecutorType, int>& executor_config) {
    if (!graph_layout_) {
      PCCL_LOG_ERROR("Invalid graph layout");
      return false;
    }
    
    executor_config_ = executor_config;
    initialize_ready_queues();
    PCCL_LOG_INFO("Graph executor initialized");
    return true;
  }
  
  void start() {
    if (running_) return;
    
    running_ = true;
    execution_thread_ = std::thread([this]() { execution_loop(); });
    PCCL_LOG_INFO("Graph executor started");
  }
  
  void stop() {
    running_ = false;
    if (execution_thread_.joinable()) {
      execution_thread_.join();
    }
    PCCL_LOG_INFO("Graph executor stopped");
  }
  
  void wait() {
    if (execution_thread_.joinable()) {
      execution_thread_.join();
    }
  }
  
  void initialize_ready_queues() {
    if (!graph_layout_ || !graph_layout_->ready_queues) return;
    
    for (uint64_t i = 0; i < graph_layout_->num_queues; ++i) {
      auto& queue = graph_layout_->ready_queues[i];
      queue.head = 0;
      queue.tail = 0;
      
      for (uint64_t j = 0; j < graph_layout_->num_operators; ++j) {
        auto& op = graph_layout_->operators[j];
        if (op.dependency_count == 0 && op.executor_type == queue.consumer_type) {
          if (queue.tail < queue.capacity) {
            OperatorLayout** buffer = static_cast<OperatorLayout**>(queue.buffer);
            buffer[queue.tail++] = &op;
          }
        }
      }
    }
  }

private:
  void execution_loop() {
    while (running_) {
      bool work_done = false;
      
      for (uint64_t i = 0; i < graph_layout_->num_queues; ++i) {
        auto& queue = graph_layout_->ready_queues[i];
        if (queue.head == queue.tail) continue;
        
        OperatorLayout** buffer = static_cast<OperatorLayout**>(queue.buffer);
        OperatorLayout* primitive = buffer[queue.head];
        queue.head = (queue.head + 1) % queue.capacity;
        
        if (primitive && primitive->remaining_executors > 0) {
          execute_primitive(primitive);
          work_done = true;
          
          if (--primitive->remaining_executors == 0) {
            for (int j = 0; j < primitive->num_next; ++j) {
              OperatorLayout* next_primitive = primitive->next_operators[j];
              if (next_primitive && --next_primitive->dependency_count == 0) {
                add_to_ready_queue(next_primitive);
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
  
  void execute_primitive(OperatorLayout* primitive) {
    const auto& config = primitive->primitive_config;
    
    switch (config.type) {
      case PrimitiveType::COPY:
        execute_copy(primitive);
        break;
      case PrimitiveType::WRITE:
        execute_write(primitive);
        break;
      case PrimitiveType::COMPUTE:
        execute_compute(primitive);
        break;
      case PrimitiveType::SIGNAL:
        execute_signal(primitive);
        break;
      case PrimitiveType::WAITSIGNAL:
        execute_waitsignal(primitive);
        break;
      default:
        PCCL_LOG_WARN("Unknown primitive type: {}", static_cast<int>(config.type));
        break;
    }
  }
  
  void execute_copy(OperatorLayout* primitive) {
    const auto& config = primitive->primitive_config;
    if (!config.src_buffer || !config.dst_buffer || config.data_size == 0) return;
    
    std::memcpy(config.dst_buffer, config.src_buffer, config.data_size);
    PCCL_LOG_DEBUG("Executed COPY primitive, size: {}", config.data_size);
  }
  
  void execute_write(OperatorLayout* primitive) {
    const auto& config = primitive->primitive_config;
    if (!config.dst_buffer || config.data_size == 0) return;
    
    std::memset(config.dst_buffer, 0, config.data_size);
    PCCL_LOG_DEBUG("Executed WRITE primitive, size: {}", config.data_size);
  }
  
  void execute_compute(OperatorLayout* primitive) {
    const auto& config = primitive->primitive_config;
    if (!config.src_buffer || !config.dst_buffer || config.data_size == 0) return;
    
    switch (config.dtype) {
      case DataType::F32:
        execute_compute_float<float>(primitive);
        break;
      case DataType::F16:
        execute_compute_float<uint16_t>(primitive);
        break;
      case DataType::BF16:
        execute_compute_float<uint16_t>(primitive);
        break;
      default:
        PCCL_LOG_WARN("Unsupported data type for compute: {}", static_cast<int>(config.dtype));
        break;
    }
  }
  
  template<typename T>
  void execute_compute_float(OperatorLayout* primitive) {
    const auto& config = primitive->primitive_config;
    const T* src = static_cast<const T*>(config.src_buffer);
    T* dst = static_cast<T*>(config.dst_buffer);
    size_t num_elements = config.data_size / sizeof(T);
    
    switch (config.compute_op) {
      case ComputeType::SUM:
        for (size_t i = 0; i < num_elements; ++i) {
          dst[i] += src[i];
        }
        break;
      case ComputeType::PROD:
        for (size_t i = 0; i < num_elements; ++i) {
          dst[i] *= src[i];
        }
        break;
      case ComputeType::MAX:
        for (size_t i = 0; i < num_elements; ++i) {
          dst[i] = std::max(dst[i], src[i]);
        }
        break;
      case ComputeType::MIN:
        for (size_t i = 0; i < num_elements; ++i) {
          dst[i] = std::min(dst[i], src[i]);
        }
        break;
      default:
        PCCL_LOG_WARN("Unsupported compute operation: {}", static_cast<int>(config.compute_op));
        break;
    }
    
    PCCL_LOG_DEBUG("Executed COMPUTE primitive, elements: {}", num_elements);
  }
  
  void execute_signal(OperatorLayout* primitive) {
    const auto& config = primitive->primitive_config;
    PCCL_LOG_DEBUG("Executed SIGNAL primitive, value: {}", config.signal_value);
  }
  
  void execute_waitsignal(OperatorLayout* primitive) {
    const auto& config = primitive->primitive_config;
    PCCL_LOG_DEBUG("Executed WAITSIGNAL primitive, value: {}", config.signal_value);
  }
  
  void add_to_ready_queue(OperatorLayout* primitive) {
    if (!graph_layout_ || !graph_layout_->ready_queues) return;
    
    for (uint64_t i = 0; i < graph_layout_->num_queues; ++i) {
      auto& queue = graph_layout_->ready_queues[i];
      if (queue.consumer_type == primitive->executor_type) {
        if (queue.tail < queue.capacity) {
          OperatorLayout** buffer = static_cast<OperatorLayout**>(queue.buffer);
          buffer[queue.tail++] = primitive;
          break;
        }
      }
    }
  }
  
  GraphBufferLayout* graph_layout_;
  std::map<ExecutorType, int> executor_config_;
  std::atomic<bool> running_;
  std::thread execution_thread_;
};

GraphExecutor::GraphExecutor() : impl_(std::make_unique<Impl>(nullptr)) {}

GraphExecutor::~GraphExecutor() {
  stop();
  wait();
}

bool GraphExecutor::initialize(GraphBufferLayout* graph_layout, const std::map<ExecutorType, int>& executor_config) {
  impl_ = std::make_unique<Impl>(graph_layout);
  return impl_->initialize(executor_config);
}

void GraphExecutor::start() {
  if (impl_) impl_->start();
}

void GraphExecutor::stop() {
  if (impl_) impl_->stop();
}

void GraphExecutor::wait() {
  if (impl_) impl_->wait();
}

void GraphExecutor::initialize_ready_queues() {
  if (impl_) impl_->initialize_ready_queues();
}

} // namespace pccl::engine
