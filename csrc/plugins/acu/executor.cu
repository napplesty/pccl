#include <plugins/acu/executor.h>
#include <plugins/acu/kernel_impls/tensor_ops.h>
#include <cuda_runtime.h>
#include <cuda/atomic>

namespace pccl::engine::cuda {

CUDAExecutorManager::CUDAExecutorManager(GraphBufferLayout* graph_layout, int num_sms)
    : graph_layout_(graph_layout), num_sms_(num_sms) {
}

CUDAExecutorManager::~CUDAExecutorManager() {
  stop();
  wait();
}

bool CUDAExecutorManager::initialize() {
  int device_count;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  if (err != cudaSuccess || device_count == 0) {
    return false;
  }
  
  err = cudaSetDevice(0);
  if (err != cudaSuccess) {
    return false;
  }
  
  initialized_ = true;
  return true;
}

void CUDAExecutorManager::start(cudaStream_t stream) {
  if (!initialized_ || !graph_layout_) return;
  
  current_stream_ = stream;
  int threads_per_block = 256;
  int num_blocks = num_sms_;
  
  if (current_stream_ == 0) {
    cudaStreamCreate(&current_stream_);
  }
  
  launch_cuda_kernel(graph_layout_, num_blocks, threads_per_block, current_stream_);
  
  if (stream == 0) {
    cudaStreamSynchronize(current_stream_);
    cudaStreamDestroy(current_stream_);
    current_stream_ = 0;
  }
}

void CUDAExecutorManager::stop() {
}

void CUDAExecutorManager::wait() {
  if (current_stream_ != 0) {
    cudaStreamSynchronize(current_stream_);
  } else {
    cudaDeviceSynchronize();
  }
}

void CUDAExecutorManager::launch_cuda_kernel(GraphBufferLayout* graph_layout, int num_blocks, int threads_per_block, cudaStream_t stream) {
  cuda_executor_kernel<<<num_blocks, threads_per_block, 0, stream>>>(graph_layout);
}

__global__ void cuda_executor_kernel(GraphBufferLayout* graph_layout) {
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  int total_threads = gridDim.x * blockDim.x;
  
  for (int round = 0; round < 1000; ++round) {
    bool work_done = false;
    
    for (uint64_t i = thread_id; i < graph_layout->num_queues; i += total_threads) {
      ReadyQueueLayout* queue = &graph_layout->ready_queues[i];
      if (queue->consumer_type != ExecutorType::CUDA) continue;
      
      OperatorLayout* primitive = nullptr;
      int execute_index = -1;
      if (cuda_try_pop_primitive(queue, &primitive, &execute_index)) {
        cuda_execute_primitive(primitive, execute_index);
        work_done = true;
        
        if (atomicSub(&primitive->remaining_executors, 1) == 1) {
          for (int j = 0; j < primitive->num_next; ++j) {
            OperatorLayout* next_primitive = primitive->next_operators[j];
            if (atomicSub(&next_primitive->dependency_count, 1) == 1) {
              for (uint64_t k = 0; k < graph_layout->num_queues; ++k) {
                ReadyQueueLayout* next_queue = &graph_layout->ready_queues[k];
                if (next_queue->producer_type == ExecutorType::CUDA && 
                    next_queue->consumer_type == next_primitive->executor_type) {
                  cuda_try_push_primitive(next_queue, next_primitive);
                  break;
                }
              }
            }
          }
        }
      }
    }
    
    __syncthreads();
    
    if (!work_done) {
      break;
    }
  }
}

__device__ bool cuda_try_pop_primitive(ReadyQueueLayout* queue, OperatorLayout** primitive, int* execute_index) {
  uint64_t current_head = atomicAdd(&queue->head, 0);
  uint64_t current_tail = atomicAdd(&queue->tail, 0);
  
  if (current_head == current_tail) return false;
  
  OperatorLayout** buffer = static_cast<OperatorLayout**>(queue->buffer);
  *primitive = buffer[current_head];
  
  int old_remaining = atomicAdd(&(*primitive)->remaining_executors, 0);
  if (old_remaining <= 0) return false;
  
  *execute_index = old_remaining - 1;
  
  uint64_t new_head = (current_head + 1) % queue->capacity;
  uint64_t expected = current_head;
  
  if (atomicCAS(&queue->head, expected, new_head) == expected) {
    return true;
  }
  
  return false;
}

__device__ bool cuda_try_push_primitive(ReadyQueueLayout* queue, OperatorLayout* primitive) {
  uint64_t current_tail = atomicAdd(&queue->tail, 0);
  uint64_t current_head = atomicAdd(&queue->head, 0);
  
  uint64_t next_tail = (current_tail + 1) % queue->capacity;
  if (next_tail == current_head) return false;
  
  OperatorLayout** buffer = static_cast<OperatorLayout**>(queue->buffer);
  buffer[current_tail] = primitive;
  
  atomicAdd(&queue->tail, 1);
  return true;
}

__device__ void cuda_execute_primitive(OperatorLayout* primitive, int execute_index) {
  const PrimitiveConfig& config = primitive->primitive_config;
  int total_executors = primitive->required_executors;
  
  switch (config.type) {
    case PrimitiveType::COPY:
      if (config.src_buffer && config.dst_buffer && config.data_size > 0) {
        switch (config.dtype) {
          case DataType::F32:
            pccl::acu::direct_copy_impl<float, 4>(
                static_cast<const float*>(config.src_buffer),
                static_cast<float*>(config.dst_buffer),
                config.data_size, execute_index, total_executors);
            break;
          case DataType::F16:
            pccl::acu::direct_copy_impl<half, 4>(
                static_cast<const half*>(config.src_buffer),
                static_cast<half*>(config.dst_buffer),
                config.data_size, execute_index, total_executors);
            break;
          case DataType::BF16:
            pccl::acu::direct_copy_impl<__nv_bfloat16, 4>(
                static_cast<const __nv_bfloat16*>(config.src_buffer),
                static_cast<__nv_bfloat16*>(config.dst_buffer),
                config.data_size, execute_index, total_executors);
            break;
        }
      }
      break;
      
    case PrimitiveType::WRITE:
      if (config.dst_buffer && config.data_size > 0) {
        switch (config.dtype) {
          case DataType::F32:
            pccl::acu::direct_write_impl<float, 4>(
                static_cast<float*>(config.dst_buffer),
                config.data_size, execute_index, total_executors);
            break;
          case DataType::F16:
            pccl::acu::direct_write_impl<half, 4>(
                static_cast<half*>(config.dst_buffer),
                config.data_size, execute_index, total_executors);
            break;
          case DataType::BF16:
            pccl::acu::direct_write_impl<__nv_bfloat16, 4>(
                static_cast<__nv_bfloat16*>(config.dst_buffer),
                config.data_size, execute_index, total_executors);
            break;
        }
      }
      break;
      
    case PrimitiveType::COMPUTE:
      if (config.src_buffer && config.dst_buffer && config.data_size > 0) {
        switch (config.dtype) {
          case DataType::F32:
            switch (config.compute_op) {
              case ComputeType::SUM:
                pccl::acu::tensor_add_impl<float, 4>(
                    static_cast<const float*>(config.src_buffer),
                    static_cast<float*>(config.dst_buffer),
                    config.data_size, execute_index, total_executors);
                break;
              case ComputeType::PROD:
                pccl::acu::tensor_multiply_impl<float, 4>(
                    static_cast<const float*>(config.src_buffer),
                    static_cast<float*>(config.dst_buffer),
                    config.data_size, execute_index, total_executors);
                break;
              case ComputeType::MAX:
                pccl::acu::tensor_max_impl<float, 4>(
                    static_cast<const float*>(config.src_buffer),
                    static_cast<float*>(config.dst_buffer),
                    config.data_size, execute_index, total_executors);
                break;
              case ComputeType::MIN:
                pccl::acu::tensor_min_impl<float, 4>(
                    static_cast<const float*>(config.src_buffer),
                    static_cast<float*>(config.dst_buffer),
                    config.data_size, execute_index, total_executors);
                break;
            }
            break;
          case DataType::F16:
            switch (config.compute_op) {
              case ComputeType::SUM:
                pccl::acu::tensor_add_impl<half, 4>(
                    static_cast<const half*>(config.src_buffer),
                    static_cast<half*>(config.dst_buffer),
                    config.data_size, execute_index, total_executors);
                break;
              case ComputeType::PROD:
                pccl::acu::tensor_multiply_impl<half, 4>(
                    static_cast<const half*>(config.src_buffer),
                    static_cast<half*>(config.dst_buffer),
                    config.data_size, execute_index, total_executors);
                break;
              case ComputeType::MAX:
                pccl::acu::tensor_max_impl<half, 4>(
                    static_cast<const half*>(config.src_buffer),
                    static_cast<half*>(config.dst_buffer),
                    config.data_size, execute_index, total_executors);
                break;
              case ComputeType::MIN:
                pccl::acu::tensor_min_impl<half, 4>(
                    static_cast<const half*>(config.src_buffer),
                    static_cast<half*>(config.dst_buffer),
                    config.data_size, execute_index, total_executors);
                break;
            }
            break;
          case DataType::BF16:
            switch (config.compute_op) {
              case ComputeType::SUM:
                pccl::acu::tensor_add_impl<__nv_bfloat16, 4>(
                    static_cast<const __nv_bfloat16*>(config.src_buffer),
                    static_cast<__nv_bfloat16*>(config.dst_buffer),
                    config.data_size, execute_index, total_executors);
                break;
              case ComputeType::PROD:
                pccl::acu::tensor_multiply_impl<__nv_bfloat16, 4>(
                    static_cast<const __nv_bfloat16*>(config.src_buffer),
                    static_cast<__nv_bfloat16*>(config.dst_buffer),
                    config.data_size, execute_index, total_executors);
                break;
              case ComputeType::MAX:
                pccl::acu::tensor_max_impl<__nv_bfloat16, 4>(
                    static_cast<const __nv_bfloat16*>(config.src_buffer),
                    static_cast<__nv_bfloat16*>(config.dst_buffer),
                    config.data_size, execute_index, total_executors);
                break;
              case ComputeType::MIN:
                pccl::acu::tensor_min_impl<__nv_bfloat16, 4>(
                    static_cast<const __nv_bfloat16*>(config.src_buffer),
                    static_cast<__nv_bfloat16*>(config.dst_buffer),
                    config.data_size, execute_index, total_executors);
                break;
            }
            break;
        }
      }
      break;
      
    default:
      break;
  }
}

}
