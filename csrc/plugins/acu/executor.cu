#include "runtime/engine/graph_executor.h"
#include <plugins/acu/executor.h>
#include <plugins/acu/kernel_impls/tensor_ops.h>
#include <cuda_runtime.h>
#include <cuda/atomic>

namespace pccl::engine::cuda {

__global__ void cuda_executor_kernel(GraphBufferLayout* graph_layout) {
  int thread_id = threadIdx.x;
  int block_id = blockIdx.x;
  int total_threads = blockDim.x * gridDim.x;

  __shared__ OperatorLayout* shared_primitive[1];
  __shared__ int execute_index_shared[1];
  __shared__ bool work_done[1];

  for (int try_time = 0; try_time < graph_layout->num_operators; try_time++) {
    if (thread_id == 0) {
      work_done[0] = false;
      shared_primitive[0] = nullptr;
      
      for (int queue_idx = 0; queue_idx < graph_layout->num_queues; queue_idx++) {
        ReadyQueueLayout* queue = &graph_layout->ready_queues[queue_idx];
        if (queue->consumer_type != ExecutorType::CUDA) continue;
        
        OperatorLayout* primitive = nullptr;
        int executor_index;
        if (cuda_try_pop_primitive(queue, &primitive, &executor_index)) {
          shared_primitive[0] = primitive;
          work_done[0] = true;
          execute_index_shared[0] = executor_index;
          break;
        }
      }
    }

    __syncthreads();

    if (!work_done[0]) {
      continue;
    }

    cuda_execute_primitive(shared_primitive[0], execute_index_shared[0]);

    __syncthreads();

    if (thread_id == 0 && shared_primitive[0] != nullptr) {
      int old_remaining = atomicSub(&shared_primitive[0]->remaining_executors, 1);
      if (old_remaining == 1) {
        for (int j = 0; j < shared_primitive[0]->num_next; ++j) {
          OperatorLayout* next_primitive = shared_primitive[0]->next_operators[j];
          if (next_primitive != nullptr) {
            int old_dependency = atomicSub(&next_primitive->dependency_count, 1);
            if (old_dependency == 1) {
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
  }
}

__device__ bool cuda_try_pop_primitive(ReadyQueueLayout* queue, OperatorLayout** primitive, int* execute_index) {
  uint64_t current_head = atomicAdd(&queue->head, 0);
  uint64_t current_tail = atomicAdd(&queue->tail, 0);
  
  if (current_head == current_tail) return false;
  
  OperatorLayout** buffer = static_cast<OperatorLayout**>(queue->buffer);
  *primitive = buffer[current_head];
  
  int required_executor = atomicAdd(&(*primitive)->required_executors, -1);
  if (required_executor <= 0) return false;
  
  *execute_index = required_executor - 1;
  
  uint64_t new_head = (current_head + 1) % queue->capacity;
  uint64_t expected = current_head;
  if (execute_index) {
    atomicCAS(&queue->head, expected, new_head);
  }
  return true;
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
