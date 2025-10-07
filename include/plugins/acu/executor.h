#pragma once
#include <runtime/engine/graph_executor.h>
#include <cuda_runtime.h>

namespace pccl::engine::cuda {

class CUDAExecutorManager {
public:
  CUDAExecutorManager(GraphBufferLayout* graph_layout, 
                      int num_sms, 
                      ReadyQueueLayout *ready_queues,
                      int num_queues,
                      cudaStream_t stream = nullptr);
  ~CUDAExecutorManager();
  bool initialize();
  void launch();
private:
  GraphBufferLayout* graph_layout_;
  int num_sms_;
  ReadyQueueLayout *ready_queues_;
  int num_queues_;
  cudaStream_t current_stream_{0};
};

__global__ void cuda_executor_kernel(GraphBufferLayout* graph_layout);
__device__ bool cuda_try_pop_primitive(ReadyQueueLayout* queue, OperatorLayout** primitive, int* execute_index);
__device__ bool cuda_try_push_primitive(ReadyQueueLayout* queue, OperatorLayout* primitive);
__device__ void cuda_execute_primitive(OperatorLayout* primitive, int execute_index);

}
