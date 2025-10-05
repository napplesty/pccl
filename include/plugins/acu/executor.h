#pragma once
#include <runtime/engine/graph_executor.h>
#include <cuda_runtime.h>

namespace pccl::engine::cuda {

class CUDAExecutorManager {
public:
  CUDAExecutorManager(GraphBufferLayout* graph_layout, int num_sms);
  ~CUDAExecutorManager();
  bool initialize();
  void start(cudaStream_t stream = 0);
  void stop();
  void wait();
  void launch_cuda_kernel(GraphBufferLayout* graph_layout, int num_blocks, int threads_per_block, cudaStream_t stream);
private:
  GraphBufferLayout* graph_layout_;
  int num_sms_;
  bool initialized_{false};
  cudaStream_t current_stream_{0};
};

__global__ void cuda_executor_kernel(GraphBufferLayout* graph_layout);
__device__ bool cuda_try_pop_primitive(ReadyQueueLayout* queue, OperatorLayout** primitive, int* execute_index);
__device__ bool cuda_try_push_primitive(ReadyQueueLayout* queue, OperatorLayout* primitive);
__device__ void cuda_execute_primitive(OperatorLayout* primitive, int execute_index);

}
