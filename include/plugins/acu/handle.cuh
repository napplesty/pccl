#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <mutex>

namespace pccl {

using PcclKernel_t = void(*)(char *);

template <typename... BlockOperations>
class CudaCompoundKernel {
public:
  constexpr static size_t block_num = sizeof...(BlockOperations);
  __constant__ static PcclKernel_t kernels[block_num];

public:
  CudaCompoundKernel() {
    PcclKernel_t host_kernels[] = {BlockOperations::execute...};
    cudaMemcpyToSymbol(kernels, host_kernels, block_num);
  }

  __global__ static void execute(char **params) {
    const size_t block_idx = blockIdx.x;
    if (block_idx < block_num) {
      kernels[block_idx](params[block_idx]);
    }
  }
};

template <typename... BlockOperations>
void cudaCompoundKernelCall(char **params, cudaStream_t stream) {
  static CudaCompoundKernel<BlockOperations...> compound;
  cudaLaunchConfig_t config = {0};
  config.gridDim = dim3(compound.block_num, 1, 1);
  config.blockDim = dim3(288, 1, 1);
  config.stream = stream;
  void *args[] = {&params};
  CUDA_CHECK(cudaLaunchKernelEx(&config, compound.execute, args));
}

}
