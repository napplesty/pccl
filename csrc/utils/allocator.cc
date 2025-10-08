#include "runtime/api/repr.h"
#include "utils/exception.hpp"
#include "utils/hex_utils.hpp"
#include <cstdlib>
#include <utils/allocator.h>
#include <cuda_runtime.h>

namespace pccl::utils {

void *allocate(runtime::ExecutorType executor_type, size_t size) {
  if (executor_type == runtime::ExecutorType::CPU) {
    void *ptr;
    cudaMallocHost(&ptr, size);
    return ptr;
  } else if (executor_type == runtime::ExecutorType::CUDA) {
    void *ptr;
    cudaMalloc(&ptr, size);
    return ptr;
  } else {
    PCCL_UNREACHABLE();
  }
}

std::string get_shareable_handle(runtime::ExecutorType executor_type, void *addr) {
  if (executor_type == runtime::ExecutorType::CUDA) {
    cudaIpcMemHandle_t handle;
    cudaIpcGetMemHandle(&handle, addr);
    return marshal_to_hex_str((void *)&handle, sizeof(handle));
  } else {
    PCCL_UNREACHABLE();
  }
}

void *from_shareable(runtime::ExecutorType executor_type, const std::string &shareable_handle) {
  if (executor_type == runtime::ExecutorType::CUDA) {
    cudaIpcMemHandle_t handle;
    unmarshal_from_hex_str((void *)&handle, shareable_handle);
    void* ptr = nullptr;
    cudaIpcOpenMemHandle(&ptr, handle, 
                        cudaIpcMemLazyEnablePeerAccess);
    return ptr;
  } else {
    PCCL_UNREACHABLE();
  }
}

void close_shareable_handle(runtime::ExecutorType executor_type, void* ptr) {
  if (executor_type == runtime::ExecutorType::CUDA) {
    if (ptr) {
      cudaIpcCloseMemHandle(ptr);
    }
  } else {
    PCCL_UNREACHABLE();
  }
}

bool generic_memcpy(runtime::ExecutorType src_type, runtime::ExecutorType dst_type, void* src, void* dst, size_t nbytes) {
  if (src_type == runtime::ExecutorType::CPU && dst_type == runtime::ExecutorType::CPU) {
    memcpy(dst, src, nbytes);
    return true;
  }
  else if (src_type == runtime::ExecutorType::CPU && dst_type == runtime::ExecutorType::CUDA) {
    cudaError_t result = cudaMemcpy(dst, src, nbytes, cudaMemcpyHostToDevice);
    return result == cudaSuccess;
  }
  else if (src_type == runtime::ExecutorType::CUDA && dst_type == runtime::ExecutorType::CPU) {
    cudaError_t result = cudaMemcpy(dst, src, nbytes, cudaMemcpyDeviceToHost);
    return result == cudaSuccess;
  }
  else if (src_type == runtime::ExecutorType::CUDA && dst_type == runtime::ExecutorType::CUDA) {
    cudaError_t result = cudaMemcpy(dst, src, nbytes, cudaMemcpyDeviceToDevice);
    return result == cudaSuccess;
  }
  else {
    return false;
  }
}

}
