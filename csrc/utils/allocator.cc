#include "runtime/api/repr.h"
#include "utils/exception.hpp"
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
    std::string str_handle(sizeof(handle)+1, '\0');
    memcpy(&str_handle[0], &handle, sizeof(handle));
    return str_handle;
  } else {
    PCCL_UNREACHABLE();
  }
}

void *from_shareable(runtime::ExecutorType executor_type, const std::string &shareable_handle) {
  if (executor_type == runtime::ExecutorType::CUDA) {
    cudaIpcMemHandle_t handle;
    memcpy(&handle, shareable_handle.data(), sizeof(handle));
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

}
