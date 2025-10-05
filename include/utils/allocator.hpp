#pragma once
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <runtime/api/repr.h>

namespace pccl {
namespace utils {

using runtime::ExecutorType;

#ifdef __CUDACC__
#define PCCL_ALLOCATOR_CUDA
#endif

class Allocator {
public:
  static bool alloc(void** addr, size_t size, ExecutorType executor_type) {
    switch (executor_type) {
      case ExecutorType::CPU:
        *addr = malloc(size);
        return *addr != nullptr;
      
      case ExecutorType::CUDA:
#ifdef PCCL_ALLOCATOR_CUDA
        return cudaMalloc(addr, size) == cudaSuccess;
#else
        return false;
#endif
      
      default:
        return false;
    }
  }
  
  static bool free(void* addr, ExecutorType executor_type) {
    if (!addr) return true;
    
    switch (executor_type) {
      case ExecutorType::CPU:
        ::free(addr);
        return true;
      
      case ExecutorType::CUDA:
#ifdef PCCL_ALLOCATOR_CUDA
        return cudaFree(addr) == cudaSuccess;
#else
        return false;
#endif
      
      default:
        return false;
    }
  }
  
  static bool memcpy(void* dst, const void* src, size_t size, 
                    ExecutorType dst_type, ExecutorType src_type) {
    if (dst_type == ExecutorType::CPU && src_type == ExecutorType::CPU) {
      std::memcpy(dst, src, size);
      return true;
    }
#ifdef PCCL_ALLOCATOR_CUDA
    else if (dst_type == ExecutorType::CUDA && src_type == ExecutorType::CPU) {
      return cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice) == cudaSuccess;
    }
    else if (dst_type == ExecutorType::CPU && src_type == ExecutorType::CUDA) {
      return cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost) == cudaSuccess;
    }
    else if (dst_type == ExecutorType::CUDA && src_type == ExecutorType::CUDA) {
      return cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice) == cudaSuccess;
    }
#endif
    return false;
  }
  
  static bool memset(void* addr, int value, size_t size, ExecutorType executor_type) {
    if (!addr) return false;
    
    switch (executor_type) {
      case ExecutorType::CPU:
        ::memset(addr, value, size);
        return true;
      
      case ExecutorType::CUDA:
#ifdef PCCL_ALLOCATOR_CUDA
        return cudaMemset(addr, value, size) == cudaSuccess;
#else
        return false;
#endif
      
      default:
        return false;
    }
  }
};

} // namespace utils
} // namespace pccl
