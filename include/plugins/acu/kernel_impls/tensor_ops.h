#pragma once
#include <plugins/acu/common.h>
#include <cute/tensor.hpp>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace pccl::acu {

template<typename T, int VecSize>
using VecType = cute::array<T, VecSize>;

template<typename T, int VecSize>
__device__ void tensor_add_impl(const T* src, T* dst, uint64_t data_size, int execute_index, int total_executors) {
  uint64_t num_elements = data_size / sizeof(T);
  uint64_t elements_per_executor = (num_elements + total_executors - 1) / total_executors;
  uint64_t start_idx = execute_index * elements_per_executor;
  uint64_t end_idx = min(start_idx + elements_per_executor, num_elements);
  uint64_t executor_elements = end_idx - start_idx;
  
  uint64_t vectorized_elements = executor_elements / VecSize;
  uint64_t remainder = executor_elements % VecSize;
  
  const VecType<T, VecSize>* src_vec = reinterpret_cast<const VecType<T, VecSize>*>(src + start_idx);
  VecType<T, VecSize>* dst_vec = reinterpret_cast<VecType<T, VecSize>*>(dst + start_idx);
  
  for (uint64_t i = threadIdx.x; i < vectorized_elements; i += blockDim.x) {
    VecType<T, VecSize> src_val = src_vec[i];
    VecType<T, VecSize> dst_val = dst_vec[i];
    
    #pragma unroll
    for (int j = 0; j < VecSize; ++j) {
      if constexpr (std::is_same_v<T, half>) {
        dst_val[j] = __hadd(dst_val[j], src_val[j]);
      } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        dst_val[j] = __hadd(dst_val[j], src_val[j]);
      } else {
        dst_val[j] += src_val[j];
      }
    }
    
    dst_vec[i] = dst_val;
  }
  
  if (remainder > 0 && threadIdx.x == 0) {
    uint64_t base_idx = start_idx + vectorized_elements * VecSize;
    for (uint64_t i = base_idx; i < end_idx; ++i) {
      if constexpr (std::is_same_v<T, half>) {
        dst[i] = __hadd(dst[i], src[i]);
      } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        dst[i] = __hadd(dst[i], src[i]);
      } else {
        dst[i] += src[i];
      }
    }
  }
}

template<typename T, int VecSize>
__device__ void tensor_multiply_impl(const T* src, T* dst, uint64_t data_size,
                                    int execute_index, int total_executors) {
  uint64_t num_elements = data_size / sizeof(T);
  uint64_t elements_per_executor = (num_elements + total_executors - 1) / total_executors;
  uint64_t start_idx = execute_index * elements_per_executor;
  uint64_t end_idx = min(start_idx + elements_per_executor, num_elements);
  uint64_t executor_elements = end_idx - start_idx;
  
  uint64_t vectorized_elements = executor_elements / VecSize;
  uint64_t remainder = executor_elements % VecSize;
  
  const VecType<T, VecSize>* src_vec = 
      reinterpret_cast<const VecType<T, VecSize>*>(src + start_idx);
  VecType<T, VecSize>* dst_vec = 
      reinterpret_cast<VecType<T, VecSize>*>(dst + start_idx);
  
  for (uint64_t i = threadIdx.x; i < vectorized_elements; i += blockDim.x) {
    VecType<T, VecSize> src_val = src_vec[i];
    VecType<T, VecSize> dst_val = dst_vec[i];
    
    #pragma unroll
    for (int j = 0; j < VecSize; ++j) {
      if constexpr (std::is_same_v<T, half>) {
        dst_val[j] = __hmul(dst_val[j], src_val[j]);
      } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        dst_val[j] = __hmul(dst_val[j], src_val[j]);
      } else {
        dst_val[j] *= src_val[j];
      }
    }
    
    dst_vec[i] = dst_val;
  }
  
  if (remainder > 0 && threadIdx.x == 0) {
    uint64_t base_idx = start_idx + vectorized_elements * VecSize;
    for (uint64_t i = base_idx; i < end_idx; ++i) {
      if constexpr (std::is_same_v<T, half>) {
        dst[i] = __hmul(dst[i], src[i]);
      } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        dst[i] = __hmul(dst[i], src[i]);
      } else {
        dst[i] *= src[i];
      }
    }
  }
}

template<typename T, int VecSize>
__device__ void tensor_max_impl(const T* src, T* dst, uint64_t data_size,
                               int execute_index, int total_executors) {
  uint64_t num_elements = data_size / sizeof(T);
  uint64_t elements_per_executor = (num_elements + total_executors - 1) / total_executors;
  uint64_t start_idx = execute_index * elements_per_executor;
  uint64_t end_idx = min(start_idx + elements_per_executor, num_elements);
  uint64_t executor_elements = end_idx - start_idx;
  
  uint64_t vectorized_elements = executor_elements / VecSize;
  uint64_t remainder = executor_elements % VecSize;
  
  const VecType<T, VecSize>* src_vec = 
      reinterpret_cast<const VecType<T, VecSize>*>(src + start_idx);
  VecType<T, VecSize>* dst_vec = 
      reinterpret_cast<VecType<T, VecSize>*>(dst + start_idx);
  
  for (uint64_t i = threadIdx.x; i < vectorized_elements; i += blockDim.x) {
    VecType<T, VecSize> src_val = src_vec[i];
    VecType<T, VecSize> dst_val = dst_vec[i];
    
    #pragma unroll
    for (int j = 0; j < VecSize; ++j) {
      if constexpr (std::is_same_v<T, half>) {
        dst_val[j] = __hgt(src_val[j], dst_val[j]) ? src_val[j] : dst_val[j];
      } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        dst_val[j] = __hgt(src_val[j], dst_val[j]) ? src_val[j] : dst_val[j];
      } else {
        dst_val[j] = max(dst_val[j], src_val[j]);
      }
    }
    
    dst_vec[i] = dst_val;
  }
  
  if (remainder > 0 && threadIdx.x == 0) {
    uint64_t base_idx = start_idx + vectorized_elements * VecSize;
    for (uint64_t i = base_idx; i < end_idx; ++i) {
      if constexpr (std::is_same_v<T, half>) {
        dst[i] = __hgt(src[i], dst[i]) ? src[i] : dst[i];
      } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        dst[i] = __hgt(src[i], dst[i]) ? src[i] : dst[i];
      } else {
        dst[i] = max(dst[i], src[i]);
      }
    }
  }
}

template<typename T, int VecSize>
__device__ void tensor_min_impl(const T* src, T* dst, uint64_t data_size,
                               int execute_index, int total_executors) {
  uint64_t num_elements = data_size / sizeof(T);
  uint64_t elements_per_executor = (num_elements + total_executors - 1) / total_executors;
  uint64_t start_idx = execute_index * elements_per_executor;
  uint64_t end_idx = min(start_idx + elements_per_executor, num_elements);
  uint64_t executor_elements = end_idx - start_idx;
  
  uint64_t vectorized_elements = executor_elements / VecSize;
  uint64_t remainder = executor_elements % VecSize;
  
  const VecType<T, VecSize>* src_vec = 
      reinterpret_cast<const VecType<T, VecSize>*>(src + start_idx);
  VecType<T, VecSize>* dst_vec = 
      reinterpret_cast<VecType<T, VecSize>*>(dst + start_idx);
  
  for (uint64_t i = threadIdx.x; i < vectorized_elements; i += blockDim.x) {
    VecType<T, VecSize> src_val = src_vec[i];
    VecType<T, VecSize> dst_val = dst_vec[i];
    
    #pragma unroll
    for (int j = 0; j < VecSize; ++j) {
      if constexpr (std::is_same_v<T, half>) {
        dst_val[j] = __hlt(src_val[j], dst_val[j]) ? src_val[j] : dst_val[j];
      } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        dst_val[j] = __hlt(src_val[j], dst_val[j]) ? src_val[j] : dst_val[j];
      } else {
        dst_val[j] = min(dst_val[j], src_val[j]);
      }
    }
    
    dst_vec[i] = dst_val;
  }
  
  if (remainder > 0 && threadIdx.x == 0) {
    uint64_t base_idx = start_idx + vectorized_elements * VecSize;
    for (uint64_t i = base_idx; i < end_idx; ++i) {
      if constexpr (std::is_same_v<T, half>) {
        dst[i] = __hlt(src[i], dst[i]) ? src[i] : dst[i];
      } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        dst[i] = __hlt(src[i], dst[i]) ? src[i] : dst[i];
      } else {
        dst[i] = min(dst[i], src[i]);
      }
    }
  }
}

template<typename T, int VecSize>
__device__ void direct_copy_impl(const T* src, T* dst, uint64_t data_size,
                                int execute_index, int total_executors) {
  uint64_t num_elements = data_size / sizeof(T);
  uint64_t elements_per_executor = (num_elements + total_executors - 1) / total_executors;
  uint64_t start_idx = execute_index * elements_per_executor;
  uint64_t end_idx = min(start_idx + elements_per_executor, num_elements);
  uint64_t executor_elements = end_idx - start_idx;
  
  uint64_t vectorized_elements = executor_elements / VecSize;
  uint64_t remainder = executor_elements % VecSize;
  
  const VecType<T, VecSize>* src_vec = 
      reinterpret_cast<const VecType<T, VecSize>*>(src + start_idx);
  VecType<T, VecSize>* dst_vec = 
      reinterpret_cast<VecType<T, VecSize>*>(dst + start_idx);
  
  for (uint64_t i = threadIdx.x; i < vectorized_elements; i += blockDim.x) {
    dst_vec[i] = src_vec[i];
  }
  
  if (remainder > 0 && threadIdx.x == 0) {
    uint64_t base_idx = start_idx + vectorized_elements * VecSize;
    for (uint64_t i = base_idx; i < end_idx; ++i) {
      dst[i] = src[i];
    }
  }
}

template<typename T, int VecSize>
__device__ void direct_write_impl(T* dst, uint64_t data_size,
                                 int execute_index, int total_executors) {
  uint64_t num_elements = data_size / sizeof(T);
  uint64_t elements_per_executor = (num_elements + total_executors - 1) / total_executors;
  uint64_t start_idx = execute_index * elements_per_executor;
  uint64_t end_idx = min(start_idx + elements_per_executor, num_elements);
  uint64_t executor_elements = end_idx - start_idx;
  
  uint64_t vectorized_elements = executor_elements / VecSize;
  uint64_t remainder = executor_elements % VecSize;
  
  VecType<T, VecSize>* dst_vec = 
      reinterpret_cast<VecType<T, VecSize>*>(dst + start_idx);
  VecType<T, VecSize> zero_vec;
  
  #pragma unroll
  for (int i = 0; i < VecSize; ++i) {
    if constexpr (std::is_same_v<T, half>) {
      zero_vec[i] = __float2half(0.0f);
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
      zero_vec[i] = __float2bfloat16(0.0f);
    } else {
      zero_vec[i] = T(0);
    }
  }
  
  for (uint64_t i = threadIdx.x; i < vectorized_elements; i += blockDim.x) {
    dst_vec[i] = zero_vec;
  }
  
  if (remainder > 0 && threadIdx.x == 0) {
    uint64_t base_idx = start_idx + vectorized_elements * VecSize;
    for (uint64_t i = base_idx; i < end_idx; ++i) {
      if constexpr (std::is_same_v<T, half>) {
        dst[i] = __float2half(0.0f);
      } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        dst[i] = __float2bfloat16(0.0f);
      } else {
        dst[i] = T(0);
      }
    }
  }
}

} // namespace pccl::acu
