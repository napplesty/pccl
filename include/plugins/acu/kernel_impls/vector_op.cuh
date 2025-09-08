#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

namespace pccl {

template<typename T>
struct VectorizationTraits {
  static constexpr int VEC_SIZE = 1;
  using VecType = T;
};

template<>
struct VectorizationTraits<float> {
  static constexpr int VEC_SIZE = 4;
  using VecType = float4;
};

template<>
struct VectorizationTraits<half> {
  static constexpr int VEC_SIZE = 2;
  using VecType = half2;
};

template<>
struct VectorizationTraits<__nv_bfloat16> {
  static constexpr int VEC_SIZE = 2;
  using VecType = __nv_bfloat162;
};

template<>
struct VectorizationTraits<__nv_fp8_e4m3> {
  static constexpr int VEC_SIZE = 4;
  using VecType = __nv_fp8x4_e4m3;
};


template<>
struct VectorizationTraits<__nv_fp8_e5m2> {
  static constexpr int VEC_SIZE = 4;
  using VecType = __nv_fp8x4_e5m2;
};

struct AddOp {
  template<typename T>
  static __forceinline__ __device__ T compute(T a, T b) { return a + b; }
  static __forceinline__ __device__ half compute(half a, half b) { return __hadd(a, b); }
  static __forceinline__ __device__ __nv_bfloat16 compute(__nv_bfloat16 a, __nv_bfloat16 b) { 
    return __hadd(a, b); 
  }
  static __forceinline__ __device__ __nv_fp8_e4m3 compute(__nv_fp8_e4m3 a, __nv_fp8_e4m3 b) {
    half a_val = __nv_cvt_fp8_to_halfraw(a.__x, __NV_E4M3);
    half b_val = __nv_cvt_fp8_to_halfraw(b.__x, __NV_E4M3);
    return __nv_fp8_e4m3(__nv_cvt_halfraw_to_fp8(__hadd(a_val, b_val), __NV_SATFINITE, __NV_E4M3));
  }
  static __forceinline__ __device__ __nv_fp8_e5m2 compute(__nv_fp8_e5m2 a, __nv_fp8_e5m2 b) {
    half a_val = __nv_cvt_fp8_to_halfraw(a.__x, __NV_E5M2);
    half b_val = __nv_cvt_fp8_to_halfraw(b.__x, __NV_E5M2);
    return __nv_fp8_e5m2(__nv_cvt_halfraw_to_fp8(__hadd(a_val, b_val), __NV_SATFINITE, __NV_E5M2));
  }
};

struct SubOp {
  template<typename T>
  static __forceinline__ __device__ T compute(T a, T b) { return a - b; }
  static __forceinline__ __device__ half compute(half a, half b) { return __hsub(a, b); }
  static __forceinline__ __device__ __nv_bfloat16 compute(__nv_bfloat16 a, __nv_bfloat16 b) { 
    return __hsub(a, b); 
  }
  static __forceinline__ __device__ __nv_fp8_e4m3 compute(__nv_fp8_e4m3 a, __nv_fp8_e4m3 b) {
    half a_val = __nv_cvt_fp8_to_halfraw(a.__x, __NV_E4M3);
    half b_val = __nv_cvt_fp8_to_halfraw(b.__x, __NV_E4M3);
    return __nv_fp8_e4m3(__nv_cvt_halfraw_to_fp8(__hsub(a_val, b_val), __NV_SATFINITE, __NV_E4M3));
  }
  static __forceinline__ __device__ __nv_fp8_e5m2 compute(__nv_fp8_e5m2 a, __nv_fp8_e5m2 b) {
    half a_val = __nv_cvt_fp8_to_halfraw(a.__x, __NV_E5M2);
    half b_val = __nv_cvt_fp8_to_halfraw(b.__x, __NV_E5M2);
    return __nv_fp8_e5m2(__nv_cvt_halfraw_to_fp8(__hsub(a_val, b_val), __NV_SATFINITE, __NV_E5M2));
  }
};

struct MulOp {
  template<typename T>
  static __forceinline__ __device__ T compute(T a, T b) { return a * b; }
  static __forceinline__ __device__ half compute(half a, half b) { return __hmul(a, b); }
  static __forceinline__ __device__ __nv_bfloat16 compute(__nv_bfloat16 a, __nv_bfloat16 b) { 
    return __hmul(a, b); 
  }
  static __forceinline__ __device__ __nv_fp8_e4m3 compute(__nv_fp8_e4m3 a, __nv_fp8_e4m3 b) {
    half a_val = __nv_cvt_fp8_to_halfraw(a.__x, __NV_E4M3);
    half b_val = __nv_cvt_fp8_to_halfraw(b.__x, __NV_E4M3);
    return __nv_fp8_e4m3(__nv_cvt_halfraw_to_fp8(__hmul(a_val, b_val), __NV_SATFINITE, __NV_E4M3));
  }
  static __forceinline__ __device__ __nv_fp8_e5m2 compute(__nv_fp8_e5m2 a, __nv_fp8_e5m2 b) {
    half a_val = __nv_cvt_fp8_to_halfraw(a.__x, __NV_E5M2);
    half b_val = __nv_cvt_fp8_to_halfraw(b.__x, __NV_E5M2);
    return __nv_fp8_e5m2(__nv_cvt_halfraw_to_fp8(__hmul(a_val, b_val), __NV_SATFINITE, __NV_E5M2));
  }
};

struct DivOp {
  template<typename T>
  static __forceinline__ __device__ T compute(T a, T b) { return a / b; }
  static __forceinline__ __device__ half compute(half a, half b) { return __hdiv(a, b); }
  static __forceinline__ __device__ __nv_bfloat16 compute(__nv_bfloat16 a, __nv_bfloat16 b) { 
    return __hdiv(a, b);
  }
  static __forceinline__ __device__ __nv_fp8_e4m3 compute(__nv_fp8_e4m3 a, __nv_fp8_e4m3 b) {
    half a_val = __nv_cvt_fp8_to_halfraw(a.__x, __NV_E4M3);
    half b_val = __nv_cvt_fp8_to_halfraw(b.__x, __NV_E4M3);
    return __nv_fp8_e4m3(__nv_cvt_halfraw_to_fp8(__hdiv(a_val, b_val), __NV_SATFINITE, __NV_E4M3));
  }
  static __forceinline__ __device__ __nv_fp8_e5m2 compute(__nv_fp8_e5m2 a, __nv_fp8_e5m2 b) {
    half a_val = __nv_cvt_fp8_to_halfraw(a.__x, __NV_E5M2);
    half b_val = __nv_cvt_fp8_to_halfraw(b.__x, __NV_E5M2);
    return __nv_fp8_e5m2(__nv_cvt_halfraw_to_fp8(__hdiv(a_val, b_val), __NV_SATFINITE, __NV_E5M2));
  }
};

template<typename T, typename Op>
struct VectorOpImpl {
  static constexpr int VEC_SIZE = VectorizationTraits<T>::VEC_SIZE;
  using VecType = typename VectorizationTraits<T>::VecType;
  
  __device__ static void apply(T* out, const T* a, const T* b, int offset, int nelem) {
    int tid = threadIdx.x;
    int total_threads = blockDim.x * blockDim.y * blockDim.z;
    
    if constexpr (VEC_SIZE > 1) {
      int vec_count = nelem / VEC_SIZE;
      int threads_needed = vec_count > total_threads ? total_threads : vec_count;
      
      if (tid < threads_needed) {
        int vec_per_thread = vec_count / threads_needed;
        
        for (int i = 0; i < vec_per_thread; i++) {
          int vec_idx = tid + i * threads_needed;
          if (vec_idx < vec_count) {
            int global_idx = offset + vec_idx * VEC_SIZE;
            VecType a_vec = *reinterpret_cast<const VecType*>(a + global_idx);
            VecType b_vec = *reinterpret_cast<const VecType*>(b + global_idx);
            VecType r_vec = compute_vec(a_vec, b_vec);
            *reinterpret_cast<VecType*>(out + global_idx) = r_vec;
          }
        }
      }
    } else {
      for (int i = tid; i < nelem; i += total_threads) {
        int global_idx = offset + i;
        out[global_idx] = Op::compute(a[global_idx], b[global_idx]);
      }
    }
  }
  
private:
  __device__ static VecType compute_vec(const VecType& a, const VecType& b) {
    if constexpr (std::is_same_v<VecType, float4>) {
      return {Op::compute(a.x, b.x), Op::compute(a.y, b.y), 
              Op::compute(a.z, b.z), Op::compute(a.w, b.w)};
    } else if constexpr (std::is_same_v<VecType, __nv_bfloat162>) {
      return {Op::compute(a.x, b.x), Op::compute(a.y, b.y)};
    } else if constexpr (std::is_same_v<VecType, half2>) {
      return {Op::compute(a.x, b.x), Op::compute(a.y, b.y)};
    } else if constexpr (std::is_same_v<VecType, __nv_fp8x4_e4m3>) {
      return {Op::compute(a.x, b.x), Op::compute(a.y, b.y), 
              Op::compute(a.z, b.z), Op::compute(a.w, b.w)};
    } else if constexpr (std::is_same_v<VecType, __nv_fp8x4_e5m2>) {
      return {Op::compute(a.x, b.x), Op::compute(a.y, b.y), 
              Op::compute(a.z, b.z), Op::compute(a.w, b.w)};
    } else {
      return Op::compute(a, b);
    }
  }
};

} // namespace pccl

