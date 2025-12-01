#pragma once

#include <cuda_fp4.h>
#include <cuda_fp6.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

namespace engine_c::cuda {

using __fp8_e4m3 = __nv_fp8_e4m3;
using __fp8_e5m2 = __nv_fp8_e5m2;
using __fp8x2_e4m3 = __nv_fp8x2_e4m3;
using __fp8x2_e5m2 = __nv_fp8x2_e5m2;
using __fp8x4_e4m3 = __nv_fp8x4_e4m3;
using __fp8x4_e5m2 = __nv_fp8x4_e5m2;
using __bfloat16 = __nv_bfloat16;
using __bfloat162 = __nv_bfloat162;

template <int Bytes>
struct alignas(Bytes) Words {
  static_assert(Bytes > 0, "Bytes must be greater than 0");
  static_assert(Bytes % 4 == 0, "Bytes must be multiple of 4");
  unsigned int w[Bytes / 4];

  __device__ __forceinline__ Words() {}

  __device__ __forceinline__ unsigned int& operator[](int i) { return w[i]; }

  __device__ __forceinline__ const unsigned int& operator[](int i) const { return w[i]; }
};

template <typename T, int N>
union alignas(sizeof(T) * N) VectorType {
  static_assert(N > 0, "N must be greater than 0");

  T data[N];
  Words<sizeof(T) * N> words;

  using ElementType = T;
  constexpr static int Size = N;

  __device__ __forceinline__ VectorType() {}

  __device__ __forceinline__ operator T*() { return data; }

  __device__ __forceinline__ operator const T*() const { return data; }

  __device__ __forceinline__ T& operator[](int i) { return data[i]; }

  __device__ __forceinline__ const T& operator[](int i) const { return data[i]; }
};

using i32x1 = VectorType<int, 1>;
using u32x1 = VectorType<unsigned int, 1>;
using f64x1 = VectorType<double, 1>;
using f32x1 = VectorType<float, 1>;

using i32x2 = VectorType<int, 2>;
using u32x2 = VectorType<unsigned int, 2>;
using f32x2 = VectorType<float, 2>;
using f16x2 = VectorType<__half, 2>;
using bf16x2 = VectorType<__bfloat16, 2>;

using i32x4 = VectorType<int, 4>;
using u32x4 = VectorType<unsigned int, 4>;
using f32x4 = VectorType<float, 4>;
using f16x4 = VectorType<__half, 4>;
using bf16x4 = VectorType<__bfloat16, 4>;

using f16x8 = VectorType<__half, 8>;
using bf16x8 = VectorType<__bfloat16, 8>;

using fp8_e4m3x2 = VectorType<__fp8_e4m3, 2>;
using fp8_e4m3x4 = VectorType<__fp8_e4m3, 4>;
using fp8_e4m3x8 = VectorType<__fp8_e4m3, 8>;
using fp8_e5m2x2 = VectorType<__fp8_e5m2, 2>;
using fp8_e5m2x4 = VectorType<__fp8_e5m2, 4>;
using fp8_e5m2x8 = VectorType<__fp8_e5m2, 8>;

}

