#pragma once

#include "config.h"
#include "cuda/sync.h"
#include "device.h"

namespace pccl {

template <typename To, typename From>
PCCL_CUDA_DEVICE_INLINE To bit_cast(const From& src) {
  static_assert(sizeof(To) == sizeof(From), "Size mismatch for bit_cast");
  union {
    From f;
    To t;
  } u;
  u.f = src;
  return u.t;
}

template <typename T>
PCCL_CUDA_DEVICE_INLINE T add_elements(T a, T b) {
  return a + b;
}
template <>
PCCL_CUDA_DEVICE_INLINE __float16 add_elements(__float16 a, __float16 b) {
  return __hadd(a, b);
}

template <>
PCCL_CUDA_DEVICE_INLINE __float162 add_elements(__float162 a, __float162 b) {
  return __hadd2(a, b);
}

template <>
PCCL_CUDA_DEVICE_INLINE __bfloat16 add_elements(__bfloat16 a, __bfloat16 b) {
  return __hadd(a, b);
}

template <>
PCCL_CUDA_DEVICE_INLINE __bfloat162 add_elements(__bfloat162 a, __bfloat162 b) {
  return __hadd2(a, b);
}

template <typename T>
PCCL_CUDA_DEVICE_INLINE int4 add_vectors_helper(int4 a, int4 b) {
  int4 ret;
  ret.w = bit_cast<int, T>(
      add_elements(bit_cast<T, int>(a.w), bit_cast<T, int>(b.w)));
  ret.x = bit_cast<int, T>(
      add_elements(bit_cast<T, int>(a.x), bit_cast<T, int>(b.x)));
  ret.y = bit_cast<int, T>(
      add_elements(bit_cast<T, int>(a.y), bit_cast<T, int>(b.y)));
  ret.z = bit_cast<int, T>(
      add_elements(bit_cast<T, int>(a.z), bit_cast<T, int>(b.z)));
  return ret;
}

template <typename T>
PCCL_CUDA_DEVICE_INLINE int4 add_vectors(int4 a, int4 b) {
  return add_vectors_helper<T>(a, b);
}

template <>
PCCL_CUDA_DEVICE_INLINE int4 add_vectors<__half>(int4 a, int4 b) {
  return add_vectors_helper<__half2>(a, b);
}

template <>
PCCL_CUDA_DEVICE_INLINE int4 add_vectors<__bfloat16>(int4 a, int4 b) {
  return add_vectors_helper<__bfloat162>(a, b);
}

template <typename T>
PCCL_CUDA_DEVICE_INLINE uint2 add_vectors_helper(uint2 a, uint2 b) {
  uint2 ret;
  ret.x = bit_cast<int, T>(
      add_elements(bit_cast<T, int>(a.x), bit_cast<T, int>(b.x)));
  ret.y = bit_cast<int, T>(
      add_elements(bit_cast<T, int>(a.y), bit_cast<T, int>(b.y)));
  return ret;
}

template <typename T>
PCCL_CUDA_DEVICE_INLINE uint2 add_vectors(uint2 a, uint2 b) {
  return add_vectors_helper<T>(a, b);
}

template <>
PCCL_CUDA_DEVICE_INLINE __attribute__((unused)) uint2
add_vectors<__half>(uint2 a, uint2 b) {
  return add_vectors_helper<__half2>(a, b);
}

template <>
PCCL_CUDA_DEVICE_INLINE __attribute__((unused)) uint2
add_vectors<__bfloat16>(uint2 a, uint2 b) {
  return add_vectors_helper<__bfloat162>(a, b);
}

template <typename T>
PCCL_CUDA_DEVICE_INLINE int add_vectors_helper(int a, int b) {
  return bit_cast<int, T>(
      add_elements(bit_cast<T, int>(a), bit_cast<T, int>(b)));
}

template <typename T>
PCCL_CUDA_DEVICE_INLINE int add_vectors(int a, int b) {
  return add_vectors_helper<T>(a, b);
}

template <>
PCCL_CUDA_DEVICE_INLINE __attribute__((unused)) int add_vectors<__half>(int a,
                                                                        int b) {
  return add_vectors_helper<__half2>(a, b);
}

template <>
PCCL_CUDA_DEVICE_INLINE __attribute__((unused)) int add_vectors<__bfloat16>(
    int a, int b) {
  return add_vectors_helper<__bfloat162>(a, b);
}

template <typename T>
PCCL_CUDA_DEVICE_INLINE uint32_t add_vectors_helper(uint32_t a, uint32_t b) {
  return bit_cast<uint32_t, T>(
      add_elements(bit_cast<T, uint32_t>(a), bit_cast<T, uint32_t>(b)));
}

template <typename T>
PCCL_CUDA_DEVICE_INLINE uint32_t add_vectors(uint32_t a, uint32_t b) {
  return add_vectors_helper<T>(a, b);
}

template <>
PCCL_CUDA_DEVICE_INLINE uint32_t add_vectors<__half>(uint32_t a, uint32_t b) {
  return add_vectors_helper<__half2>(a, b);
}

template <>
PCCL_CUDA_DEVICE_INLINE uint32_t add_vectors<__bfloat16>(uint32_t a,
                                                         uint32_t b) {
  return add_vectors_helper<__bfloat162>(a, b);
}

template <typename T>
struct VectorType {
  using type = T;
  using nvls_type = T;
  using nvls_type2 = T;
};

template <>
struct VectorType<__half> {
  using type = __half2;
  using nvls_type = uint4;
  using nvls_type2 = uint1;
};

template <>
struct VectorType<__bfloat16> {
  using type = __bfloat162;
  using nvls_type = uint4;
  using nvls_type2 = uint1;
};

template <>
struct VectorType<float> {
  using type = float;
  using nvls_type = uint4;
  using nvls_type2 = uint1;
};

}  // namespace pccl
