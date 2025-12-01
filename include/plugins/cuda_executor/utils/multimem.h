#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <plugins/cuda_executor/utils/vector_datatype.h>

namespace engine_c::cuda {

template <class>
constexpr bool dependentFalse = false;

template <typename IType>
__device__ __forceinline__ IType 
multimem_ld_reduce(IType *ptr) {
  IType val;
  if constexpr (std::is_same_v<IType, i32x1>) {
    asm("multimem.ld_reduce.relaxed.sys.global.add.s32 %0, [%1];" : "=r"(val.words[0]) : "l"(ptr) : "memory");
  } else if constexpr (std::is_same_v<IType, u32x1>) {
    asm("multimem.ld_reduce.relaxed.sys.global.add.u32 %0, [%1];" : "=r"(val.words[0]) : "l"(ptr) : "memory");
  } else if constexpr (std::is_same_v<IType, f32x1>) {
    asm("multimem.ld_reduce.relaxed.sys.global.add.f32 %0, [%1];" : "=r"(val.words[0]) : "l"(ptr) : "memory");
  } else if constexpr (std::is_same_v<IType, f32x2>) {
    asm("multimem.ld_reduce.relaxed.sys.global.add.v2.f32 {%0,%1}, [%2];"
        : "=r"(val.words[0]), "=r"(val.words[1])
        : "l"(ptr)
        : "memory");
  } else if constexpr (std::is_same_v<IType, f32x4>) {
    asm("multimem.ld_reduce.relaxed.sys.global.add.v4.f32 {%0,%1,%2,%3}, [%4];"
        : "=r"(val.words[0]), "=r"(val.words[1]), "=r"(val.words[2]), "=r"(val.words[3])
        : "l"(ptr)
        : "memory");
  } else if constexpr (std::is_same_v<IType, f16x2>) {
    asm("multimem.ld_reduce.relaxed.sys.global.add.f16x2 %0, [%1];" : "=r"(val.words[0]) : "l"(ptr) : "memory");
  } else if constexpr (std::is_same_v<IType, f16x4>) {
    asm("multimem.ld_reduce.relaxed.sys.global.add.v2.f16x2 {%0,%1}, [%2];"
        : "=r"(val.words[0]), "=r"(val.words[1])
        : "l"(ptr)
        : "memory");
  } else if constexpr (std::is_same_v<IType, f16x8>) {
    asm("multimem.ld_reduce.relaxed.sys.global.add.v4.f16x2 {%0,%1,%2,%3}, [%4];"
        : "=r"(val.words[0]), "=r"(val.words[1]), "=r"(val.words[2]), "=r"(val.words[3])
        : "l"(ptr)
        : "memory");
  } else if constexpr (std::is_same_v<IType, bf16x2>) {
      asm("multimem.ld_reduce.relaxed.sys.global.add.bf16x2 %0, [%1];" : "=r"(val.words[0]) : "l"(ptr) : "memory");
  } else if constexpr (std::is_same_v<IType, bf16x4>) {
    asm("multimem.ld_reduce.relaxed.sys.global.add.v2.bf16x2 {%0,%1}, [%2];"
        : "=r"(val.words[0]), "=r"(val.words[1])
        : "l"(ptr)
        : "memory");
  } else if constexpr (std::is_same_v<IType, bf16x8>) {
    asm("multimem.ld_reduce.relaxed.sys.global.add.v4.bf16x2 {%0,%1,%2,%3}, [%4];"
        : "=r"(val.words[0]), "=r"(val.words[1]), "=r"(val.words[2]), "=r"(val.words[3])
        : "l"(ptr)
        : "memory");
  } else {
    static_assert(dependentFalse<IType>, "Not supported type");
  }
}

template <typename IType, typename T>
__device__ __forceinline__ void multimemStore(const IType& val, T* ptr) {
  if constexpr (std::is_same_v<IType, i32x1>) {
    asm volatile("multimem.st.relaxed.sys.global.s32 [%0], %1;" ::"l"(ptr), "r"(val.words[0]) : "memory");
  } else if constexpr (std::is_same_v<IType, u32x1>) {
    asm volatile("multimem.st.relaxed.sys.global.u32 [%0], %1;" ::"l"(ptr), "r"(val.words[0]) : "memory");
  } else if constexpr (std::is_same_v<IType, f64x1>) {
    asm volatile("multimem.st.relaxed.sys.global.f64 [%0], %1;" ::"l"(ptr), "d"(val.words[0]) : "memory");
  } else if constexpr (std::is_same_v<IType, f32x1>) {
    asm volatile("multimem.st.relaxed.sys.global.f32 [%0], %1;" ::"l"(ptr), "r"(val.words[0]) : "memory");
  } else if constexpr (std::is_same_v<IType, f32x2>) {
    asm volatile("multimem.st.relaxed.sys.global.v2.f32 [%0], {%1,%2};" ::"l"(ptr), "r"(val.words[0]),
                  "r"(val.words[1])
                  : "memory");
  } else if constexpr (std::is_same_v<IType, f32x4>) {
    asm volatile("multimem.st.relaxed.sys.global.v4.f32 [%0], {%1,%2,%3,%4};" ::"l"(ptr), "r"(val.words[0]),
                  "r"(val.words[1]), "r"(val.words[2]), "r"(val.words[3])
                  : "memory");
  } else if constexpr (std::is_same_v<IType, f16x2>) {
    asm volatile("multimem.st.relaxed.sys.global.f16x2 [%0], %1;" ::"l"(ptr), "r"(val.words[0]) : "memory");
  } else if constexpr (std::is_same_v<IType, f16x4>) {
    asm volatile("multimem.st.relaxed.sys.global.v2.f16x2 [%0], {%1,%2};" ::"l"(ptr), "r"(val.words[0]),
                  "r"(val.words[1])
                  : "memory");
  } else if constexpr (std::is_same_v<IType, f16x8>) {
    asm volatile("multimem.st.relaxed.sys.global.v4.f16x2 [%0], {%1,%2,%3,%4};" ::"l"(ptr), "r"(val.words[0]),
                  "r"(val.words[1]), "r"(val.words[2]), "r"(val.words[3])
                  : "memory");
  } else if constexpr (std::is_same_v<IType, bf16x2>) {
    asm volatile("multimem.st.relaxed.sys.global.bf16x2 [%0], %1;" ::"l"(ptr), "r"(val.words[0]) : "memory");
  } else if constexpr (std::is_same_v<IType, bf16x4>) {
    asm volatile("multimem.st.relaxed.sys.global.v2.bf16x2 [%0], {%1,%2};" ::"l"(ptr), "r"(val.words[0]),
                  "r"(val.words[1])
                  : "memory");
  } else if constexpr (std::is_same_v<IType, bf16x8>) {
    asm volatile("multimem.st.relaxed.sys.global.v4.bf16x2 [%0], {%1,%2,%3,%4};" ::"l"(ptr), "r"(val.words[0]),
                  "r"(val.words[1]), "r"(val.words[2]), "r"(val.words[3])
                  : "memory");
  } else {
    static_assert(dependentFalse<IType>, "Not supported type");
  }
};

template <typename TValue, typename T>
__device__ __forceinline__ void multimemStoreReduce(const TValue& val, T* ptr) {
  if constexpr (std::is_same_v<TValue, float4> && std::is_same_v<T, float>) {
    asm volatile("multimem.red.relaxed.sys.global.add.v4.f32 [%0], {%1,%2,%3,%4};" ::"l"(ptr), "r"(val.x), "r"(val.y),
                  "r"(val.z), "r"(val.w)
                  : "memory");
  } else if constexpr (std::is_same_v<TValue, uint2> && std::is_same_v<T, float>) {
    asm volatile("multimem.red.relaxed.sys.global.add.v2.f32 [%0], {%1,%2};" ::"l"(ptr), "r"(val.x), "r"(val.y)
                  : "memory");
  } else if constexpr (std::is_same_v<TValue, uint1> && std::is_same_v<T, float>) {
    asm volatile("multimem.red.relaxed.sys.global.add.f32 [%0], {%1};" ::"l"(ptr), "r"(val.x) : "memory");
  } else if constexpr (std::is_same_v<TValue, uint4> && std::is_same_v<T, __half2>) {
    asm volatile("multimem.red.relaxed.sys.global.add.v4.f16x2 [%0], {%1,%2,%3,%4};" ::"l"(ptr), "r"(val.x),
                  "r"(val.y), "r"(val.z), "r"(val.w)
                  : "memory");
  } else if constexpr (std::is_same_v<TValue, uint2> && std::is_same_v<T, __half2>) {
    asm volatile("multimem.red.relaxed.sys.global.add.v2.f16x2 [%0], {%1,%2};" ::"l"(ptr), "r"(val.x), "r"(val.y)
                  : "memory");
  } else if constexpr (std::is_same_v<TValue, uint1> && std::is_same_v<T, __half2>) {
    asm volatile("multimem.red.relaxed.sys.global.add.f16x2 [%0], {%1};" ::"l"(ptr), "r"(val.x) : "memory");
  } else {
    static_assert(dependentFalse<T>, "Not supported type");
  }
};

}

