#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace pccl::acu {

enum class ComputeType {
  SUM,
  PROD,
  MAX,
  MIN,
};

enum class DataType {
  F32,
  F16,
  BF16,
};

template<DataType DT>
struct TypeTraits;

template<>
struct TypeTraits<DataType::F32> {
  using Type = float;
};

template<>
struct TypeTraits<DataType::F16> {
  using Type = half;
};

template<>
struct TypeTraits<DataType::BF16> {
  using Type = __nv_bfloat16;
};

}
