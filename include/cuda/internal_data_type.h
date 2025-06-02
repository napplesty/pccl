#pragma once

#include <cstdint>

#include "device.h"

namespace pccl {

// Unified Message Defination for host and device

using SemaphoreId = uint32_t;
using MemoryId = uint32_t;
using TriggerType = uint64_t;
constexpr TriggerType TriggerData = 0b1;
constexpr TriggerType TriggerSignal = 0b10;
constexpr TriggerType TriggerSync = 0b100;
constexpr TriggerType TriggerLib = 1ull;
constexpr TriggerType TriggerHost = 2ull;
constexpr TriggerType TriggerDevice = 3ull;
constexpr TriggerType TriggerIssuedMask = 1ull << 62;
constexpr TriggerType TriggerCompletedMask = 1ull << 63;

struct alignas(16) ProxyTriggerLayout {
  union {
    uint64_t fst;
    uint64_t snd;
    struct {
      uint64_t size : 32;
      uint64_t srcOffset : 32;
      uint64_t dstOffset : 32;
      uint64_t src_memory_id : 2;
      uint64_t dst_memory_id : 2;
      uint64_t type : 3;
      uint64_t chanId : 23; // maximum to 2^23
      uint64_t issued : 1;
      uint64_t completed : 1;
    };
  };

  PCCL_CUDA_HOST_DEVICE_INLINE bool checkCompleted() {
    return atomicLoad(&snd, memoryOrderRelaxed) & TriggerCompletedMask;
  }
  PCCL_CUDA_HOST_DEVICE_INLINE bool checkIssued() {
    return atomicLoad(&snd, memoryOrderRelaxed) & TriggerIssuedMask;
  }
  PCCL_CUDA_HOST_DEVICE_INLINE void setCompleted() {
    uint64_t val = atomicLoad(&snd, memoryOrderAcquire);
    val |= TriggerCompletedMask;
    atomicStore(&snd, val, memoryOrderRelease);
  }
  PCCL_CUDA_HOST_DEVICE_INLINE void setIssued() {
    uint64_t val = atomicLoad(&snd, memoryOrderAcquire);
    val |= TriggerIssuedMask;
    atomicStore(&snd, val, memoryOrderRelease);
  }
  PCCL_CUDA_HOST_DEVICE_INLINE void clean() {
    atomicStore(&snd, 0ul, memoryOrderRelease);
  }
};

struct alignas(16) InterSmMessageLayout {
  union {
    uint64_t data[8];
    struct {
      uint64_t occupy[7];
      uint64_t : 62;
      uint64_t issued : 1;
      uint64_t completed : 1;
    };
  };

  PCCL_CUDA_HOST_DEVICE_INLINE bool checkCompleted() {
    return atomicLoad(&data[7], memoryOrderRelaxed) & TriggerCompletedMask;
  }
  PCCL_CUDA_HOST_DEVICE_INLINE bool checkIssued() {
    return atomicLoad(&data[7], memoryOrderRelaxed) & TriggerIssuedMask;
  }
  PCCL_CUDA_HOST_DEVICE_INLINE void setCompleted() {
    uint64_t val = atomicLoad(&data[7], memoryOrderAcquire);
    val |= TriggerCompletedMask;
    atomicStore(&data[7], val, memoryOrderRelease);
  }
  PCCL_CUDA_HOST_DEVICE_INLINE void setIssued() {
    uint64_t val = atomicLoad(&data[7], memoryOrderAcquire);
    val |= TriggerIssuedMask;
    atomicStore(&data[7], val, memoryOrderRelease);
  }
  PCCL_CUDA_HOST_DEVICE_INLINE void clean() {
    atomicStore(&data[7], 0ul, memoryOrderRelease);
  }
};

} // namespace pccl