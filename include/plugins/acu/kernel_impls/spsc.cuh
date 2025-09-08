#pragma once
#include <cstdint>
#include "plugins/acu/utils.cuh"
#include "plugins/acu/kernel_impls/atomic.cuh"

namespace pccl {

using TriggerType = uint64_t;
constexpr TriggerType TriggerData = 0x1;  // Trigger a data transfer.
constexpr TriggerType TriggerFlag = 0x2;  // Trigger a signaling.
constexpr TriggerType TriggerSync = 0x4;  // Trigger a flush.

constexpr unsigned int TriggerBitsSize = 32;
constexpr unsigned int TriggerBitsOffset = 32;
constexpr unsigned int TriggerBitsMemoryId = 18;
constexpr unsigned int TriggerBitsType = 3;
constexpr unsigned int TriggerBitsSemaphoreId = 18;
constexpr unsigned int TriggerBitsFifoReserved = 1;

static_assert(TriggerBitsSize + TriggerBitsOffset <= 64, "First 64-bit field overflow");
static_assert(TriggerBitsMemoryId * 2 + TriggerBitsType + 
              TriggerBitsSemaphoreId + TriggerBitsFifoReserved <= 64, 
              "Second 64-bit field overflow");

union alignas(16) ProxyTrigger {
  struct {
    uint64_t fst;
    uint64_t snd;
  };
  struct {
    uint64_t size : TriggerBitsSize;
    uint64_t srcOffset : TriggerBitsOffset;
    uint64_t : (64 - TriggerBitsSize - TriggerBitsOffset);
    
    uint64_t srcMemoryId : TriggerBitsMemoryId;
    uint64_t dstMemoryId : TriggerBitsMemoryId;
    uint64_t type : TriggerBitsType;
    uint64_t semaphoreId : TriggerBitsSemaphoreId;
    uint64_t : (64 - TriggerBitsMemoryId * 2 - TriggerBitsType - TriggerBitsSemaphoreId - TriggerBitsFifoReserved);
    uint64_t reserved : TriggerBitsFifoReserved;
  } fields;

  ProxyTrigger() = default;

  __device__ __forceinline__ ProxyTrigger(TriggerType type, uint32_t dstId, 
                                          uint32_t srcId, uint64_t offset, uint64_t bytes, 
                                          uint32_t semaphoreId) {
    ASSERT_DEVICE(offset < (1ull << TriggerBitsOffset), "offset is too large")
    ASSERT_DEVICE(type < (1ull << TriggerBitsType), "type is too large");
    ASSERT_DEVICE(dstId < (1ull << TriggerBitsMemoryId), "dstId is too large");
    ASSERT_DEVICE(srcId < (1ull << TriggerBitsMemoryId), "srcId is too large");
    ASSERT_DEVICE(bytes != 0, "bytes must not be zero");
    ASSERT_DEVICE(bytes < (1ull << TriggerBitsSize), "bytes is too large");
    ASSERT_DEVICE(semaphoreId < (1ull << TriggerBitsSemaphoreId), "semaphoreId is too large");

    constexpr uint64_t maskSize = (1ULL << TriggerBitsSize) - 1;
    constexpr uint64_t maskSrcOffset = (1ULL << TriggerBitsOffset) - 1;
    constexpr uint64_t maskDstOffset = (1ULL << TriggerBitsOffset) - 1;
    constexpr uint64_t maskSrcMemoryId = (1ULL << TriggerBitsMemoryId) - 1;
    constexpr uint64_t maskDstMemoryId = (1ULL << TriggerBitsMemoryId) - 1;
    constexpr uint64_t maskType = (1ULL << TriggerBitsType) - 1;
    constexpr uint64_t maskSemaphoreId = (1ULL << TriggerBitsSemaphoreId) - 1;
    
    fst = (((srcOffset & maskSrcOffset) << TriggerBitsSize) | (bytes & maskSize));
    snd = (((((((((semaphoreId & maskSemaphoreId) << TriggerBitsType) | ((uint64_t)type & maskType))
                << TriggerBitsMemoryId) |
               (dstId & maskDstMemoryId))
              << TriggerBitsMemoryId) |
             (srcId & maskSrcMemoryId))
            << TriggerBitsOffset) |
           (dstOffset & maskDstOffset));
  }

};

struct SpscHandle {
  __device__ __forceinline__ uint64_t push(ProxyTrigger trigger, int64_t maxSpinCount = 0xffffffff) {
    uint64_t localTailCache = *tailCache;
    uint64_t currentHead = atomicFetchAdd<uint64_t, scopeDevice>(head, 1, memoryOrderRelaxed);
    uint64_t wrappedHead = currentHead % size;
    
    if (currentHead - localTailCache >= size) {
      localTailCache = atomicLoad(tail, memoryOrderAcquire);
      *tailCache = localTailCache;
      
      if (currentHead - localTailCache >= size) {
        sync(currentHead - size, maxSpinCount);
        localTailCache = *tailCache;
      }
    }

    constexpr uint64_t flipMask = uint64_t{1} << uint64_t{63};
    trigger.snd ^= flipMask;

    ProxyTrigger* triggerPtr = &(triggers[wrappedHead]);
    atomicStore(&(triggerPtr->snd), trigger.snd, memoryOrderRelaxed);
    atomicStore(&(triggerPtr->fst), trigger.fst, memoryOrderRelease);

    return currentHead;
  }

  __device__ __forceinline__ void sync(uint64_t fifoHead, int64_t maxSpinCount = 0xffffffff) {
    uint64_t currentTail;
    POLL_MAYBE_JAILBREAK((fifoHead >= (currentTail = atomicLoad(tail, memoryOrderAcquire))), maxSpinCount);
    uint64_t currentCache = *tailCache;
    while (currentTail > currentCache) {
      currentCache = atomicCompareExchange(tailCache, currentCache, currentTail);
    }
  }

  ProxyTrigger* triggers;
  uint64_t* head;
  uint64_t* tail;
  uint64_t* tailCache;
  uint64_t size;
};

} // namespace pccl

