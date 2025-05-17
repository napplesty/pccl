#pragma once

#include "config.h"
#include "device.h"

namespace pccl {

struct alignas(16) ProxyTrigger {
  uint64_t fst, snd;
};

struct DeviceFifoHandle {
  PCCL_CUDA_DEVICE_INLINE uint64_t push(ProxyTrigger trigger, int64_t maxSpinCount = 3e9) {
    uint64_t curFifoTail = atomicFetchAdd(tail, (uint64_t)1, memoryOrderRelaxed);
    trigger.snd ^= (1ull << 53);

    // 检查FIFO是否已满，如果已满则等待
    uint64_t curHead = atomicLoad(head, memoryOrderAcquire);
    if (curFifoTail >= Config::FIFO_BUFFER_SIZE + curHead) {
      POLL_MAYBE_JAILBREAK(
          (curFifoTail >= Config::FIFO_BUFFER_SIZE + atomicLoad(head, memoryOrderAcquire)),
          maxSpinCount);
    }

    // 写入触发器到FIFO缓冲区
    ProxyTrigger *triggerPtr = &(triggers[curFifoTail % Config::FIFO_BUFFER_SIZE]);
    atomicStore(&(triggerPtr->snd), trigger.snd, memoryOrderRelaxed);
    // 最后写入fst值，使用release内存序确保之前的写入对其他线程可见
    atomicStore(&(triggerPtr->fst), trigger.fst, memoryOrderRelease);

    return curFifoTail;
  }

  PCCL_CUDA_DEVICE_INLINE ProxyTrigger poll() {
    ProxyTrigger trigger;
    uint64_t curHead = atomicLoad(head, memoryOrderRelaxed);
    ProxyTrigger *ptr = triggers + (curHead % Config::FIFO_BUFFER_SIZE);

    // 使用acquire内存序读取fst，确保能看到最新的写入
    trigger.fst = atomicLoad(&ptr->fst, memoryOrderAcquire);
    trigger.snd = ptr->snd;

    return trigger;
  }

  PCCL_CUDA_DEVICE_INLINE void pop() {
    uint64_t curHead = atomicFetchAdd(head, (uint64_t)1, memoryOrderRelaxed);
    // 使用release内存序清零fst，确保这个操作对其他线程可见
    atomicStore(&(triggers[curHead % Config::FIFO_BUFFER_SIZE].fst), uint64_t{0},
                memoryOrderRelease);
  }

  PCCL_CUDA_DEVICE_INLINE void sync(uint64_t expected_head, int64_t maxSpinCount = 3e9) {
    // 使用acquire内存序读取head，确保能看到最新的写入
    if (expected_head > atomicLoad(head, memoryOrderAcquire)) {
      POLL_MAYBE_JAILBREAK((expected_head <= atomicLoad(head, memoryOrderAcquire)), maxSpinCount);
    }
  }

  ProxyTrigger *triggers;
  uint64_t *tail;
  uint64_t *head;
};

struct FifoDeviceHandle {
#if defined(PCCL_CUDA_DEVICE_COMPILE)
  PCCL_CUDA_DEVICE_INLINE uint64_t push(ProxyTrigger trigger, int64_t maxSpinCount = 3e9) {
    uint64_t curFifoHead = atomicFetchAdd(this->head, (uint64_t)1, memoryOrderRelaxed);
    trigger.snd ^= (1ull << 53);
    if (curFifoHead >= Config::FIFO_BUFFER_SIZE + *(this->tailReplica)) {
      OR_POLL_MAYBE_JAILBREAK(
          (curFifoHead >=
           Config::FIFO_BUFFER_SIZE + atomicLoad(this->tailReplica, memoryOrderRelaxed)),
          (atomicLoad(&(this->triggers[curFifoHead % Config::FIFO_BUFFER_SIZE].fst),
                      memoryOrderRelaxed) != 0),
          maxSpinCount);
    }
    ProxyTrigger *triggerPtr = &(this->triggers[curFifoHead % Config::FIFO_BUFFER_SIZE]);
    atomicStore(&(triggerPtr->snd), trigger.snd, memoryOrderRelaxed);
    atomicStore(&(triggerPtr->fst), trigger.fst, memoryOrderRelaxed);
    return curFifoHead;
  }

  PCCL_CUDA_DEVICE_INLINE void sync(uint64_t curFifoHead, int64_t maxSpinCount = 1000000) {
    OR_POLL_MAYBE_JAILBREAK(
        (curFifoHead >= atomicLoad(this->tailReplica, memoryOrderRelaxed)),
        (atomicLoad(&(this->triggers[curFifoHead % Config::FIFO_BUFFER_SIZE].fst),
                    memoryOrderRelaxed) != 0),
        maxSpinCount);
  }
#endif
  ProxyTrigger *triggers;
  uint64_t *tailReplica;
  uint64_t *head;
};

class Fifo {
 public:
  Fifo();
  ~Fifo();
  ProxyTrigger poll();
  void pop();
  void flushTail(bool sync = false);
  int size() const;
  FifoDeviceHandle deviceHandle();
  using DeviceHandle = FifoDeviceHandle;

 private:
  struct Impl;
  ::std::unique_ptr<Impl> pimpl;
};

enum class ProxyHandlerResult {
  Continue,
  FlushFifoTailAndContinue,
  Stop,
};

}  // namespace pccl
