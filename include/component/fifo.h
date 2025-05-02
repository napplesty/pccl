#pragma once

#include <memory>

#include "config.h"
#include "device.h"

namespace pccl {

struct alignas(16) ProxyTrigger {
  uint64_t fst, snd;
};

struct FifoDeviceHandle {
#if defined(PCCL_CUDA_DEVICE_COMPILE)
  PCCL_CUDA_DEVICE_INLINE uint64_t push(ProxyTrigger trigger,
                                        int64_t maxSpinCount = 1000000) {
    uint64_t curFifoHead =
        atomicFetchAdd(this->head, (uint64_t)1, memoryOrderRelaxed);
    trigger.snd ^= ((uint64_t)1 << (uint64_t)63);
    if (curFifoHead >= size + *(this->tailReplica)) {
      OR_POLL_MAYBE_JAILBREAK(
          (curFifoHead >=
           size + atomicLoad(this->tailReplica, memoryOrderRelaxed)),
          (atomicLoad(&(this->triggers[curFifoHead % size].fst),
                      memoryOrderRelaxed) != 0),
          maxSpinCount);
    }
    ProxyTrigger *triggerPtr = &(this->triggers[curFifoHead % size]);
#if defined(USE_CUDA)
    asm volatile(
        "st.global.release.sys.v2.u64 [%0], {%1,%2};" ::"l"(triggerPtr),
        "l"(trigger.fst), "l"(trigger.snd));
#else
    atomicStore(&(triggerPtr->snd), trigger.snd, memoryOrderRelaxed);
    atomicStore(&(triggerPtr->fst), trigger.fst, memoryOrderRelaxed);
#endif
    return curFifoHead;
  }

  PCCL_CUDA_DEVICE_INLINE void sync(uint64_t curFifoHead,
                                    int64_t maxSpinCount = 1000000) {
    OR_POLL_MAYBE_JAILBREAK(
        (curFifoHead >= atomicLoad(this->tailReplica, memoryOrderRelaxed)),
        (atomicLoad(&(this->triggers[curFifoHead % size].fst),
                    memoryOrderRelaxed) != 0),
        maxSpinCount);
  }
#endif
  ProxyTrigger *triggers;
  uint64_t *tailReplica;
  uint64_t *head;
  int size;
};

class Fifo {
 public:
  Fifo(int size = DEFAULT_FIFO_SIZE);
  ~Fifo();
  ProxyTrigger poll();
  void pop();
  void flushTail(bool sync = false);
  int size() const;
  FifoDeviceHandle deviceHandle();

 private:
  struct Impl;
  ::std::unique_ptr<Impl> pimpl;
};

}  // namespace pccl
