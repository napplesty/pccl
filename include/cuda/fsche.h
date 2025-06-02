#pragma once

#include "config.h"
#include "cuda/internal_data_type.h"
#include "device.h"

namespace pccl {

using ProxyTrigger = ProxyTriggerLayout;

struct FifoDeviceHandle {
  PCCL_CUDA_DEVICE_INLINE uint64_t push(ProxyTrigger trigger,
                                        int64_t maxSpinCount = 3e9) {
    uint64_t curFifoTail =
        atomicFetchAdd(tail, (uint64_t)1, memoryOrderRelaxed);
    trigger.setIssued();

    uint64_t curHead = atomicLoad(head, memoryOrderAcquire);
    if (curFifoTail >= Config::FIFO_BUFFER_SIZE + curHead) {
      AND_POLL_MAYBE_JAILBREAK(
          (curFifoTail >=
           Config::FIFO_BUFFER_SIZE + atomicLoad(head, memoryOrderAcquire)),
          trigger_buffer[curFifoTail % Config::FIFO_BUFFER_SIZE].checkIssued(),
          maxSpinCount);
    }
    ProxyTrigger *triggerPtr =
        &(trigger_buffer[curFifoTail % Config::FIFO_BUFFER_SIZE]);
    atomicStore(&(triggerPtr->snd), trigger.snd, memoryOrderRelaxed);
    atomicStore(&(triggerPtr->fst), trigger.fst, memoryOrderRelease);
    return curFifoTail;
  }

  PCCL_CUDA_DEVICE_INLINE ProxyTrigger poll() {
    ProxyTrigger trigger;
    uint64_t curHead = atomicLoad(head, memoryOrderRelaxed);
    ProxyTrigger *ptr = trigger_buffer + (curHead % Config::FIFO_BUFFER_SIZE);

    trigger.fst = atomicLoad(&ptr->fst, memoryOrderRelaxed);
    trigger.snd = atomicLoad(&ptr->snd, memoryOrderAcquire);
    return trigger;
  }

  PCCL_CUDA_DEVICE_INLINE void pop() {
    uint64_t curHead = atomicFetchAdd(head, (uint64_t)1, memoryOrderRelaxed);
    trigger_buffer[curHead % Config::FIFO_BUFFER_SIZE].clean();
  }

  PCCL_CUDA_DEVICE_INLINE void sync(uint64_t expected_head,
                                    int64_t maxSpinCount = 3e9) {
    if (expected_head > atomicLoad(head, memoryOrderAcquire)) {
      POLL_MAYBE_JAILBREAK(
          (expected_head <= atomicLoad(head, memoryOrderAcquire)),
          maxSpinCount);
    }
  }

  ProxyTrigger *trigger_buffer;
  uint64_t *tail;
  uint64_t *head;
};

using InterSmMessage = InterSmMessageLayout;
struct InterSmFifoDeviceHandle {
  PCCL_CUDA_DEVICE_INLINE uint64_t push(InterSmMessage msg, int dstSmIdx,
                                        int64_t maxSpinCount = 3e9) {
    InterSmMessage *fifo = fifos[dstSmIdx];
    uint64_t *tailPtr = tails[dstSmIdx];
    uint64_t *headPtr = heads[dstSmIdx];

    uint64_t curTail = atomicFetchAdd(tailPtr, (uint64_t)1, memoryOrderRelaxed);
    msg.setIssued();

    uint64_t curHead = atomicLoad(headPtr, memoryOrderAcquire);
    if (curTail >= Config::INTER_SM_FIFO_SIZE + curHead) {
      AND_POLL_MAYBE_JAILBREAK(
          (curTail >= Config::INTER_SM_FIFO_SIZE +
                          atomicLoad(headPtr, memoryOrderAcquire)),
          fifo[curTail % Config::INTER_SM_FIFO_SIZE].checkIssued(),
          maxSpinCount);
    }
    InterSmMessage *slot = &fifo[curTail % Config::INTER_SM_FIFO_SIZE];
    atomicStore(&slot->data[7], msg.data[7], memoryOrderRelease);
#pragma unroll 7
    for (int i = 0; i < 7; i++) {
      atomicStore(&slot->data[i], msg.data[i], memoryOrderRelease);
    }

    return curTail;
  }

  PCCL_CUDA_DEVICE_INLINE InterSmMessage poll(int srcSmId) {
    int fifoIdx = srcSmId;

    InterSmMessage *fifo = fifos[fifoIdx];
    uint64_t *headPtr = heads[fifoIdx];

    uint64_t curHead = atomicLoad(headPtr, memoryOrderRelaxed);
    InterSmMessage *slot = &fifo[curHead % Config::INTER_SM_FIFO_SIZE];

    InterSmMessage msg;
#pragma unroll 7
    for (int i = 0; i < 7; i++) {
      msg.data[i] = atomicLoad(&slot->data[i], memoryOrderRelaxed);
    }
    msg.data[7] = atomicLoad(&slot->data[7], memoryOrderAcquire);

    return msg;
  }

  PCCL_CUDA_DEVICE_INLINE void pop(int srcSmId) {
    int fifoIdx = srcSmId;

    InterSmMessage *fifo = fifos[fifoIdx];
    uint64_t *headPtr = heads[fifoIdx];

    uint64_t curHead = atomicFetchAdd(headPtr, (uint64_t)1, memoryOrderRelaxed);
    fifo[curHead % Config::INTER_SM_FIFO_SIZE].clean();
  }

  PCCL_CUDA_DEVICE_INLINE int getSmId() const { return blockIdx.x; }

  InterSmMessage **fifos;
  uint64_t **tails;
  uint64_t **heads;
  int num_sms;
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
  std::unique_ptr<Impl> pimpl;
};

class InterSmFifo {
public:
  InterSmFifo(int num_sm);
  ~InterSmFifo();
  InterSmFifoDeviceHandle deviceHandle();
  using DeviceHandle = InterSmFifoDeviceHandle;

private:
  struct Impl;
  std::unique_ptr<Impl> pimpl;
};

} // namespace pccl
