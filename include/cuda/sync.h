#pragma once

#include "device.h"

namespace pccl {

struct DeviceSyncer {
 public:
  DeviceSyncer() = default;
  ~DeviceSyncer() = default;

#if defined(PCCL_CUDA_DEVICE_COMPILE)
  PCCL_CUDA_DEVICE_INLINE void sync(int blockNum,
                                    int64_t maxSpinCount = 100000000) {
    unsigned int maxOldCnt = blockNum - 1;
    __syncthreads();
    if (blockNum == 1) return;
    if (threadIdx.x == 0) {
      __threadfence();
      unsigned int tmp = preFlag_ ^ 1;
      if (atomicInc(&count_, maxOldCnt) == maxOldCnt) {
        atomicStore(&flag_, tmp, memoryOrderRelaxed);
      } else {
        POLL_MAYBE_JAILBREAK((atomicLoad(&flag_, memoryOrderRelaxed) != tmp),
                             maxSpinCount);
      }
      preFlag_ = tmp;
    }
    __syncthreads();
  }
#endif
 private:
  unsigned int flag_;
  unsigned int count_;
  unsigned int preFlag_;
};

}  // namespace pccl
