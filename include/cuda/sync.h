#pragma once

#include "config.h"
#include "device.h"

namespace pccl {

struct DeviceSyncer {
 public:
  DeviceSyncer() {}
  ~DeviceSyncer() = default;

#if defined(PCCL_CUDA_DEVICE_COMPILE)
  PCCL_CUDA_DEVICE_INLINE void sync(int smNum, int64_t maxSpinCount = 3e9) {
    unsigned int maxOldCnt = smNum - 1;
    __syncthreads();
    if (smNum == 1) return;
    if (threadIdx.x == 0) {
      __threadfence();
      unsigned int tmp = preFlag_ ^ 1;
      if (atomicInc(&count_, maxOldCnt) == maxOldCnt) {
        atomicStore(&flag_, tmp, memoryOrderRelaxed);
        atomicStore(&count_, 0u, memoryOrderRelaxed);
      } else {
        POLL_MAYBE_JAILBREAK((atomicLoad(&flag_, memoryOrderRelaxed) != tmp), maxSpinCount);
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

#if defined(PCCL_CUDA_DEVICE_COMPILE)
#endif

}  // namespace pccl
