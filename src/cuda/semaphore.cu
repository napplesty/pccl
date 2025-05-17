#include "cuda/semaphore.h"

namespace pccl {

static UniqueGpuPtr<uint64_t> createGpuSemaphoreId() {
#if defined(USE_CUDA)
  return gpuCallocUnique<uint64_t>();
#elif defined(USE_HIP)
  return gpuCallocUncachedUnique<uint64_t>();
#else
  throw std::runtime_error("Unsupported platform");
#endif  // !defined(USE_CUDA)
}

}  // namespace pccl