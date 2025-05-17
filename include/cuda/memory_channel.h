#pragma once

#include "cuda/reduce_kernel.h"
#include "cuda/semaphore.h"

namespace pccl {
#if defined(PCCL_CUDA_DEVICE_COMPILE)

#if defined(USE_CUDA)

template <typename T>
PCCL_CUDA_DEVICE_INLINE void regCopy(T* dst, T* src, uint64_t numElems, uint32_t threadId,
                                     uint32_t numThreads) {
  T reg;
  for (size_t i = threadId; i < numElems; i += numThreads) {
    reg = src[i];
    dst[i] = reg;
  }
}

#else
template <typename T>
PCCL_CUDA_DEVICE_INLINE void regCopy(T* dst, T* src, uint64_t numElems, uint32_t threadId,
                                     uint32_t numThreads) {
  T reg;
  for (size_t i = threadId; i < numElems; i += numThreads) {
    reg = src[i];
    dst[i] = reg;
  }
}

#endif

#endif

struct MemoryChannelDeviceHandle {
  MemoryDevice2DeviceSemaphoreDeviceHandle semaphore_;
  void* src_;
  void* dst_;
  void* getPacketBuffer_;
#if defined(PCCL_CUDA_DEVICE_COMPILE)
  template <typename T>
  PCCL_CUDA_DEVICE_INLINE T read(uint64_t index) {
    return *(reinterpret_cast<T*>(dst_) + index);
  }

  template <typename T>
  PCCL_CUDA_DEVICE_INLINE void write(uint64_t index, const T& v) {
    *(reinterpret_cast<T*>(dst_) + index) = v;
  }

  template <typename T, bool CopyRemainder = true>
  PCCL_CUDA_DEVICE_INLINE void copy_helper(void* dst, void* src, uint64_t bytes, uint32_t threadId,
                                           uint32_t numThreads) {
    int* dstInt = reinterpret_cast<int*>(dst);
    int* srcInt = reinterpret_cast<int*>(src);
    const uintptr_t dstPtr = reinterpret_cast<uintptr_t>(dst);
    const uintptr_t srcPtr = reinterpret_cast<uintptr_t>(src);
    const uint64_t numInt = bytes / sizeof(int);
    T* dstElem = reinterpret_cast<T*>((dstPtr + sizeof(T) - 1) / sizeof(T) * sizeof(T));
    T* srcElem = reinterpret_cast<T*>((srcPtr + sizeof(T) - 1) / sizeof(T) * sizeof(T));
    uint64_t nFirstInt = (reinterpret_cast<uintptr_t>(dstElem) - dstPtr) / sizeof(int);
    if (CopyRemainder) {
      regCopy<int>(dstInt, srcInt, nFirstInt, threadId, numThreads);
    }
    constexpr uint64_t nIntPerElem = sizeof(T) / sizeof(int);
    uint64_t nElem = (numInt - nFirstInt) / nIntPerElem;
    regCopy<T>(dstElem, srcElem, nElem, threadId, numThreads);
    if (CopyRemainder && nIntPerElem > 1) {
      uint64_t nLastInt = (numInt - nFirstInt) % nIntPerElem;
      regCopy<int>(dstInt + nFirstInt + nElem * nIntPerElem,
                   srcInt + nFirstInt + nElem * nIntPerElem, nLastInt, threadId, numThreads);
    }
  }

  template <int Alignment = 16, bool CopyRemainder = true>
  PCCL_CUDA_DEVICE_INLINE void copy(void* dst, void* src, uint64_t bytes, uint32_t threadId,
                                    uint32_t numThreads) {
    if (Alignment == 4) {
      copy_helper<int, CopyRemainder>(dst, src, bytes, threadId, numThreads);
    } else if (Alignment == 8) {
      copy_helper<long long, CopyRemainder>(dst, src, bytes, threadId, numThreads);
    } else if (Alignment == 16) {
      copy_helper<longlong2, CopyRemainder>(dst, src, bytes, threadId, numThreads);
    } else {
      static_assert(Alignment == 4 || Alignment == 8 || Alignment == 16, "Unsupported alignment");
    }
  }

  template <int Alignment = 16, bool CopyRemainder = true>
  PCCL_CUDA_DEVICE_INLINE void put(uint64_t targetOffset, uint64_t originOffset,
                                   uint64_t originBytes, uint32_t threadId, uint32_t numThreads) {
    copy<Alignment, CopyRemainder>((char*)dst_ + targetOffset, (char*)src_ + originOffset,
                                   originBytes, threadId, numThreads);
  }

  template <int Alignment = 16, bool CopyRemainder = true>
  PCCL_CUDA_DEVICE_INLINE void get(uint64_t targetOffset, uint64_t originOffset,
                                   uint64_t originBytes, uint32_t threadId, uint32_t numThreads) {
    copy<Alignment, CopyRemainder>((char*)src_ + originOffset, (char*)dst_ + targetOffset,
                                   originBytes, threadId, numThreads);
  }

  template <int Alignment = 16, bool CopyRemainder = true>
  PCCL_CUDA_DEVICE_INLINE void put(uint64_t offset, uint64_t bytes, uint32_t threadId,
                                   uint32_t numThreads) {
    put<Alignment, CopyRemainder>(offset, offset, bytes, threadId, numThreads);
  }

  template <int Alignment = 16, bool CopyRemainder = true>
  PCCL_CUDA_DEVICE_INLINE void get(uint64_t offset, uint64_t bytes, uint32_t threadId,
                                   uint32_t numThreads) {
    get<Alignment, CopyRemainder>(offset, offset, bytes, threadId, numThreads);
  }

  PCCL_CUDA_DEVICE_INLINE void signal() { semaphore_.signal(); }

  PCCL_CUDA_DEVICE_INLINE void semaphoreIncrement() { semaphore_.semaphoreIncrement(); }

  PCCL_CUDA_DEVICE_INLINE uint64_t semaphoreGetLocal() const {
    return semaphore_.semaphoreGetLocal();
  }

  PCCL_CUDA_DEVICE_INLINE bool poll() { return semaphore_.poll(); }

  PCCL_CUDA_DEVICE_INLINE void wait(int64_t maxSpinCount = 10000000) {
    semaphore_.wait(maxSpinCount);
  }
#endif
};

struct MemoryChannel {
 private:
  std::shared_ptr<MemoryDevice2DeviceSemaphore> semaphore_;
  std::shared_ptr<RegisteredMemory> dst_;
  void* src_;
  void* getPacketBuffer_;

 public:
  MemoryChannel() = default;

  MemoryChannel(std::shared_ptr<MemoryDevice2DeviceSemaphore> semaphore, RegisteredMemory dst,
                void* src, void* getPacketBuffer = nullptr);
  using DeviceHandle = MemoryChannelDeviceHandle;
  DeviceHandle deviceHandle() const;
};

}  // namespace pccl
