#pragma once

#include "device.h"
#include "runtime.h"
#include "utils.h"

namespace pccl {

struct Host2DeviceSemaphoreDeviceHandle {
#if defined(PCCL_CUDA_DEVICE_COMPILE)
  PCCL_CUDA_DEVICE_INLINE bool poll() {
    bool signaled =
        (atomicLoad(inboundSemaphoreId, memoryOrderAcquire) > *expectedInboundSemaphoreId);
    if (signaled) (*expectedInboundSemaphoreId)++;
    return signaled;
  }

  PCCL_CUDA_DEVICE_INLINE void wait(int64_t max_spin_cnt = 3e9) {
    POLL_MAYBE_JAILBREAK(
        (atomicLoad(inboundSemaphoreId, memoryOrderRelaxed) < *expectedInboundSemaphoreId),
        max_spin_cnt);
    (*expectedInboundSemaphoreId)++;
  }
#endif
  uint64_t* inboundSemaphoreId;
  uint64_t* expectedInboundSemaphoreId;
};

struct MemoryDevice2DeviceSemaphoreDeviceHandle {
#if defined(PCCL_CUDA_DEVICE_COMPILE)
  PCCL_CUDA_DEVICE_INLINE bool poll() {
    bool signaled =
        (atomicLoad(inboundSemaphoreId, memoryOrderAcquire) > (*expectedInboundSemaphoreId));
    if (signaled) (*expectedInboundSemaphoreId) += 1;
    return signaled;
  }

  PCCL_CUDA_DEVICE_INLINE void wait(int64_t maxSpinCount = 1e9) {
    (*expectedInboundSemaphoreId) += 1;
    POLL_MAYBE_JAILBREAK(
        (atomicLoad(inboundSemaphoreId, memoryOrderAcquire) < (*expectedInboundSemaphoreId)),
        maxSpinCount);
  }

  PCCL_CUDA_DEVICE_INLINE void signal() {
    semaphoreIncrement();
    atomicStore(remoteInboundSemaphoreId, semaphoreGetLocal(), memoryOrderSeqCst);
  }

  PCCL_CUDA_DEVICE_INLINE void relaxedSignal() {
    semaphoreIncrement();
    atomicStore(remoteInboundSemaphoreId, semaphoreGetLocal(), memoryOrderRelaxed);
  }

  PCCL_CUDA_DEVICE_INLINE void signalPacket() {
    semaphoreIncrement();
    *remoteInboundSemaphoreId = semaphoreGetLocal();
  }

  PCCL_CUDA_DEVICE_INLINE void semaphoreIncrement() { *outboundSemaphoreId += 1; }

  PCCL_CUDA_DEVICE_INLINE uint64_t semaphoreGetLocal() const { return *outboundSemaphoreId; }
#endif

  uint64_t* inboundSemaphoreId;
  uint64_t* outboundSemaphoreId;
  uint64_t* remoteInboundSemaphoreId;
  uint64_t* expectedInboundSemaphoreId;
};

template <template <typename> typename InboundDeleter, template <typename> typename OutboundDeleter>
class BaseSemaphore {
 protected:
  NonblockingFuture<RegisteredMemory> remoteInboundSemaphoreIdsRegMem_;
  std::unique_ptr<uint64_t, InboundDeleter<uint64_t>> localInboundSemaphore_;
  std::unique_ptr<uint64_t, InboundDeleter<uint64_t>> expectedInboundSemaphore_;
  std::unique_ptr<uint64_t, OutboundDeleter<uint64_t>> outboundSemaphore_;

 public:
  BaseSemaphore(std::unique_ptr<uint64_t, InboundDeleter<uint64_t>> localInboundSemaphoreId,
                std::unique_ptr<uint64_t, InboundDeleter<uint64_t>> expectedInboundSemaphoreId,
                std::unique_ptr<uint64_t, OutboundDeleter<uint64_t>> outboundSemaphoreId)
      : localInboundSemaphore_(std::move(localInboundSemaphoreId)),
        expectedInboundSemaphore_(std::move(expectedInboundSemaphoreId)),
        outboundSemaphore_(std::move(outboundSemaphoreId)) {}
};

class Host2DeviceSemaphore : public BaseSemaphore<GpuDeleter, std::default_delete> {
 private:
  std::shared_ptr<Connection> connection_;

 public:
  Host2DeviceSemaphore(Communicator& communicator, std::shared_ptr<Connection> connection);
  std::shared_ptr<Connection> connection();
  void signal();
  using DeviceHandle = Host2DeviceSemaphoreDeviceHandle;
  DeviceHandle deviceHandle();
};

class Host2HostSemaphore : public BaseSemaphore<std::default_delete, std::default_delete> {
 public:
  Host2HostSemaphore(Communicator& communicator, std::shared_ptr<Connection> connection);
  std::shared_ptr<Connection> connection();
  void signal();
  bool poll();
  void wait(int64_t maxSpinCount = 1e9);

 private:
  std::shared_ptr<Connection> connection_;
};

class MemoryDevice2DeviceSemaphore : public BaseSemaphore<GpuDeleter, GpuDeleter> {
 public:
  MemoryDevice2DeviceSemaphore(Communicator& communicator, std::shared_ptr<Connection> connection);
  MemoryDevice2DeviceSemaphore() = delete;
  using DeviceHandle = MemoryDevice2DeviceSemaphoreDeviceHandle;
  DeviceHandle deviceHandle() const;

  bool isRemoteInboundSemaphoreIdSet_;
};

}  // namespace pccl
