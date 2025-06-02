#pragma once

#include "cuda/internal_data_type.h"
#include "cuda/proxy.h"
#include "cuda/semaphore.h"

namespace pccl {

struct BasePortChannel;
struct PortChannel;

union ChannelTrigger {
  ProxyTrigger value;
#if defined(PCCL_CUDA_DEVICE_COMPILE)
  PCCL_CUDA_DEVICE_INLINE ChannelTrigger() {}
  PCCL_CUDA_DEVICE_INLINE ChannelTrigger(ProxyTrigger trigger) : value(std::move(trigger)) {}
  PCCL_CUDA_DEVICE_INLINE ChannelTrigger(TriggerType type, MemoryId dstMemoryId, uint64_t dstOffset, MemoryId srcMemoryId, uint64_t srcOffset,
                                         uint64_t size, SemaphoreId chanId) {
    constexpr uint64_t maskSize = (1ULL << 32) - 1;
    constexpr uint64_t maskSrcOffset = (1ULL << 32) - 1;
    constexpr uint64_t maskDstOffset = (1ULL << 32) - 1;
    constexpr uint64_t maskSrcMemoryId = (1ULL << 8) - 1;
    constexpr uint64_t maskDstMemoryId = (1ULL << 8) - 1;
    constexpr uint64_t maskType = (1ULL << 4) - 1;
    constexpr uint64_t maskChanId = (1ULL << 12) - 1;
    value.fst = (((srcOffset & maskSrcOffset) << 32) + (size & maskSize));
    value.snd = (((((((((chanId & maskChanId) << 4) + ((uint64_t)type & maskType)) << 8) + (dstMemoryId & maskDstMemoryId)) << 8) +
                   (srcMemoryId & maskSrcMemoryId))
                  << 32) +
                 (dstOffset & maskDstOffset));
  }
#endif
};

struct BasePortChannelDeviceHandle {
  SemaphoreId semaphoreId_;
  Host2DeviceSemaphoreDeviceHandle semaphore_;
  FifoDeviceHandle fifo_;

  PCCL_CUDA_DEVICE_INLINE BasePortChannelDeviceHandle() {}
  PCCL_CUDA_DEVICE_INLINE BasePortChannelDeviceHandle(SemaphoreId semaphoreId, Host2DeviceSemaphoreDeviceHandle semaphore, FifoDeviceHandle fifo)
      : semaphoreId_(semaphoreId), semaphore_(semaphore), fifo_(fifo) {}

#if defined(PCCL_CUDA_DEVICE_COMPILE)
  PCCL_CUDA_DEVICE_INLINE void put(MemoryId dst, uint64_t dstOffset, MemoryId src, uint64_t srcOffset, uint64_t size) {
    fifo_.push(ChannelTrigger(TriggerData, dst, dstOffset, src, srcOffset, size, semaphoreId_).value);
  }

  PCCL_CUDA_DEVICE_INLINE void put(MemoryId dst, MemoryId src, uint64_t offset, uint64_t size) { put(dst, offset, src, offset, size); }

  PCCL_CUDA_DEVICE_INLINE void flush() {
    uint64_t curFifoHead = fifo_.push(ChannelTrigger(TriggerSync, 0, 0, 0, 0, 1, semaphoreId_).value);
    fifo_.sync(curFifoHead);
  }

  PCCL_CUDA_DEVICE_INLINE void putWithSignal(MemoryId dst, uint64_t dstOffset, MemoryId src, uint64_t srcOffset, uint64_t size) {
    fifo_.push(ChannelTrigger(TriggerData | TriggerSignal, dst, dstOffset, src, srcOffset, size, semaphoreId_).value);
  }

  PCCL_CUDA_DEVICE_INLINE void putWithSignalAndFlush(MemoryId dst, uint64_t dstOffset, MemoryId src, uint64_t srcOffset, uint64_t size) {
    uint64_t curFifoHead =
        fifo_.push(ChannelTrigger(TriggerData | TriggerSignal | TriggerSync, dst, dstOffset, src, srcOffset, size, semaphoreId_).value);
    fifo_.sync(curFifoHead);
  }

  PCCL_CUDA_DEVICE_INLINE void signal() { fifo_.push(ChannelTrigger(TriggerSignal, 0, 0, 0, 0, 1, semaphoreId_).value); }

  PCCL_CUDA_DEVICE_INLINE bool poll() { return semaphore_.poll(); }

  PCCL_CUDA_DEVICE_INLINE void wait(int64_t maxSpinCount = 10000000) { semaphore_.wait(maxSpinCount); }
#endif
};

struct PortChannelDeviceHandle : public BasePortChannelDeviceHandle {
  MemoryId dst_;
  MemoryId src_;
#if defined(PCCL_CUDA_DEVICE_COMPILE)
  PCCL_CUDA_HOST_DEVICE_INLINE PortChannelDeviceHandle() {};

  PCCL_CUDA_HOST_DEVICE_INLINE PortChannelDeviceHandle(SemaphoreId semaphoreId, Host2DeviceSemaphoreDeviceHandle semaphore, FifoDeviceHandle fifo,
                                                       MemoryId dst, MemoryId src)
      : BasePortChannelDeviceHandle(semaphoreId, semaphore, fifo), dst_(dst), src_(src) {}

  PCCL_CUDA_DEVICE_INLINE void put(uint64_t dstOffset, uint64_t srcOffset, uint64_t size) {
    BasePortChannelDeviceHandle::put(dst_, dstOffset, src_, srcOffset, size);
  }

  PCCL_CUDA_DEVICE_INLINE void put(uint64_t offset, uint64_t size) { put(offset, offset, size); }

  PCCL_CUDA_DEVICE_INLINE void putWithSignal(uint64_t dstOffset, uint64_t srcOffset, uint64_t size) {
    BasePortChannelDeviceHandle::putWithSignal(dst_, dstOffset, src_, srcOffset, size);
  }

  PCCL_CUDA_DEVICE_INLINE void putWithSignal(uint64_t offset, uint64_t size) { putWithSignal(offset, offset, size); }

  PCCL_CUDA_DEVICE_INLINE void putWithSignalAndFlush(uint64_t dstOffset, uint64_t srcOffset, uint64_t size) {
    BasePortChannelDeviceHandle::putWithSignalAndFlush(dst_, dstOffset, src_, srcOffset, size);
  }

  PCCL_CUDA_DEVICE_INLINE void putWithSignalAndFlush(uint64_t offset, uint64_t size) { putWithSignalAndFlush(offset, offset, size); }
#endif
};

class BaseProxyService {
 public:
  BaseProxyService() = default;
  virtual ~BaseProxyService() = default;
  virtual void startProxy() = 0;
  virtual void stopProxy() = 0;
};

class ProxyService : public BaseProxyService {
 public:
  ProxyService(size_t fifoSize = Config::FIFO_BUFFER_SIZE);
  SemaphoreId buildAndAddSemaphore(Communicator& communicator, std::shared_ptr<Connection> connection);
  SemaphoreId addSemaphore(std::shared_ptr<Host2DeviceSemaphore> semaphore);
  MemoryId addMemory(RegisteredMemory memory);
  std::shared_ptr<Host2DeviceSemaphore> semaphore(SemaphoreId id) const;
  BasePortChannel basePortChannel(SemaphoreId id);
  PortChannel portChannel(SemaphoreId id, MemoryId dst, MemoryId src);

  void startProxy();
  void stopProxy();

 private:
  std::vector<std::shared_ptr<Host2DeviceSemaphore>> semaphores_;
  std::vector<RegisteredMemory> memories_;
  std::shared_ptr<Proxy> proxy_;
  int deviceNumaNode;
  std::unordered_map<std::shared_ptr<Connection>, int> inflightRequests;
  ProxyHandlerResult handleTrigger(ProxyTrigger triggerRaw);
};

struct BasePortChannel {
 protected:
  SemaphoreId semaphoreId_;
  std::shared_ptr<Host2DeviceSemaphore> semaphore_;
  std::shared_ptr<Proxy> proxy_;

 public:
  BasePortChannel() = default;
  BasePortChannel(SemaphoreId semaphoreId, std::shared_ptr<Host2DeviceSemaphore> semaphore, std::shared_ptr<Proxy> proxy);
  BasePortChannel(const BasePortChannel& other) = default;
  BasePortChannel& operator=(BasePortChannel& other) = default;
  using DeviceHandle = BasePortChannelDeviceHandle;
  DeviceHandle deviceHandle() const;
};

struct PortChannel : public BasePortChannel {
 private:
  MemoryId dst_;
  MemoryId src_;

 public:
  PortChannel() = default;
  PortChannel(SemaphoreId semaphoreId, std::shared_ptr<Host2DeviceSemaphore> semaphore, std::shared_ptr<Proxy> proxy, MemoryId dst, MemoryId src);

  PortChannel(const PortChannel& other) = default;
  PortChannel& operator=(PortChannel& other) = default;
  using DeviceHandle = PortChannelDeviceHandle;
  DeviceHandle deviceHandle() const;
};

}  // namespace pccl