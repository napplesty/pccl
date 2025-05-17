#include "cuda/proxy.h"

#include <atomic>
#include <functional>
#include <memory>
#include <thread>

#include "component/logging.h"
#include "config.h"
#include "device.h"
#include "runtime.h"
#include "utils.h"

namespace pccl {

struct Fifo::Impl {
  UniqueGpuHostPtr<ProxyTrigger> triggers;
  UniqueGpuPtr<uint64_t> head;
  UniqueGpuPtr<uint64_t> tailReplica;
  const int size;
  uint64_t hostTail;

  // for transferring fifo tail
  CudaStreamWithFlags stream;

  Impl(int size)
      : triggers(gpuCallocHostUnique<ProxyTrigger>(size)),
        head(gpuCallocUnique<uint64_t>()),
        tailReplica(gpuCallocUnique<uint64_t>()),
        size(size),
        hostTail(0),
        stream(cudaStreamNonBlocking) {}
};

PCCL_API Fifo::Fifo(int size) : pimpl(::std::make_unique<Impl>(size)) {}
PCCL_API Fifo::~Fifo() = default;

PCCL_API ProxyTrigger Fifo::poll() {
  ProxyTrigger trigger;
  ProxyTrigger* ptr = pimpl->triggers.get() + pimpl->hostTail % pimpl->size;
  trigger.fst = atomicLoad(&ptr->fst, memoryOrderRelaxed);
  trigger.snd = ptr->snd;
  return trigger;
}

PCCL_API void Fifo::pop() {
  atomicStore(&(pimpl->triggers.get()[pimpl->hostTail % pimpl->size].fst), uint64_t{0},
              memoryOrderRelease);
  (pimpl->hostTail)++;
}

PCCL_API void Fifo::flushTail(bool sync) {
  AvoidCudaGraphCaptureGuard cgcGuard;
  CUDACHECK(cudaMemcpyAsync(pimpl->tailReplica.get(), &pimpl->hostTail, sizeof(uint64_t),
                            cudaMemcpyHostToDevice, pimpl->stream));
  if (sync) {
    CUDACHECK(cudaStreamSynchronize(pimpl->stream));
  }
}

PCCL_API int Fifo::size() const { return pimpl->size; }

PCCL_API FifoDeviceHandle Fifo::deviceHandle() {
  FifoDeviceHandle deviceHandle;
  deviceHandle.triggers = pimpl->triggers.get();
  deviceHandle.head = pimpl->head.get();
  deviceHandle.tailReplica = pimpl->tailReplica.get();
  deviceHandle.size = pimpl->size;
  return deviceHandle;
}

struct Proxy::Impl {
  ProxyHandler handler;
  Fifo fifo;
  ::std::thread service;
  ::std::atomic<bool> running;

  Impl(ProxyHandler handler, size_t fifoSize) : handler(handler), fifo(fifoSize), running(false) {}
};

PCCL_API Proxy::Proxy(ProxyHandler handler, size_t fifoSize) {
  pimpl = ::std::make_unique<Impl>(handler, fifoSize);
}

PCCL_API Proxy::~Proxy() {
  if (pimpl) {
    stop();
  }
}

PCCL_API void Proxy::start() {
  int cudaDevice;
  CUDACHECK(cudaGetDevice(&cudaDevice));

  pimpl->running = true;
  pimpl->service = ::std::thread([this, cudaDevice] {
    bindToCpu();
    CUDACHECK(cudaSetDevice(cudaDevice));

    ProxyHandler handler = this->pimpl->handler;
    Fifo& fifo = this->pimpl->fifo;
    ::std::atomic<bool>& running = this->pimpl->running;
    ProxyTrigger trigger;

    int flushPeriod = ::std::min(fifo.size(), Config::ProxyFlushPeriod);

    int runCnt = Config::ProxyStopCheckPeriod;
    uint64_t flushCnt = 0;
    for (;;) {
      if (runCnt-- == 0) {
        runCnt = Config::ProxyStopCheckPeriod;
        if (!running) {
          break;
        }
      }

      trigger = fifo.poll();
      if (trigger.fst == 0 || trigger.snd == 0) {
        continue;
      }
      trigger.snd ^= ((uint64_t)1 << (uint64_t)63);

      ProxyHandlerResult result = handler(trigger);
      fifo.pop();
      if ((++flushCnt % flushPeriod) == 0 ||
          result == ProxyHandlerResult::FlushFifoTailAndContinue) {
        fifo.flushTail();
      }

      if (result == ProxyHandlerResult::Stop) {
        break;
      }
    }

    fifo.flushTail(/*sync=*/true);
    ::std::this_thread::yield();
  });
}

PCCL_API void Proxy::stop() {
  pimpl->running = false;
  if (pimpl->service.joinable()) {
    pimpl->service.join();
  }
}

PCCL_API Fifo& Proxy::fifo() { return pimpl->fifo; }

}  // namespace pccl
