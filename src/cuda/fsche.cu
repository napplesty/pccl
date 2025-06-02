#include "config.h"
#include "cuda/fsche.h"
#include "utils.h"

namespace pccl {

struct Fifo::Impl {
  UniqueGpuHostPtr<ProxyTrigger> triggers;
  UniqueGpuPtr<uint64_t> head;
  UniqueGpuPtr<uint64_t> tailReplica;
  constexpr static int size = Config::FIFO_BUFFER_SIZE;
  uint64_t hostTail;
  CudaStreamWithFlags stream;

  Impl()
      : triggers(gpuCallocHostUnique<ProxyTrigger>(size)),
        head(gpuCallocUnique<uint64_t>()),
        tailReplica(gpuCallocUnique<uint64_t>()), hostTail(0),
        stream(cudaStreamNonBlocking) {}
};

struct InterSmFifo::Impl {
  int num_sm;

  UniqueGpuPtr<InterSmMessage> fifos;
  UniqueGpuPtr<uint64_t> tails;
  UniqueGpuPtr<uint64_t> heads;

  InterSmMessage **fifo_ptrs;
  uint64_t **tail_ptrs;
  uint64_t **head_ptrs;

  Impl(int num_sm)
      : num_sm(num_sm), fifos(gpuCallocUnique<InterSmMessage>(
                            num_sm * Config::INTER_SM_FIFO_SIZE)),
        tails(gpuCallocUnique<uint64_t>(num_sm)),
        heads(gpuCallocUnique<uint64_t>(num_sm)),
        fifo_ptrs(new InterSmMessage *[num_sm]),
        tail_ptrs(new uint64_t *[num_sm]), head_ptrs(new uint64_t *[num_sm]) {
    CUDACHECK(cudaMemset(tails.get(), 0, num_sm * sizeof(uint64_t)));
    CUDACHECK(cudaMemset(heads.get(), 0, num_sm * sizeof(uint64_t)));
    CUDACHECK(cudaMemset(fifos.get(), 0,
                         num_sm * Config::INTER_SM_FIFO_SIZE *
                             sizeof(InterSmMessage)));
    for (int i = 0; i < num_sm; i++) {
      fifo_ptrs[i] = (fifos.get() + i * Config::INTER_SM_FIFO_SIZE);
      tail_ptrs[i] = tails.get() + i;
      head_ptrs[i] = heads.get() + i;
    }
  }

  ~Impl() {
    delete[] fifo_ptrs;
    delete[] tail_ptrs;
    delete[] head_ptrs;
  }
};

PCCL_API Fifo::Fifo() : pimpl(std::make_unique<Impl>()) {}
PCCL_API Fifo::~Fifo() = default;

PCCL_API ProxyTrigger Fifo::poll() {
  ProxyTrigger trigger;
  ProxyTrigger *ptr = pimpl->triggers.get() + pimpl->hostTail % pimpl->size;
  trigger.fst = atomicLoad(&ptr->fst, memoryOrderRelaxed);
  trigger.snd = ptr->snd;
  return trigger;
}

PCCL_API void Fifo::pop() {
  atomicStore(&(pimpl->triggers.get()[pimpl->hostTail % pimpl->size].fst),
              uint64_t{0}, memoryOrderRelease);
  (pimpl->hostTail)++;
}

PCCL_API void Fifo::flushTail(bool sync) {
  AvoidCudaGraphCaptureGuard cgcGuard;
  CUDACHECK(cudaMemcpyAsync(pimpl->tailReplica.get(), &pimpl->hostTail,
                            sizeof(uint64_t), cudaMemcpyHostToDevice,
                            pimpl->stream));
  if (sync) {
    CUDACHECK(cudaStreamSynchronize(pimpl->stream));
  }
}

PCCL_API int Fifo::size() const { return pimpl->size; }

PCCL_API FifoDeviceHandle Fifo::deviceHandle() {
  FifoDeviceHandle deviceHandle;
  deviceHandle.trigger_buffer = pimpl->triggers.get();
  deviceHandle.head = pimpl->head.get();
  deviceHandle.tail = pimpl->tailReplica.get();
  return deviceHandle;
}

PCCL_API InterSmFifo::InterSmFifo(int num_sm)
    : pimpl(std::make_unique<Impl>(num_sm)) {}
PCCL_API InterSmFifo::~InterSmFifo() = default;

PCCL_API InterSmFifoDeviceHandle InterSmFifo::deviceHandle() {
  InterSmFifoDeviceHandle deviceHandle;
  deviceHandle.fifos = pimpl->fifo_ptrs;
  deviceHandle.tails = pimpl->tail_ptrs;
  deviceHandle.heads = pimpl->head_ptrs;
  return deviceHandle;
}

} // namespace pccl