// #include <atomic>
// #include <functional>
// #include <memory>
// #include <thread>

// #include "component/logging.h"
// #include "config.h"
// #include "cuda/proxy.h"
// #include "device.h"
// #include "runtime.h"
// #include "utils.h"

// namespace pccl {

// struct Proxy::Impl {
//   ProxyHandler handler;
//   Fifo fifo;
//   std::thread service;
//   std::atomic<bool> running;

//   Impl(ProxyHandler handler, size_t fifoSize)
//       : handler(handler), fifo(fifoSize), running(false) {}
// };

// PCCL_API Proxy::Proxy(ProxyHandler handler, size_t fifoSize) {
//   pimpl = std::make_unique<Impl>(handler, fifoSize);
// }

// PCCL_API Proxy::~Proxy() {
//   if (pimpl) {
//     stop();
//   }
// }

// PCCL_API void Proxy::start() {
//   int cudaDevice;
//   CUDACHECK(cudaGetDevice(&cudaDevice));

//   pimpl->running = true;
//   pimpl->service = std::thread([this, cudaDevice] {
//     bindToCpu();
//     CUDACHECK(cudaSetDevice(cudaDevice));

//     ProxyHandler handler = this->pimpl->handler;
//     Fifo &fifo = this->pimpl->fifo;
//     std::atomic<bool> &running = this->pimpl->running;
//     ProxyTrigger trigger;

//     int flushPeriod = std::min(fifo.size(), Config::ProxyFlushPeriod);

//     int runCnt = Config::ProxyStopCheckPeriod;
//     uint64_t flushCnt = 0;
//     for (;;) {
//       if (runCnt-- == 0) {
//         runCnt = Config::ProxyStopCheckPeriod;
//         if (!running) {
//           break;
//         }
//       }

//       trigger = fifo.poll();
//       if (trigger.fst == 0 || trigger.snd == 0) {
//         continue;
//       }
//       trigger.snd ^= ((uint64_t)1 << (uint64_t)63);

//       ProxyHandlerResult result = handler(trigger);
//       fifo.pop();
//       if ((++flushCnt % flushPeriod) == 0 ||
//           result == ProxyHandlerResult::FlushFifoTailAndContinue) {
//         fifo.flushTail();
//       }

//       if (result == ProxyHandlerResult::Stop) {
//         break;
//       }
//     }

//     fifo.flushTail(/*sync=*/true);
//     std::this_thread::yield();
//   });
// }

// PCCL_API void Proxy::stop() {
//   pimpl->running = false;
//   if (pimpl->service.joinable()) {
//     pimpl->service.join();
//   }
// }

// PCCL_API Fifo &Proxy::fifo() { return pimpl->fifo; }

// } // namespace pccl
