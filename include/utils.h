#pragma once

#include <pthread.h>
#include <sched.h>
#include <unistd.h>

#include <cstdint>
#include <future>

#define PCCL_API __attribute__((visibility("default")))
#define PCCL_OFFSET_OF(type, member) ((size_t)&((type*)0)->member)

namespace pccl {

PCCL_API int bindToCpu();
PCCL_API uint64_t getHostHash();
PCCL_API uint64_t getPidHash();

template <typename T>
class NonblockingFuture {
  std::shared_future<T> future;

 public:
  NonblockingFuture() = default;
  NonblockingFuture(std::shared_future<T>&& future) : future(std::move(future)) {}
  bool ready() const {
    return future.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
  }

  T get() const {
    if (!ready()) throw std::runtime_error("NonblockingFuture::get() called before ready");
    return future.get();
  }
};
}  // namespace pccl