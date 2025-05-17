#pragma once

#include <functional>
#include <memory>

#include "config.h"
#include "device.h"
#include "fsche.h"
namespace pccl {

class Proxy;
using ProxyHandler = ::std::function<ProxyHandlerResult(ProxyTrigger)>;

class Proxy {
 public:
  Proxy(ProxyHandler handler, size_t fifoSize = Config::FIFO_BUFFER_SIZE);
  ~Proxy();

  void start();
  void stop();
  Fifo &fifo();

 private:
  struct Impl;
  ::std::unique_ptr<Impl> pimpl;
};

}  // namespace pccl