#pragma once

#include <memory>
#include <unordered_map>
#include <vector>

#include "device.h"
#include "plugin/ib.h"
#include "plugin/sock.h"
#include "runtime.h"

namespace pccl {

struct ConnectionContext::Impl {
  int active_phase;
  std::shared_ptr<IbCtx> ib_ctx_0;
  std::shared_ptr<IbCtx> ib_ctx_1;
  std::shared_ptr<SockCtx> sock_ctx_0;
  std::shared_ptr<SockCtx> sock_ctx_1;
  std::shared_ptr<CudaStreamWithFlags> ipcStream_;
  CUmemGenericAllocationHandle mcHandle_;
  Impl();
  void flip();
  std::shared_ptr<IbCtx> getIbContext(Transport ibTransport, bool active);
  std::shared_ptr<IbCtx> getSockContext(Transport sockTransport, bool active);
};

} // namespace pccl