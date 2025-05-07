#pragma once

#include <memory>
#include <unordered_map>
#include <vector>

#include "device.h"
#include "plugin/ib.h"
#include "runtime.h"

namespace pccl {

struct ConnectionContext::Impl {
  ::std::vector<::std::shared_ptr<Connection>> connections_;
  ::std::unordered_map<Transport, ::std::unique_ptr<IbCtx>> ib_ctxs_;
  ::std::shared_ptr<CudaStreamWithFlags> ipcStream_;
  CUmemGenericAllocationHandle mcHandle_;
  Impl();
  IbCtx* getIbContext(Transport ibTransport);
};

}  // namespace pccl