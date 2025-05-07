#pragma once

#include "device.h"
#include "plugin/ib.h"
#include "runtime.h"

namespace pccl {

struct TransportInfo {
  Transport transport;
  bool ibLocal;
  union {
    // CUDA IPC
    struct {
      cudaIpcMemHandle_t cudaIpcBaseHandle;
      size_t cudaIpcOffsetFromBase;
    };
    // IB
    struct {
      const IbMr* ibMr;
      IbMrInfo ibMrInfo;
    };
    // Dma
    struct {
      union {
        char shareableHandle[64];
        struct {
          pid_t rootPid;
          int fileDesc;
        };
      };
      size_t offsetFromBase;
    };
  };
};

struct RegisteredMemory::Impl {
  void* data;

  void* originalDataPtr;
  size_t size;
  uint64_t hostHash;
  uint64_t pidHash;
  bool isCuMemMapAlloc;
  TransportFlags transports;
  ::std::vector<TransportInfo> transportInfos;

  int fileDesc = -1;

  Impl(void* data, size_t size, TransportFlags transports,
       ConnectionContext::Impl& contextImpl);  // for local
  Impl(const ::std::vector<char>& data);       // for remote
  ~Impl();

  const TransportInfo& getTransportInfo(Transport transport) const;
};

}  // namespace pccl