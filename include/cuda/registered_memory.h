#pragma once

#include "device.h"
#include "plugin/ib.h"
#include "plugin/sock.h"
#include "runtime.h"

namespace pccl {

struct TransportInfo {
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
    // Sock
    struct {
      SockMr* sockMr;
      SockMrInfo sockMrInfo;
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
  Transport transport;
  bool ibLocal;
};

struct RegisteredMemory::Impl {
  void* host_ptr;
  void* device_ptr;
  void* origianl_ptr;
  int original_rank;
  bool is_host_memory;
  bool is_lib_memory;
  size_t size;
  uint64_t hostHash;
  uint64_t pidHash;
  bool isCuMemMapAlloc;
  TransportFlags transports;
  ::std::vector<TransportInfo> transportInfos;

  int fileDesc = -1;

  Impl(bool isHostMemory, bool isLibMemory, TransportFlags transports,
       ConnectionContext::Impl& contextImpl);  // for local
  Impl(const ::std::vector<char>& data);       // for remote
  ~Impl();

  const TransportInfo& getTransportInfo(Transport transport) const;
};

}  // namespace pccl