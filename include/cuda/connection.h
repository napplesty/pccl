#pragma once
#pragma once

#include "cuda/registered_memory.h"
#include "plugin/ib.h"
#include "plugin/sock.h"
#include "runtime.h"

namespace pccl {

class CudaIpcConnection : public Connection {
 public:
  CudaIpcConnection(TransportInfo info, int remoteRank);
  ~CudaIpcConnection();

  void flip() override;
  void write(RegisteredMemory dst, uint64_t dstOffset, RegisteredMemory src, uint64_t srcOffset,
             uint64_t size) override;
  void updateAndSync(RegisteredMemory dst, uint64_t dstOffset, RegisteredMemory src,
                     uint64_t srcOffset, uint64_t size) override;
  void flush(int64_t timeoutUsec = 3e9) override;
  Transport transport() override;
  Transport remoteTransport() override;
  int remoteRank() override;
  bool connected() override;
  uint64_t bandwidth() override;
  uint64_t latency() override;

 private:
  TransportInfo info_;
  int remoteRank_;
};

class IbConnection : public Connection {
 public:
  IbConnection(TransportInfo info, int remoteRank);
  ~IbConnection();

  void flip() override;
  void write(RegisteredMemory dst, uint64_t dstOffset, RegisteredMemory src, uint64_t srcOffset,
             uint64_t size) override;
  void updateAndSync(RegisteredMemory dst, uint64_t dstOffset, RegisteredMemory src,
                     uint64_t srcOffset, uint64_t size) override;
  void flush(int64_t timeoutUsec = 3e9) override;
  Transport transport() override;
  Transport remoteTransport() override;
  int remoteRank() override;
  bool connected() override;
  uint64_t bandwidth() override;
  uint64_t latency() override;

 private:
  std::shared_ptr<IbCtx> ctx_;
  std::shared_ptr<IbQp> qp_;
  int remoteRank_;
};

class SockConnection : public Connection {
 public:
  SockConnection(TransportInfo info, int remoteRank);
  ~SockConnection();

  void flip() override;
  void write(RegisteredMemory dst, uint64_t dstOffset, RegisteredMemory src, uint64_t srcOffset,
             uint64_t size) override;
  void updateAndSync(RegisteredMemory dst, uint64_t dstOffset, RegisteredMemory src,
                     uint64_t srcOffset, uint64_t size) override;
  void flush(int64_t timeoutUsec = 3e9) override;
  Transport transport() override;
  Transport remoteTransport() override;
  int remoteRank() override;
  bool connected() override;
  uint64_t bandwidth() override;
  uint64_t latency() override;

 private:
  std::shared_ptr<SockCtx> ctx_;
  std::shared_ptr<SockQp> qp_;
  int remoteRank_;
};

}  // namespace pccl