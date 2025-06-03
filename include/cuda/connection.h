#pragma once

#include "cuda/registered_memory.h"
#include "plugin/ib.h"
#include "plugin/sock.h"
#include "runtime.h"

namespace pccl {

// class NvlsConnection : public Connection {
// public:
//   NvlsConnection(Endpoint remote, Endpoint local);
//   ~NvlsConnection();

//   void write(RegisteredMemory dst, uint64_t dstOffset, RegisteredMemory src,
//              uint64_t srcOffset, uint64_t size) override;
//   void updateAndSync(RegisteredMemory dst, uint64_t dstOffset,
//                      RegisteredMemory src, uint64_t srcOffset,
//                      uint64_t size) override;
//   void flush(int64_t timeoutUsec = 3e9) override;
//   Transport transport() override;
//   Transport remoteTransport() override;
//   int remoteRank() override;
//   bool connected() override;
//   uint64_t bandwidth() override;
//   uint64_t latency() override;
// };

class CudaIpcConnection : public Connection {
public:
  CudaIpcConnection(Endpoint remote, Endpoint local);
  ~CudaIpcConnection();

  void switchBackend(void *backend) override;
  void write(RegisteredMemory dst, uint64_t dstOffset, RegisteredMemory src,
             uint64_t srcOffset, uint64_t size) override;
  void updateAndSync(RegisteredMemory dst, uint64_t dstOffset,
                     RegisteredMemory src, uint64_t srcOffset,
                     uint64_t size) override;
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

  void switchBackend(void *backend) override;
  void write(RegisteredMemory dst, uint64_t dstOffset, RegisteredMemory src,
             uint64_t srcOffset, uint64_t size) override;
  void updateAndSync(RegisteredMemory dst, uint64_t dstOffset,
                     RegisteredMemory src, uint64_t srcOffset,
                     uint64_t size) override;
  void flush(int64_t timeoutUsec = 3e9) override;
  Transport transport() override;
  Transport remoteTransport() override;
  int remoteRank() override;
  bool connected() override;
  uint64_t bandwidth() override; // bytes/sec
  uint64_t latency() override;   // usec

private:
  IbCtx *ctx_;
  IbQp *qp_;
  int remoteRank_;
};

class SockConnection : public Connection {
public:
  SockConnection(TransportInfo info, int remoteRank);
  ~SockConnection();

  void switchBackend(void *backend) override;
  void write(RegisteredMemory dst, uint64_t dstOffset, RegisteredMemory src,
             uint64_t srcOffset, uint64_t size) override;
  void updateAndSync(RegisteredMemory dst, uint64_t dstOffset,
                     RegisteredMemory src, uint64_t srcOffset,
                     uint64_t size) override;
  void flush(int64_t timeoutUsec = 3e9) override;
  Transport transport() override;
  Transport remoteTransport() override;
  int remoteRank() override;
  bool connected() override;
  uint64_t bandwidth() override;
  uint64_t latency() override;

private:
  SockCtx *ctx_;
  SockQp *qp_;
  int remoteRank_;
};

} // namespace pccl