#pragma once

#include <bitset>
#include <cstdint>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "config.h"

namespace pccl {

::std::string version();

enum class ChannelType : int {
  NONE,
  MEMORY,
  PORT,
  NVLS,
  INNETWORK,
  ChannelTypeEnd,
};

enum class DeviceType : int {
  HOST,
  CUDA,
  HIP,
  DeviceTypeEnd,
};

enum class NetworkType : int {
  OVS_FLOW,
  NetworkTypeEnd,
};

enum class DataType : int {
  INT32,
  UINT32,
  FLOAT16,
  FLOAT32,
  BFLOAT16,
  DataTypeEnd,
};

enum class ReduceOpType : int {
  SUM,
  ReduceOpEnd,
};

enum class Transport : int {
  Unknown,
  CudaIpc,
  IB0,
  IB1,
  Ethernet,
  TransportEnd,
};

enum class OperationType : int {
  NOP,
  BARRIER,
  PUT,
  GET,
  COPY,
  SIGNAL,
  WAIT,
  FLUSH,
  REDUCE,
  REDUCE_WRITE,
  READ_REDUCE,
  READ_REDUCE_WRITE,
  MULTI_READ_REDUCE_STORE,
  NET_CONFIGURE,
  OperationTypeEnd,
};

enum class PacketType {
  LL16,
  Simple,
  PacketTypeEnd,
};

const ::std::string TransportNames[] = {"UNK", "IPC",     "IB0", "IB1",
                                        "ETH", "Netconf", "NUM"};

namespace detail {

constexpr int TransportFlagsSize = static_cast<int>(Transport::TransportEnd);
using TransportFlagsBase = ::std::bitset<TransportFlagsSize>;

}  // namespace detail

class TransportFlags : private detail::TransportFlagsBase {
 public:
  TransportFlags() = default;
  TransportFlags(Transport transport);
  bool has(Transport transport) const;
  bool none() const;
  bool any() const;
  bool all() const;
  size_t count() const;
  TransportFlags& operator|=(TransportFlags other);
  TransportFlags operator|(TransportFlags other) const;
  TransportFlags operator|(Transport transport) const;
  TransportFlags& operator&=(TransportFlags other);
  TransportFlags operator&(TransportFlags other) const;
  TransportFlags operator&(Transport transport) const;
  TransportFlags& operator^=(TransportFlags other);
  TransportFlags operator^(TransportFlags other) const;
  TransportFlags operator^(Transport transport) const;
  TransportFlags operator~() const;
  bool operator==(TransportFlags other) const;
  bool operator!=(TransportFlags other) const;
  detail::TransportFlagsBase toBitset() const;
  void set(size_t pos, bool value = true) {
    detail::TransportFlagsBase::set(pos, value);
  }
  bool test(size_t pos) const { return detail::TransportFlagsBase::test(pos); }

 private:
  TransportFlags(detail::TransportFlagsBase bitset);
};

inline TransportFlags operator|(Transport transport1, Transport transport2) {
  return TransportFlags(transport1) | transport2;
}
inline TransportFlags operator&(Transport transport1, Transport transport2) {
  return TransportFlags(transport1) & transport2;
}
inline TransportFlags operator^(Transport transport1, Transport transport2) {
  return TransportFlags(transport1) ^ transport2;
}

class Connection;

class RegisteredMemory {
 public:
  RegisteredMemory(int global_buffer_id);
  ~RegisteredMemory();
  void* data() const;
  void* originalDataPtr() const;
  size_t size();
  TransportFlags transports();
  ::std::vector<char> serialize();
  static RegisteredMemory deserialize(const ::std::vector<char>& data);

 private:
  struct Impl;
  RegisteredMemory(::std::shared_ptr<Impl> pimpl);
  ::std::shared_ptr<Impl> pimpl_;
  friend class Context;
  friend class Connection;
};

class MemoryContext {
 public:
  MemoryContext();
  ~MemoryContext();
  std::vector<RegisteredMemory> prepareMemoryForOperator(int op_id,
                                                         int num_chunks);

 private:
  struct Impl;
  ::std::shared_ptr<Impl> pimpl_;
};

class Endpoint {
 public:
  Endpoint(Transport transport);
  int rank() const;
  Transport transport();
  int maxWriteQueueSize();
  ::std::vector<char> serialize();
  static Endpoint deserialize(const ::std::vector<char>& data);

 private:
  struct Impl;
  Endpoint(::std::shared_ptr<Impl> pimpl);
  ::std::shared_ptr<Impl> pimpl_;
  friend class Context;
  friend class Connection;
};

class Connection {
 public:
  Connection() = default;
  virtual ~Connection() = default;
  virtual ::std::vector<RegisteredMemory> prepareOperationAndGetGlobalBufferIds(
      int op_id, int num_chunks, std::vector<RegisteredMemory>& dst_mems) = 0;
  virtual void write(RegisteredMemory dst, uint64_t dstOffset,
                     RegisteredMemory src, uint64_t srcOffset,
                     uint64_t size) = 0;
  virtual void updateAndSync(RegisteredMemory dst, uint64_t dstOffset,
                             uint64_t* src, uint64_t newValue) = 0;
  virtual void flush(int64_t timeoutUsec = 3e7) = 0;
  virtual Transport transport() = 0;
  virtual Transport remoteTransport() = 0;
  virtual bool idle() = 0;
  virtual bool connected() = 0;
  virtual uint64_t bandwidth() = 0;
  virtual uint64_t latency() = 0;
  ::std::string getTransportName();

 protected:
  static ::std::shared_ptr<RegisteredMemory::Impl> getImpl(
      RegisteredMemory& memory);
  static ::std::shared_ptr<Endpoint::Impl> getImpl(Endpoint& memory);
};

class ConnectionContext {
 public:
  ConnectionContext();
  ~ConnectionContext();
  RegisteredMemory getRegisteredMemory(int bufferId);
  Endpoint createEndpoint(TransportFlags transport);
  ::std::shared_ptr<Connection> getConnection(Endpoint localEndpoint,
                                              Endpoint remoteEndpoint);
  void connect(Endpoint localEndpoint, Endpoint remoteEndpoint);
  void disconnect(Endpoint localEndpoint, Endpoint remoteEndpoint);
  bool isConnected(Endpoint localEndpoint, Endpoint remoteEndpoint);

  ::std::shared_ptr<Connection> getConnection(int localRank, int remoteRank,
                                              TransportFlags transport);
  bool isConnected(int localRank, int remoteRank, TransportFlags transport);
  bool disconnect(int localRank, int remoteRank, TransportFlags transport);

 private:
  struct Impl;
  ::std::unique_ptr<Impl> pimpl_;
  friend class RegisteredMemory;
  friend class Endpoint;
};

struct NetworkAddress {
  char ip_address[32];
  uint16_t port;
  bool is_ipv6;
};

class Device {
 public:
  Device(int id, int rank,
         ::std::vector<::std::tuple<TransportFlags, NetworkAddress>>
             endpoint_infos);
  NetworkAddress getNetworkAddress(TransportFlags transport);

 private:
  struct Impl;
  ::std::unique_ptr<Impl> pimpl_;
};

class Switch {
 public:
  Switch(int id, const ::std::string& name);
  void configureStart();
  void configureEnd();
  void changeRoute(TransportFlags transport, int src, int dst,
                   const ::std::vector<int>& outPorts);
  ::std::vector<int> getRoute(TransportFlags transport, int src, int dst);

 private:
  struct Impl;
  ::std::unique_ptr<Impl> pimpl_;
};

class OpticalSwitch {
 public:
  OpticalSwitch(int id, const ::std::string& name);
  void configureStart();
  void configureEnd();
  void connectInternalConnect(int internalPort0, int internalPort1);
  void breakInternalConnect(int internalPort0, int internalPort1);
  ::std::vector<::std::tuple<int, int>> getInternalConnects();

 private:
  struct Impl;
  ::std::unique_ptr<Impl> pimpl_;
};

class Communicator;

class Cluster {
 public:
  Cluster(::std::string& topoFile, NetworkType networkType,
          ::std::shared_ptr<Communicator> communicator);
  ::std::vector<char> getConnectableView();
  void registerRoutePhase(int routePhase, ::std::string& routePhaseFile);
  void setRoutePhase(int routePhase);
  void registerTopologyPhase(int topologyPhase,
                             ::std::string& topologyPhaseFile);
  void setTopologyPhase(int topologyPhase);

  ::std::vector<char> getDeviceConnectivity(int routePhase, int topologyPhase);

 private:
  struct Impl;
  ::std::unique_ptr<Impl> pimpl_;
};

class Operator;

class Communicator {
 public:
  Communicator();
  ~Communicator();
  void registerRemoteEndpoint(::std::vector<char>& data, int remoteRank,
                              int bufferId);
  ::std::vector<char> getRemoteEndpoint(int rank, int bufferId);
  ::std::shared_ptr<MemoryContext> memoryContext();
  ::std::shared_ptr<ConnectionContext> connectionContext();
  ::std::shared_ptr<Cluster> cluster();
  RegisteredMemory buffer(int globalBufferId);

 protected:
  friend class Cluster;
  void beginClusterPhase(int topologyPhase, int routePhase);
  void endClusterPhase();

 protected:
  friend class Operator;
  void registerExecution(Operator* op);
  void unregisterExecution(Operator* op);

 private:
  struct Impl;
  ::std::unique_ptr<Impl> pimpl_;
};

class Operator {
 public:
  Operator(::std::shared_ptr<Communicator> communicator, ::std::string& path);
  ::std::string name() const;
  ::std::string collective() const;

  bool isInplace() const;
  bool isConfigurable() const;
  void execute(int rank, void* input, void* output, DataType dtype,
               size_t inputSize, size_t outputSize, void* stream = nullptr);

 public:
  struct Impl;
  ::std::unique_ptr<Impl> impl_;
};

}  // namespace pccl

namespace std {

template <>
struct hash<pccl::TransportFlags> {
  size_t operator()(const pccl::TransportFlags& flags) const {
    return flags.toBitset().to_ullong();
  }
};

}  // namespace std
