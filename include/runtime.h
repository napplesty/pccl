#pragma once

#include <netinet/in.h>
#include <sys/socket.h>

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
  IB,
  Ethernet,
  NVLS,
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
  NETCONF,
  CHANNEL_FILP,
  OperationTypeEnd,
};

enum class PacketType {
  LL16,
  Simple,
  PacketTypeEnd,
};

const ::std::string TransportNames[] = {"UNK", "CUDA_IPC", "IB", "ETH", "NUM"};

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
  void set(size_t pos, bool value = true) { detail::TransportFlagsBase::set(pos, value); }
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

class RegisteredMemory {
 public:
  RegisteredMemory(uint64_t bufferOffset, bool isHostMemory);
  ~RegisteredMemory();
  int rankOf() const;
  void* data() const;
  void* hostPtr() const;
  void* devicePtr() const;
  size_t size();
  TransportFlags transports();
  ::std::vector<char> serialize();
  static RegisteredMemory deserialize(const ::std::vector<char>& data);

 private:
  struct Impl;
  RegisteredMemory(::std::shared_ptr<Impl> pimpl);
  ::std::shared_ptr<Impl> pimpl_;
  friend class MemoryContext;
  friend class Connection;
};

class MemoryContext {
 public:
  MemoryContext();
  ~MemoryContext();
  void registerMemory(RegisteredMemory memory);
  RegisteredMemory getLibMemory();
  RegisteredMemory allocateWorkspace(size_t size, bool isHostMemory, int tag);
  RegisteredMemory getRemoteLibMemory(int rank);
  RegisteredMemory getRemoteWorkspace(int tag, int rank);
  void freeWorkspace(RegisteredMemory memory);

  ::std::vector<char> serialize();

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
  virtual void flip() = 0;
  virtual void write(RegisteredMemory dst, uint64_t dstOffset, RegisteredMemory src,
                     uint64_t srcOffset, uint64_t size) = 0;
  virtual void updateAndSync(RegisteredMemory dst, uint64_t dstOffset, RegisteredMemory src,
                             uint64_t srcOffset, uint64_t size) = 0;
  virtual void flush(int64_t timeoutUsec = 3e9) = 0;
  virtual Transport transport() = 0;
  virtual Transport remoteTransport() = 0;
  virtual int remoteRank() = 0;
  virtual bool connected() = 0;
  virtual uint64_t bandwidth() = 0;
  virtual uint64_t latency() = 0;
};

class ConnectionContext {
 public:
  ConnectionContext();
  ~ConnectionContext();
  TransportFlags getChannelTypes(::std::shared_ptr<Connection> connection);
  int remoteRankOf(::std::shared_ptr<Connection> connection);
  ::std::shared_ptr<Connection> getConnection(Endpoint localEndpoint, Endpoint remoteEndpoint);
  void connect(Endpoint localEndpoint, Endpoint remoteEndpoint);
  void disconnect(Endpoint localEndpoint, Endpoint remoteEndpoint);
  bool isConnected(Endpoint localEndpoint, Endpoint remoteEndpoint);

 private:
  struct Impl;
  ::std::unique_ptr<Impl> pimpl_;
  friend class RegisteredMemory;
  friend class Endpoint;
};

struct NetConfEntry {
  sa_family_t family;
  union {
    struct {
      uint32_t ip;
      uint16_t port;
    } v4;
    struct {
      struct in6_addr ip;
      uint16_t port;
    } v6;
  };
};

struct NetConfConnection {
  int src, dst;
  TransportFlags transport;
  bool connected;
};

class Device {
 public:
  Device(int id, int rank,
         ::std::vector<::std::tuple<TransportFlags, NetConfEntry>> endpoint_infos);
  NetConfEntry getConfEntry(TransportFlags transport);

 private:
  struct Impl;
  ::std::unique_ptr<Impl> pimpl_;
};

class Switch {
 public:
  Switch(int id, const ::std::string& name, NetConfEntry network_address);
  NetConfEntry getConfEntry() const;
  void command(const ::std::string& command);
  void commit();

 private:
  struct Impl;
  ::std::unique_ptr<Impl> pimpl_;
};

class OpticalSwitch {
 public:
  OpticalSwitch(int id, const ::std::string& name, NetConfEntry network_address);
  NetConfEntry getConfEntry() const;
  void command(const ::std::string& command);
  void commit();

 private:
  struct Impl;
  ::std::unique_ptr<Impl> pimpl_;
};

class Cluster {
 public:
  Cluster(::std::string& topoFile, NetworkType networkType);
  NetConfEntry getConfEntry() const;
  void registerPhase(int phase, ::std::vector<NetConfConnection>& connections);
  int getPhase() const;
  void registerPhaseTransform(
      int prevPhase, int nextPhase,
      ::std::vector<::std::tuple<NetConfConnection, ::std::string>>& commands);
  void preCommit(int nextPhase);
  void commit();

 private:
  struct Impl;
  ::std::unique_ptr<Impl> pimpl_;
};

class Communicator {
 public:
  Communicator();
  ~Communicator();
  ::std::vector<char> serialize();
  void registerRemoteInfos(::std::vector<char>& data, int remoteRank);
  ::std::vector<RegisteredMemory> getOperatorSpace(size_t bufferSize, int tag,
                                                   ::std::vector<int>& ranks);
  ::std::vector<Connection> getConnections(int tag, ::std::vector<int>& ranks);
  void switchPhase(int phase);
  ::std::shared_ptr<MemoryContext> memoryContext();
  ::std::shared_ptr<ConnectionContext> connectionContext();
  ::std::shared_ptr<Cluster> cluster();

 private:
  struct Impl;
  ::std::unique_ptr<Impl> pimpl_;
};

class Capsule {
 public:
  Capsule(int id, ::std::string& path);
  ::std::string name() const;
  ::std::string collective() const;

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
  void execute(int rank, void* input, void* output, DataType dtype, size_t inputSize,
               size_t outputSize, void* stream = nullptr);

 public:
  struct Impl;
  ::std::unique_ptr<Impl> impl_;
};

template <class T>
using DeviceHandle = typename T::DeviceHandle;

template <typename T>
DeviceHandle<::std::remove_reference_t<T>> deviceHandle(T&& t) {
  return t.deviceHandle();
}

template <class T>
using PacketPayload = typename T::Payload;

}  // namespace pccl

namespace std {

template <>
struct hash<pccl::TransportFlags> {
  size_t operator()(const pccl::TransportFlags& flags) const {
    return flags.toBitset().to_ullong();
  }
};

}  // namespace std
