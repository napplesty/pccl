#pragma once

#include <netinet/in.h>
#include <sys/socket.h>

#include <bitset>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "config.h"

namespace pccl {

class Communicator;

std::string version();

enum class ChannelType : int8_t {
  NONE,
  MEMORY,
  PORT,
  NVLS,
  INNETWORK,
  ChannelTypeEnd,
};

enum class DeviceType : int8_t {
  HOST,
  CUDA,
  HIP,
  DeviceTypeEnd,
};

enum class BufferType : int8_t {
  LIB,
  HOST,
  DEVICE,
  TEMP,
  BufferTypeEnd,
};

enum class NetworkType : int8_t {
  OVS_FLOW,
  NetworkTypeEnd,
};

enum class DataType : int8_t {
  I8,
  I16,
  I32,
  I64,
  U8,
  U16,
  U32,
  U64,
  FP16,
  FP32,
  BF16,
  FP8_E4M3,
  FP8_E5M2,
  DataTypeEnd,
};

enum class ReduceOpType : int8_t {
  SUM,
  ReduceOpEnd,
};

enum class Transport : int8_t {
  Unknown,
  HostIpc,
  CudaIpc,
  IB,
  Ethernet,
  NVLS,
  TransportEnd,
};

enum class OperationType : int16_t {
  NOP,
  BARRIER,
  PUT,
  GET,
  SIGNAL,
  WAIT,
  FLUSH,
  REDUCE,                  // locally
  REDUCE_WRITE,            // reduce-copy-to-remote
  READ_REDUCE,             // reduce-without-copy-to-remote
  MULTI_READ_REDUCE_STORE, // atomic-reduce-and-wait-signal-copy-to-local
  NETCONF,                 // network configuration
  OperationTypeEnd,
};

enum class PacketType {
  Simple,
  LL16,
  PacketTypeEnd,
};

constexpr int TransportFlagsSize = static_cast<int>(Transport::TransportEnd);
using TransportFlagsBase = std::bitset<TransportFlagsSize>;

class TransportFlags : private TransportFlagsBase {
public:
  TransportFlags() = default;
  TransportFlags(Transport transport);
  bool has(Transport transport) const;
  bool none() const;
  bool any() const;
  bool all() const;
  size_t count() const;
  TransportFlags &operator|=(TransportFlags other);
  TransportFlags operator|(TransportFlags other) const;
  TransportFlags operator|(Transport transport) const;
  TransportFlags &operator&=(TransportFlags other);
  TransportFlags operator&(TransportFlags other) const;
  TransportFlags operator&(Transport transport) const;
  TransportFlags &operator^=(TransportFlags other);
  TransportFlags operator^(TransportFlags other) const;
  TransportFlags operator^(Transport transport) const;
  TransportFlags operator~() const;
  bool operator==(TransportFlags other) const;
  bool operator!=(TransportFlags other) const;
  TransportFlagsBase toBitset() const;
  void set(size_t pos, bool value = true) {
    TransportFlagsBase::set(pos, value);
  }
  bool test(size_t pos) const { return TransportFlagsBase::test(pos); }

  static TransportFlags fromString(const std::string &s);

private:
  TransportFlags(TransportFlagsBase bitset);
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
  ~RegisteredMemory() = default;
  int rankOf() const;
  void *hostPtr() const;
  void *devicePtr() const;
  size_t size() const;
  BufferType type() const;
  TransportFlags transports() const;
  uint64_t tag() const;

  std::vector<char> serialize() const;
  static RegisteredMemory deserialize(const std::vector<char> &data);

private:
  struct Impl;
  RegisteredMemory(std::shared_ptr<Impl> pimpl);
  std::shared_ptr<Impl> pimpl_;
  friend class MemoryContext;
  friend class Connection;
};

class MemoryContext {
public:
  ~MemoryContext();
  RegisteredMemory getPredefinedMemory(BufferType type);
  RegisteredMemory allocateWorkSpace(size_t size, bool isHostMemory, int tag);
  RegisteredMemory registerAsWorkSpace(void *buffer, bool isHostMemory, int tag,
                                       Transport transport);
  void unregister(RegisteredMemory memory);
  std::vector<RegisteredMemory> waitWorkSpaceReady(std::vector<int> &ranks,
                                                   int tag);

private:
  friend class Communicator;
  MemoryContext(std::shared_ptr<Communicator> communicator);
  class Impl;
  std::unique_ptr<Impl> pimpl_;
};

class Endpoint {
public:
  int rank() const;
  Transport transport() const;
  int maxWriteQueueSize() const;
  int maxCompleteQueueSize() const;
  uint64_t latency() const;
  uint64_t bandwidth() const;
  std::vector<char> serialize() const;
  static Endpoint deserialize(const std::vector<char> &data);

private:
  friend class Communicator;
  Endpoint(std::shared_ptr<Communicator> communicator);
  struct Impl;
  Endpoint(std::shared_ptr<Impl> pimpl);
  std::shared_ptr<Impl> pimpl_;
  friend class Context;
  friend class Connection;
};

class Connection {
public:
  Connection() = default;
  virtual ~Connection() = default;
  virtual void write(RegisteredMemory dst, uint64_t dstOffset,
                     RegisteredMemory src, uint64_t srcOffset,
                     uint64_t size) = 0;
  virtual void updateAndSync(RegisteredMemory dst, uint64_t dstOffset,
                             RegisteredMemory src, uint64_t srcOffset,
                             uint64_t size) = 0;
  virtual void flush(int64_t timeoutUsec = 3e9) = 0;
  virtual Transport transport() = 0;
  virtual Transport remoteTransport() = 0;
  virtual int remoteRank() = 0;
  virtual bool connected() = 0;
  virtual uint64_t bandwidth() = 0;
  virtual uint64_t latency() = 0;
};

struct NetConfEntry {
  union {
    struct {
      struct in_addr ip;
      uint16_t port;
    } v4;
    struct {
      struct in6_addr ip;
      uint16_t port;
    } v6;
  };
  sa_family_t family;
};

class Device {
public:
  Device(int id, int rank,
         std::vector<std::tuple<TransportFlags, NetConfEntry>> endpoint_infos);
  NetConfEntry getConfEntry(TransportFlags transport);
  int uid() const;

private:
  struct Impl;
  std::unique_ptr<Impl> pimpl_;
};

class Switch {
public:
  Switch(int id, const std::string &name, NetConfEntry network_address);
  NetConfEntry getConfEntry() const;
  int uid() const;

private:
  struct Impl;
  std::unique_ptr<Impl> pimpl_;
};

class OpticalSwitch {
public:
  OpticalSwitch(int id, const std::string &name, NetConfEntry network_address);
  NetConfEntry getConfEntry() const;
  int uid() const;

private:
  struct Impl;
  std::unique_ptr<Impl> pimpl_;
};

struct NetConfConnection {
  int src, dst;
  TransportFlags transport;
};

class ClusterContext {
public:
  ClusterContext();
  ~ClusterContext();
  NetConfEntry getConfEntry() const;
  void registerPhase(int phase, std::vector<NetConfConnection> &connections);
  int getPhase() const;
  void registerPhaseTransform(
      int prevPhase, int nextPhase,
      std::vector<std::tuple<int, NetConfConnection, std::string>> &commands);
  void preCommit(int nextPhase);
  void commit();

private:
  class Impl;
  std::unique_ptr<Impl> pimpl_;
};

class ConnectionContext {
public:
  ~ConnectionContext();
  void registerEndpoint(int rank, Endpoint endpoint);
  Endpoint getEndpoint(int rank);
  TransportFlags getChannelTypes(std::shared_ptr<Connection> connection);
  int remoteRankOf(std::shared_ptr<Connection> connection);
  std::shared_ptr<Connection> getConnection(Endpoint localEndpoint,
                                            Endpoint remoteEndpoint,
                                            TransportFlags transport);
  void connect(Endpoint localEndpoint, Endpoint remoteEndpoint);
  void disconnect(Endpoint localEndpoint, Endpoint remoteEndpoint);
  bool isConnected(Endpoint localEndpoint, Endpoint remoteEndpoint);
  bool notifyOperator(RegisteredMemory mem, std::vector<Endpoint> &endpoints);
  TransportFlags getTransportFlags(Endpoint localEndpoint,
                                   Endpoint remoteEndpoint);
  static TransportFlags getAvailableTransports();

  std::shared_ptr<ClusterContext> clusterContext();

private:
  friend class Communicator;
  ConnectionContext(std::shared_ptr<Communicator> communicator);
  struct Impl;
  std::unique_ptr<Impl> pimpl_;
  friend class RegisteredMemory;
  friend class Endpoint;
};

struct Event {
  std::function<void()> flush;
  std::function<void()> wait;
  std::function<void()> record;
};

struct Operation {
  OperationType type;
  int8_t num_op;
  union {
    struct {
      uint32_t connections[Config::MAX_CHANNEL_PER_OPERATION];
      uint32_t local_buffer_slices[Config::MAX_CHANNEL_PER_OPERATION];
      uint32_t remote_buffer_slices[Config::MAX_CHANNEL_PER_OPERATION];
    } putget;
    struct {
      uint32_t local_buffer_slices[Config::MAX_CHANNEL_PER_OPERATION];
      uint32_t peer_buffer_slices[Config::MAX_CHANNEL_PER_OPERATION];
      ReduceOpType reduce_op;
    } memop;
    struct {
      uint32_t src_buffer_slices[Config::MAX_CHANNEL_PER_OPERATION];
      uint32_t dst_buffer_slices[Config::MAX_CHANNEL_PER_OPERATION];
      ReduceOpType reduce_op;
    } reduceop;
  } meta;
};

class Capsule {
public:
  Capsule(int id, std::string &path);
  std::string name() const;
  std::string collective() const;

private:
  struct Impl;
  std::unique_ptr<Impl> pimpl_;
};

class Operator {
public:
  ~Operator();
  std::string name() const;
  std::string collective() const;

  bool isInplace() const;
  bool isConfigurable() const;
  Event execute(int rank, void *input, void *output, DataType dtype,
                size_t inputSize, size_t outputSize, Event &event, bool flush,
                uint64_t tag);

public:
  Operator(std::string &path, std::shared_ptr<Communicator> communicator);
  class Impl;
  std::unique_ptr<Impl> impl_;
};

class Communicator : public std::enable_shared_from_this<Communicator> {
public:
  Communicator();
  ~Communicator();

  std::shared_ptr<Endpoint> getEndpoint();
  void registerRemoteInfos(std::shared_ptr<Endpoint> endpoint, int remoteRank);

  std::shared_ptr<MemoryContext> memoryContext();
  std::shared_ptr<ConnectionContext> connectionContext();
  std::shared_ptr<ClusterContext> clusterContext();

  std::shared_ptr<Operator> registerOperator(const std::string &operator_path);

private:
  class Impl;
  std::unique_ptr<Impl> pimpl_;
};

} // namespace pccl

namespace std {

template <> struct hash<pccl::TransportFlags> {
  size_t operator()(const pccl::TransportFlags &flags) const {
    return flags.toBitset().to_ullong();
  }
};

} // namespace std
