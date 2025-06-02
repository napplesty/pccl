#include "runtime.h"
#include "component/lib_buffer.h"
#include "component/logging.h"
#include "config.h"
#include "device.h"

#include <map>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>

namespace pccl {

std::string version() { return "1.0.0"; }

TransportFlags::TransportFlags(Transport transport) {
  this->set((size_t)transport, true);
}

TransportFlags::TransportFlags(TransportFlagsBase bitset)
    : TransportFlagsBase(bitset) {}

bool TransportFlags::has(Transport transport) const {
  return test((size_t)transport);
}

bool TransportFlags::none() const { return TransportFlagsBase::none(); }

bool TransportFlags::any() const { return TransportFlagsBase::any(); }

bool TransportFlags::all() const { return TransportFlagsBase::all(); }

size_t TransportFlags::count() const { return TransportFlagsBase::count(); }

TransportFlags &TransportFlags::operator|=(TransportFlags other) {
  TransportFlagsBase::operator|=(other);
  return *this;
}

TransportFlags TransportFlags::operator|(TransportFlags other) const {
  TransportFlagsBase result = static_cast<TransportFlagsBase>(*this) |
                              static_cast<TransportFlagsBase>(other);
  return TransportFlags(result);
}

TransportFlags TransportFlags::operator|(Transport transport) const {
  TransportFlags result = *this;
  result.set((size_t)transport, true);
  return result;
}

TransportFlags &TransportFlags::operator&=(TransportFlags other) {
  TransportFlagsBase::operator&=(other);
  return *this;
}

TransportFlags TransportFlags::operator&(TransportFlags other) const {
  TransportFlagsBase result = static_cast<TransportFlagsBase>(*this) &
                              static_cast<TransportFlagsBase>(other);
  return TransportFlags(result);
}

TransportFlags TransportFlags::operator&(Transport transport) const {
  TransportFlags other(transport);
  return *this & other;
}

TransportFlags &TransportFlags::operator^=(TransportFlags other) {
  TransportFlagsBase::operator^=(other);
  return *this;
}

TransportFlags TransportFlags::operator^(TransportFlags other) const {
  TransportFlagsBase result = static_cast<TransportFlagsBase>(*this) ^
                              static_cast<TransportFlagsBase>(other);
  return TransportFlags(result);
}

TransportFlags TransportFlags::operator^(Transport transport) const {
  TransportFlags other(transport);
  return *this ^ other;
}

TransportFlags TransportFlags::operator~() const {
  return TransportFlags(TransportFlagsBase::operator~());
}

bool TransportFlags::operator==(TransportFlags other) const {
  return TransportFlagsBase::operator==(other);
}

bool TransportFlags::operator!=(TransportFlags other) const {
  return !(*this == other);
}

TransportFlagsBase TransportFlags::toBitset() const {
  return static_cast<TransportFlagsBase>(*this);
}

// Implementation for the new static method
TransportFlags TransportFlags::fromString(const std::string &s) {
  if (s.length() > TransportFlagsSize) {
    LOG_WARNING << "TransportFlags string for deserialization is too long: "
                << s.length() << ", max expected: " << TransportFlagsSize;
    return TransportFlags();
  }
  if (s.find_first_not_of("01") != std::string::npos) {
    LOG_WARNING << "TransportFlags string contains invalid characters: " << s;
    return TransportFlags();
  }
  return TransportFlags(TransportFlagsBase(s));
}

struct RegisteredMemory::Impl {
  int rank_;
  void *device_ptr_;
  void *host_ptr_;
  size_t size_;
  BufferType type_;
  uint64_t tag_;
  bool is_host_memory_;
  TransportFlags transports_;

  Impl(int rank, void *device_ptr, void *host_ptr, size_t size, BufferType type,
       uint64_t tag, bool is_host_memory, TransportFlags transports)
      : rank_(rank), device_ptr_(device_ptr), host_ptr_(host_ptr), size_(size),
        type_(type), tag_(tag), is_host_memory_(is_host_memory),
        transports_(transports) {}
};

RegisteredMemory::RegisteredMemory(std::shared_ptr<Impl> pimpl)
    : pimpl_(pimpl) {}

int RegisteredMemory::rankOf() const { return pimpl_ ? pimpl_->rank_ : 0; }

void *RegisteredMemory::hostPtr() const {
  return pimpl_ ? pimpl_->host_ptr_ : nullptr;
}

void *RegisteredMemory::devicePtr() const {
  return pimpl_ ? pimpl_->device_ptr_ : nullptr;
}

size_t RegisteredMemory::size() const { return pimpl_ ? pimpl_->size_ : 0; }

BufferType RegisteredMemory::type() const {
  return pimpl_ ? pimpl_->type_
                : BufferType::LIB; // Default to LIB if pimpl_ is null
}

uint64_t RegisteredMemory::tag() const {
  return pimpl_ ? pimpl_->tag_ : 0; // Default to 0 if pimpl_ is null
}

TransportFlags RegisteredMemory::transports() const {
  return pimpl_ ? pimpl_->transports_ : TransportFlags();
}

std::vector<char> RegisteredMemory::serialize() const {
  nlohmann::json j;
  if (pimpl_) {
    j["rank"] = pimpl_->rank_;
    j["isHostMemory"] = pimpl_->is_host_memory_;
    j["size"] = pimpl_->size_;
    j["type"] = static_cast<int>(pimpl_->type_);
    j["tag"] = pimpl_->tag_;
    j["transports"] = pimpl_->transports_.toBitset().to_string();
  }
  std::string s = j.dump();
  return std::vector<char>(s.begin(), s.end());
}

RegisteredMemory RegisteredMemory::deserialize(const std::vector<char> &data) {
  if (data.empty()) {
    LOG_WARNING << "RegisteredMemory::deserialize called with empty data.";
    return RegisteredMemory(nullptr);
  }

  std::string s(data.begin(), data.end());
  nlohmann::json j;
  try {
    j = nlohmann::json::parse(s);
  } catch (const nlohmann::json::parse_error &e) {
    LOG_ERROR << "Failed to parse RegisteredMemory JSON: " << e.what() << ".";
    return RegisteredMemory(nullptr);
  }

  if (!j.is_object() || j.empty()) {
    LOG_ERROR << "Deserialized JSON for RegisteredMemory is not a valid object "
                 "or is empty. JSON string: "
              << s;
    return RegisteredMemory(nullptr);
  }

  TransportFlags transports_val;
  std::string transports_str = j.value("transports", "");
  if (!transports_str.empty()) {
    transports_val = TransportFlags::fromString(transports_str);
  }

  auto impl_ptr = std::make_shared<RegisteredMemory::Impl>(
      j.value("rank", 0), nullptr, nullptr,
      j.value("size", static_cast<size_t>(0)),
      static_cast<BufferType>(
          j.value("type", static_cast<int>(BufferType::LIB))),
      j.value("tag", static_cast<uint64_t>(0)), j.value("isHostMemory", false),
      transports_val);

  return RegisteredMemory(impl_ptr);
}

class MemoryContext::Impl {
public:
  Impl() {
    auto lib_buf = GpuBuffer<char>(Config::LIB_BUFFER_SIZE, true);
    auto host_buf = GpuBuffer<char>(Config::HOST_BUFFER_SIZE, true);
    auto device_buf = GpuBuffer<char>(Config::DEVICE_BUFFER_SIZE, false);

    free_lib_func_ = lib_buf.free_func_;
    free_host_func_ = host_buf.free_func_;
    free_device_func_ = device_buf.free_func_;

    lib_buffer_ = std::make_unique<RegisteredMemory>(
        std::make_shared<RegisteredMemory::Impl>(
            getEnv()->rank, lib_buf.devicePtr(), lib_buf.hostPtr(),
            Config::LIB_BUFFER_SIZE, BufferType::LIB, 0, true,
            ConnectionContext::getAvailableTransports()));

    host_buffer_ = std::make_unique<RegisteredMemory>(
        std::make_shared<RegisteredMemory::Impl>(
            getEnv()->rank, host_buf.devicePtr(), host_buf.hostPtr(),
            Config::HOST_BUFFER_SIZE, BufferType::HOST, 0, true,
            ConnectionContext::getAvailableTransports()));

    device_buffer_ = std::make_unique<RegisteredMemory>(
        std::make_shared<RegisteredMemory::Impl>(
            getEnv()->rank, device_buf.devicePtr(), nullptr,
            Config::DEVICE_BUFFER_SIZE, BufferType::DEVICE, 0, false,
            ConnectionContext::getAvailableTransports()));

    lib = new (lib_buffer_->hostPtr()) LibBufferSlot();
  }

  ~Impl() {
    free_lib_func_(lib_buffer_->hostPtr());
    free_host_func_(host_buffer_->hostPtr());
    free_device_func_(device_buffer_->devicePtr());
  }

  std::optional<RegisteredMemory>
  tryAllocateMemory(size_t size, bool isHostMemory, int tag) {
    std::lock_guard<std::mutex> lock(mutex);
    static constexpr size_t granularity =
        Config::HOST_BUFFER_SIZE / Config::NUM_SLOT;
    size_t last_index = isHostMemory ? 0 : Config::NUM_SLOT;
    size_t max_index = last_index + Config::NUM_SLOT;
    for (; last_index < max_index &&
           lib->predefined_slot[last_index].status == SlotStatus::ALLOCATED;
         last_index++) {
    }

    if (last_index == max_index) {
      return std::nullopt;
    }

    size_t allocated_slot_count = (size + granularity - 1) / granularity;

    for (size_t next_index = last_index + 1; next_index <= max_index;
         next_index++) {
      if (lib->predefined_slot[next_index - 1].status ==
          SlotStatus::ALLOCATED) {
        last_index = next_index;
        continue;
      }
      if (next_index - last_index == allocated_slot_count) {
        break;
      }
    }

    if (last_index + allocated_slot_count > max_index) {
      return std::nullopt;
    }

    for (size_t i = 0; i < allocated_slot_count; i++) {
      lib->predefined_slot[last_index + i].status = SlotStatus::ALLOCATED;
      lib->predefined_slot[last_index + i].tag = tag;
    }

    auto impl_ptr = std::make_shared<RegisteredMemory::Impl>(
        getEnv()->rank,
        (char *)lib_buffer_->devicePtr() + last_index * granularity,
        (char *)lib_buffer_->hostPtr() + last_index * granularity, size,
        isHostMemory ? BufferType::HOST : BufferType::DEVICE, tag, isHostMemory,
        ConnectionContext::getAvailableTransports());

    return RegisteredMemory(impl_ptr);
  }

  bool registerMemory(RegisteredMemory memory) {
    std::lock_guard<std::mutex> lock(mutex);
    std::vector<char> mem_info = memory.serialize();
    for (size_t i = 0; i < Config::NUM_SLOT; i++) {
      if (lib->meta_slot[i].status == SlotStatus::FREE) {
        lib->meta_slot[i].status = SlotStatus::ALLOCATED;
        lib->meta_slot[i].tag = memory.tag();
        std::memcpy(lib->meta_slot[i].handle, mem_info.data(), mem_info.size());
        lib->meta_slot[i].transport = memory.transports();
        return true;
      }
    }
    return false;
  }

  void freeMemory(RegisteredMemory memory) {
    std::lock_guard<std::mutex> lock(mutex);
    size_t granularity = Config::HOST_BUFFER_SIZE / Config::NUM_SLOT;
    size_t index =
        (char *)memory.devicePtr() - (char *)lib_buffer_->devicePtr();
    size_t allocated_slot_count =
        (memory.size() + granularity - 1) / granularity;
  }

  RegisteredMemory getLibMemory() { return *lib_buffer_; }
  RegisteredMemory getHostMemory() { return *host_buffer_; }
  RegisteredMemory getDeviceMemory() { return *device_buffer_; }

private:
  std::function<void(void *)> free_lib_func_;
  std::function<void(void *)> free_host_func_;
  std::function<void(void *)> free_device_func_;
  std::unique_ptr<RegisteredMemory> lib_buffer_;
  std::unique_ptr<RegisteredMemory> host_buffer_;
  std::unique_ptr<RegisteredMemory> device_buffer_;
  LibBufferSlot *lib;
  std::mutex mutex;
};

MemoryContext::MemoryContext() : pimpl_(std::make_shared<Impl>()) {}

MemoryContext::~MemoryContext() {}

void MemoryContext::registerMemory(RegisteredMemory memory) {
  pimpl_->registerMemory(memory);
}

RegisteredMemory MemoryContext::getLibMemory() {
  return pimpl_->getLibMemory();
}

RegisteredMemory MemoryContext::allocateWorkspace(size_t size,
                                                  bool isHostMemory, int tag) {
  return pimpl_->allocateWorkspace(size, isHostMemory, tag);
}

std::vector<RegisteredMemory>
MemoryContext::waitWorkSpace(std::vector<int> &ranks, int tag) {
  return pimpl_->waitWorkSpace(ranks, tag);
}

RegisteredMemory MemoryContext::getRemoteLibMemory(int rank) {
  return pimpl_->getRemoteLibMemory(rank);
}

RegisteredMemory MemoryContext::getRemoteWorkspace(int tag, int rank) {
  return pimpl_->getRemoteWorkspace(tag, rank);
}

void MemoryContext::freeWorkspace(RegisteredMemory memory) {
  pimpl_->freeWorkspace(memory);
}

std::vector<char> MemoryContext::serialize() { return pimpl_->serialize(); }

// Endpoint::Impl 实现
struct Endpoint::Impl {
  int rank_;
  Transport transport_;
  int maxWriteQueueSize_;
  NetConfEntry networkAddress_;

  Impl() : rank_(0), transport_(Transport::Unknown), maxWriteQueueSize_(0) {}
};

Endpoint::Endpoint() : pimpl_(std::make_shared<Impl>()) {}

Endpoint::Endpoint(std::shared_ptr<Impl> pimpl) : pimpl_(pimpl) {}

int Endpoint::rank() const { return pimpl_ ? pimpl_->rank_ : 0; }

Transport Endpoint::transport() {
  return pimpl_ ? pimpl_->transport_ : Transport::Unknown;
}

int Endpoint::maxWriteQueueSize() {
  return pimpl_ ? pimpl_->maxWriteQueueSize_ : 0;
}

std::vector<char> Endpoint::serialize() const {
  nlohmann::json j;
  if (pimpl_) {
    j["rank"] = pimpl_->rank_;
    j["transport"] = static_cast<int>(pimpl_->transport_);
    j["maxWriteQueueSize"] = pimpl_->maxWriteQueueSize_;
  }
  std::string s = j.dump();
  return std::vector<char>(s.begin(), s.end());
}

Endpoint Endpoint::deserialize(const std::vector<char> &data) {
  std::string s(data.begin(), data.end());
  nlohmann::json j = nlohmann::json::parse(s);

  auto impl = std::make_shared<Impl>();
  impl->rank_ = j["rank"];
  impl->transport_ = static_cast<Transport>(j["transport"]);
  impl->maxWriteQueueSize_ = j["maxWriteQueueSize"];

  return Endpoint(impl);
}

// Device::Impl 实现
struct Device::Impl {
  int id_;
  int rank_;
  std::vector<std::tuple<TransportFlags, NetConfEntry>> endpoint_infos_;

  Impl(int id, int rank,
       std::vector<std::tuple<TransportFlags, NetConfEntry>> infos)
      : id_(id), rank_(rank), endpoint_infos_(infos) {}
};

Device::Device(
    int id, int rank,
    std::vector<std::tuple<TransportFlags, NetConfEntry>> endpoint_infos)
    : pimpl_(std::make_unique<Impl>(id, rank, endpoint_infos)) {}

NetConfEntry Device::getConfEntry(TransportFlags transport) {
  if (pimpl_) {
    for (const auto &[flags, entry] : pimpl_->endpoint_infos_) {
      if ((flags & transport).any()) {
        return entry;
      }
    }
  }
  return NetConfEntry{};
}

// Switch::Impl 实现
struct Switch::Impl {
  int id_;
  std::string name_;
  NetConfEntry network_address_;
  std::vector<std::string> pending_commands_;

  Impl(int id, const std::string &name, NetConfEntry addr)
      : id_(id), name_(name), network_address_(addr) {}
};

Switch::Switch(int id, const std::string &name, NetConfEntry network_address)
    : pimpl_(std::make_unique<Impl>(id, name, network_address)) {}

NetConfEntry Switch::getConfEntry() const {
  return pimpl_ ? pimpl_->network_address_ : NetConfEntry{};
}

void Switch::command(const std::string &command) {
  if (pimpl_) {
    pimpl_->pending_commands_.push_back(command);
  }
}

void Switch::commit() {
  if (pimpl_) {
    // 提交所有待执行的命令
    pimpl_->pending_commands_.clear();
  }
}

// OpticalSwitch::Impl 实现
struct OpticalSwitch::Impl {
  int id_;
  std::string name_;
  NetConfEntry network_address_;
  std::vector<std::string> pending_commands_;

  Impl(int id, const std::string &name, NetConfEntry addr)
      : id_(id), name_(name), network_address_(addr) {}
};

OpticalSwitch::OpticalSwitch(int id, const std::string &name,
                             NetConfEntry network_address)
    : pimpl_(std::make_unique<Impl>(id, name, network_address)) {}

NetConfEntry OpticalSwitch::getConfEntry() const {
  return pimpl_ ? pimpl_->network_address_ : NetConfEntry{};
}

void OpticalSwitch::command(const std::string &command) {
  if (pimpl_) {
    pimpl_->pending_commands_.push_back(command);
  }
}

void OpticalSwitch::commit() {
  if (pimpl_) {
    // 提交所有待执行的光路切换命令
    pimpl_->pending_commands_.clear();
  }
}

// ClusterContext::Impl 实现
class ClusterContext::Impl {
public:
  struct FlowTable {
    int src_rank;
    int dst_rank;
    int switch_id;
    bool is_optical;
  };

  Impl() : current_phase_(0) {}
  ~Impl() {}

  NetConfEntry getConfEntry() const { return cluster_controller_addr_; }

  void registerPhase(int phase, std::vector<NetConfConnection> &connections) {
    phase_connections_[phase] = connections;
  }

  int getPhase() const { return current_phase_; }

  void registerPhaseTransform(
      int prevPhase, int nextPhase,
      std::vector<std::tuple<NetConfConnection, std::string>> &commands) {
    phase_transforms_[{prevPhase, nextPhase}] = commands;
  }

  void preCommit(int nextPhase) {
    next_phase_ = nextPhase;
    // 准备阶段转换配置但不激活
  }

  void commit() {
    current_phase_ = next_phase_;
    // 提交网络配置更改
  }

  Device &get_device(int rank) {
    static Device dummy(0, 0, {});
    return dummy;
  }

  Switch &get_switch(int dev_id) {
    static Switch dummy(0, "", NetConfEntry{});
    return dummy;
  }

  OpticalSwitch &get_optical_switch(int dev_id) {
    static OpticalSwitch dummy(0, "", NetConfEntry{});
    return dummy;
  }

private:
  int current_phase_;
  int next_phase_;
  NetConfEntry cluster_controller_addr_;
  std::map<int, std::vector<NetConfConnection>> phase_connections_;
  std::map<std::pair<int, int>,
           std::vector<std::tuple<NetConfConnection, std::string>>>
      phase_transforms_;
  std::vector<Device> devices;
  std::vector<Switch> switches;
  std::vector<OpticalSwitch> optical_switches;
  std::map<int, int> connections;
  std::map<FlowTable, int> flow_tables;
};

ClusterContext::ClusterContext() : pimpl_(std::make_unique<Impl>()) {}

NetConfEntry ClusterContext::getConfEntry() const {
  return pimpl_->getConfEntry();
}

void ClusterContext::registerPhase(
    int phase, std::vector<NetConfConnection> &connections) {
  pimpl_->registerPhase(phase, connections);
}

int ClusterContext::getPhase() const { return pimpl_->getPhase(); }

void ClusterContext::registerPhaseTransform(
    int prevPhase, int nextPhase,
    std::vector<std::tuple<NetConfConnection, std::string>> &commands) {
  pimpl_->registerPhaseTransform(prevPhase, nextPhase, commands);
}

void ClusterContext::preCommit(int nextPhase) { pimpl_->preCommit(nextPhase); }

void ClusterContext::commit() { pimpl_->commit(); }

// ConnectionContext::Impl 实现
struct ConnectionContext::Impl {
  std::shared_ptr<ClusterContext> cluster_context_;
  std::map<std::pair<Endpoint, Endpoint>, std::shared_ptr<Connection>>
      connections_;

  Impl() : cluster_context_(std::make_shared<ClusterContext>()) {}

  TransportFlags getChannelTypes(std::shared_ptr<Connection> connection) {
    // 获取连接支持的传输通道类型
    return connection ? TransportFlags(connection->transport())
                      : TransportFlags();
  }

  int remoteRankOf(std::shared_ptr<Connection> connection) {
    return connection ? connection->remoteRank() : 0;
  }

  std::shared_ptr<Connection> getConnection(Endpoint localEndpoint,
                                            Endpoint remoteEndpoint,
                                            TransportFlags transport) {
    auto key = std::make_pair(localEndpoint, remoteEndpoint);
    auto it = connections_.find(key);
    if (it != connections_.end()) {
      return it->second;
    }
    // TODO: 创建新连接
    return nullptr;
  }

  void connect(Endpoint localEndpoint, Endpoint remoteEndpoint) {
    auto key = std::make_pair(localEndpoint, remoteEndpoint);
    // TODO: 建立实际连接
  }

  void disconnect(Endpoint localEndpoint, Endpoint remoteEndpoint) {
    auto key = std::make_pair(localEndpoint, remoteEndpoint);
    connections_.erase(key);
  }

  bool isConnected(Endpoint localEndpoint, Endpoint remoteEndpoint) {
    auto key = std::make_pair(localEndpoint, remoteEndpoint);
    return connections_.find(key) != connections_.end();
  }

  bool notifyOperatior(RegisteredMemory mem, std::vector<Endpoint> &endpoints,
                       int tag) {
    // 通知其他进程关于操作符的内存分配
    return true; // 简化实现
  }

  std::shared_ptr<ClusterContext> clusterContext() { return cluster_context_; }
};

ConnectionContext::ConnectionContext() : pimpl_(std::make_unique<Impl>()) {}

ConnectionContext::~ConnectionContext() {}

TransportFlags
ConnectionContext::getChannelTypes(std::shared_ptr<Connection> connection) {
  return pimpl_->getChannelTypes(connection);
}

int ConnectionContext::remoteRankOf(std::shared_ptr<Connection> connection) {
  return pimpl_->remoteRankOf(connection);
}

std::shared_ptr<Connection> ConnectionContext::getConnection(
    Endpoint localEndpoint, Endpoint remoteEndpoint, TransportFlags transport) {
  return pimpl_->getConnection(localEndpoint, remoteEndpoint, transport);
}

void ConnectionContext::connect(Endpoint localEndpoint,
                                Endpoint remoteEndpoint) {
  pimpl_->connect(localEndpoint, remoteEndpoint);
}

void ConnectionContext::disconnect(Endpoint localEndpoint,
                                   Endpoint remoteEndpoint) {
  pimpl_->disconnect(localEndpoint, remoteEndpoint);
}

bool ConnectionContext::isConnected(Endpoint localEndpoint,
                                    Endpoint remoteEndpoint) {
  return pimpl_->isConnected(localEndpoint, remoteEndpoint);
}

bool ConnectionContext::notifyOperatior(RegisteredMemory mem,
                                        std::vector<Endpoint> &endpoints,
                                        int tag) {
  return pimpl_->notifyOperatior(mem, endpoints, tag);
}

std::shared_ptr<ClusterContext> ConnectionContext::clusterContext() {
  return pimpl_->clusterContext();
}

// Communicator::Impl 实现
class Communicator::Impl : public std::enable_shared_from_this<Impl> {
public:
  Impl()
      : mem_context_(std::make_shared<MemoryContext>()),
        conn_context_(std::make_shared<ConnectionContext>()) {}
  ~Impl() {}

  std::vector<char> serialize() const { return self_endpoint_.serialize(); }

  void registerRemoteInfos(std::vector<char> &data, int remoteRank) {
    endpoints_[remoteRank] = Endpoint::deserialize(data);
  }

  std::vector<RegisteredMemory> getOperatorSpace(size_t bufferSize, int tag,
                                                 std::vector<int> &ranks) {
    RegisteredMemory mem =
        mem_context_->allocateWorkspace(bufferSize, false, tag);
    std::vector<Endpoint> endpoints;
    for (int rank : ranks) {
      auto it = endpoints_.find(rank);
      if (it != endpoints_.end()) {
        endpoints.push_back(it->second);
      }
    }
    if (!conn_context_->notifyOperatior(mem, endpoints, tag)) {
      mem_context_->freeWorkspace(mem);
      // LOG_ERROR << "Failed to notify operator";
      return {};
    }
    return mem_context_->waitWorkSpace(ranks, tag);
  }

  std::vector<Connection> getConnections(int tag, std::vector<int> &ranks,
                                         TransportFlags transport) {
    std::vector<Connection> result;
    for (int rank : ranks) {
      auto it = endpoints_.find(rank);
      if (it != endpoints_.end()) {
        auto conn =
            conn_context_->getConnection(self_endpoint_, it->second, transport);
        if (conn) {
          // 由于Connection是抽象基类，这里需要解引用
          // 简化实现，暂时返回空vector
        }
      }
    }
    return result;
  }

  void switchPhase(int phase) {
    if (auto cluster_ctx = conn_context_->clusterContext()) {
      cluster_ctx->preCommit(phase);
      cluster_ctx->commit();
    }
  }

  std::shared_ptr<MemoryContext> memoryContext() { return mem_context_; }

  std::shared_ptr<ConnectionContext> connectionContext() {
    return conn_context_;
  }

private:
  std::shared_ptr<MemoryContext> mem_context_;
  std::shared_ptr<ConnectionContext> conn_context_;
  Endpoint self_endpoint_;
  std::map<int, Endpoint> endpoints_;
};

Communicator::Communicator() : pimpl_(std::make_unique<Impl>()) {}

Communicator::~Communicator() {}

std::vector<char> Communicator::serialize() const {
  return pimpl_->serialize();
}

void Communicator::registerRemoteInfos(std::vector<char> &data,
                                       int remoteRank) {
  pimpl_->registerRemoteInfos(data, remoteRank);
}

std::vector<RegisteredMemory>
Communicator::getOperatorSpace(size_t bufferSize, int tag,
                               std::vector<int> &ranks) {
  return pimpl_->getOperatorSpace(bufferSize, tag, ranks);
}

std::vector<Connection> Communicator::getConnections(int tag,
                                                     std::vector<int> &ranks,
                                                     TransportFlags transport) {
  return pimpl_->getConnections(tag, ranks, transport);
}

void Communicator::switchPhase(int phase) { pimpl_->switchPhase(phase); }

std::shared_ptr<MemoryContext> Communicator::memoryContext() {
  return pimpl_->memoryContext();
}

std::shared_ptr<ConnectionContext> Communicator::connectionContext() {
  return pimpl_->connectionContext();
}

// Capsule::Impl 实现
struct Capsule::Impl {
  int id_;
  std::string path_;
  std::string name_;
  std::string collective_;

  Impl(int id, const std::string &path) : id_(id), path_(path) {
    // 从path加载胶囊文件，解析操作元数据
    name_ = "capsule_" + std::to_string(id);
    collective_ = "allreduce"; // 默认集合操作
  }
};

Capsule::Capsule(int id, std::string &path)
    : pimpl_(std::make_unique<Impl>(id, path)) {}

std::string Capsule::name() const { return pimpl_ ? pimpl_->name_ : ""; }

std::string Capsule::collective() const {
  return pimpl_ ? pimpl_->collective_ : "";
}

// Operator::Impl 实现
class Operator::Impl {
public:
  Impl(std::shared_ptr<Communicator> comm, const std::string &path)
      : communicator_(comm), path_(path), is_inplace_(false),
        is_configurable_(true) {
    name_ = "operator_from_" + path;
    collective_ = "allreduce";
  }

  ~Impl() {}

  std::string name() const { return name_; }
  std::string collective() const { return collective_; }
  bool isInplace() const { return is_inplace_; }
  bool isConfigurable() const { return is_configurable_; }

  Event execute(int rank, void *input, void *output, DataType dtype,
                size_t inputSize, size_t outputSize, Event &event, bool flush) {
    Event result;
    result.flush = []() {
      // 刷新网络队列，确保数据发送完成
    };
    result.wait = []() {
      // 等待操作完成
    };
    result.record = []() {
      // 记录事件，用于性能分析或同步
    };
    return result;
  }

private:
  std::weak_ptr<Communicator> communicator_;
  std::string path_;
  std::string name_;
  std::string collective_;
  bool is_inplace_;
  bool is_configurable_;
  std::vector<int> ranks_;
  std::vector<RegisteredMemory> operator_spaces_;
  Event complete_event;
};

Operator::Operator(std::shared_ptr<Communicator> communicator,
                   std::string &path)
    : impl_(std::make_unique<Impl>(communicator, path)) {}

Operator::~Operator() {}

std::string Operator::name() const { return impl_->name(); }

std::string Operator::collective() const { return impl_->collective(); }

bool Operator::isInplace() const { return impl_->isInplace(); }

bool Operator::isConfigurable() const { return impl_->isConfigurable(); }

Event Operator::execute(int rank, void *input, void *output, DataType dtype,
                        size_t inputSize, size_t outputSize, Event &event,
                        bool flush) {
  return impl_->execute(rank, input, output, dtype, inputSize, outputSize,
                        event, flush);
}

}; // namespace pccl