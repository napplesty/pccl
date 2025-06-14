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

std::string TransportFlags::toString() const {
  return TransportFlagsBase::to_string();
}

// struct RegisteredMemory::Impl {
//   int rank_;
//   void *device_ptr_;
//   void *host_ptr_;
//   size_t size_;
//   BufferType type_;
//   uint64_t tag_;
//   bool is_host_memory_;
//   TransportFlags transports_;

//   Impl(int rank, void *device_ptr, void *host_ptr, size_t size, BufferType
//   type,
//        uint64_t tag, bool is_host_memory, TransportFlags transports)
//       : rank_(rank), device_ptr_(device_ptr), host_ptr_(host_ptr),
//       size_(size),
//         type_(type), tag_(tag), is_host_memory_(is_host_memory),
//         transports_(transports) {}
// };

// RegisteredMemory::RegisteredMemory(std::shared_ptr<Impl> pimpl)
//     : pimpl_(pimpl) {}

// int RegisteredMemory::rankOf() const { return pimpl_ ? pimpl_->rank_ : 0; }

// void *RegisteredMemory::hostPtr() const {
//   return pimpl_ ? pimpl_->host_ptr_ : nullptr;
// }

// void *RegisteredMemory::devicePtr() const {
//   return pimpl_ ? pimpl_->device_ptr_ : nullptr;
// }

// size_t RegisteredMemory::size() const { return pimpl_ ? pimpl_->size_ : 0; }

// BufferType RegisteredMemory::type() const {
//   return pimpl_ ? pimpl_->type_
//                 : BufferType::LIB; // Default to LIB if pimpl_ is null
// }

// uint64_t RegisteredMemory::tag() const {
//   return pimpl_ ? pimpl_->tag_ : 0; // Default to 0 if pimpl_ is null
// }

// TransportFlags RegisteredMemory::transports() const {
//   return pimpl_ ? pimpl_->transports_ : TransportFlags();
// }

// std::vector<char> RegisteredMemory::serialize() const {
//   nlohmann::json j;
//   if (pimpl_) {
//     j["rank"] = pimpl_->rank_;
//     j["isHostMemory"] = pimpl_->is_host_memory_;
//     j["size"] = pimpl_->size_;
//     j["type"] = static_cast<int>(pimpl_->type_);
//     j["tag"] = pimpl_->tag_;
//     j["transports"] = pimpl_->transports_.toBitset().to_string();
//   }
//   std::string s = j.dump();
//   return std::vector<char>(s.begin(), s.end());
// }

// RegisteredMemory RegisteredMemory::deserialize(const std::vector<char> &data)
// {
//   if (data.empty()) {
//     LOG_WARNING << "RegisteredMemory::deserialize called with empty data.";
//     return RegisteredMemory(nullptr);
//   }

//   std::string s(data.begin(), data.end());
//   nlohmann::json j;
//   try {
//     j = nlohmann::json::parse(s);
//   } catch (const nlohmann::json::parse_error &e) {
//     LOG_ERROR << "Failed to parse RegisteredMemory JSON: " << e.what() <<
//     "."; return RegisteredMemory(nullptr);
//   }

//   if (!j.is_object() || j.empty()) {
//     LOG_ERROR << "Deserialized JSON for RegisteredMemory is not a valid
//     object "
//                  "or is empty. JSON string: "
//               << s;
//     return RegisteredMemory(nullptr);
//   }

//   TransportFlags transports_val;
//   std::string transports_str = j.value("transports", "");
//   if (!transports_str.empty()) {
//     transports_val = TransportFlags::fromString(transports_str);
//   }

//   auto impl_ptr = std::make_shared<RegisteredMemory::Impl>(
//       j.value("rank", 0), nullptr, nullptr,
//       j.value("size", static_cast<size_t>(0)),
//       static_cast<BufferType>(
//           j.value("type", static_cast<int>(BufferType::LIB))),
//       j.value("tag", static_cast<uint64_t>(0)), j.value("isHostMemory",
//       false), transports_val);

//   return RegisteredMemory(impl_ptr);
// }

// class MemoryContext::Impl {
// public:
//   Impl(std::shared_ptr<Communicator> communicator)
//       : communicator_(communicator) {
//     auto lib_buf = GpuBuffer<char>(Config::LIB_BUFFER_SIZE, true);
//     auto host_buf = GpuBuffer<char>(Config::HOST_BUFFER_SIZE, true);
//     auto device_buf = GpuBuffer<char>(Config::DEVICE_BUFFER_SIZE, false);

//     free_lib_func_ = lib_buf.free_func_;
//     free_host_func_ = host_buf.free_func_;
//     free_device_func_ = device_buf.free_func_;

//     lib_buffer_ = std::make_unique<RegisteredMemory>(
//         MemoryContext::createRegisteredMemory(
//             std::make_shared<RegisteredMemory::Impl>(
//                 getEnv()->rank, lib_buf.devicePtr(), lib_buf.hostPtr(),
//                 Config::LIB_BUFFER_SIZE, BufferType::LIB, 0, true,
//                 ConnectionContext::getAvailableTransports())));

//     host_buffer_ = std::make_unique<RegisteredMemory>(
//         MemoryContext::createRegisteredMemory(
//             std::make_shared<RegisteredMemory::Impl>(
//                 getEnv()->rank, host_buf.devicePtr(), host_buf.hostPtr(),
//                 Config::HOST_BUFFER_SIZE, BufferType::HOST, 0, true,
//                 ConnectionContext::getAvailableTransports())));

//     device_buffer_ = std::make_unique<RegisteredMemory>(
//         MemoryContext::createRegisteredMemory(
//             std::make_shared<RegisteredMemory::Impl>(
//                 getEnv()->rank, device_buf.devicePtr(), nullptr,
//                 Config::DEVICE_BUFFER_SIZE, BufferType::DEVICE, 0, false,
//                 ConnectionContext::getAvailableTransports())));

//     lib = new (lib_buffer_->hostPtr()) LibBufferSlot();
//   }

//   ~Impl() {
//     free_lib_func_(lib_buffer_->hostPtr());
//     free_host_func_(host_buffer_->hostPtr());
//     free_device_func_(device_buffer_->devicePtr());
//   }

//   RegisteredMemory getPredefinedMemory(BufferType type) {
//     switch (type) {
//     case BufferType::LIB:
//       return *lib_buffer_;
//     case BufferType::HOST:
//       return *host_buffer_;
//     case BufferType::DEVICE:
//       return *device_buffer_;
//     default:
//       return MemoryContext::createRegisteredMemory(nullptr);
//     }
//   }

//   RegisteredMemory allocateWorkSpace(size_t size, bool isHostMemory, int tag)
//   {
//     std::lock_guard<std::mutex> lock(mutex_);
//     static constexpr size_t granularity =
//         Config::HOST_BUFFER_SIZE / Config::NUM_SLOT;
//     size_t last_index = isHostMemory ? 0 : Config::NUM_SLOT;
//     size_t max_index = last_index + Config::NUM_SLOT;
//     for (; last_index < max_index &&
//            lib->predefined_slot[last_index].status == SlotStatus::ALLOCATED;
//          last_index++) {
//     }

//     if (last_index == max_index) {
//       return MemoryContext::createRegisteredMemory(nullptr);
//     }

//     size_t allocated_slot_count = (size + granularity - 1) / granularity;

//     for (size_t next_index = last_index + 1; next_index <= max_index;
//          next_index++) {
//       if (lib->predefined_slot[next_index - 1].status ==
//           SlotStatus::ALLOCATED) {
//         last_index = next_index;
//         continue;
//       }
//       if (next_index - last_index == allocated_slot_count) {
//         break;
//       }
//     }

//     if (last_index + allocated_slot_count > max_index) {
//       return MemoryContext::createRegisteredMemory(nullptr);
//     }

//     for (size_t i = 0; i < allocated_slot_count; i++) {
//       lib->predefined_slot[last_index + i].status = SlotStatus::ALLOCATED;
//       lib->predefined_slot[last_index + i].tag = tag;
//     }

//     auto impl_ptr = std::make_shared<RegisteredMemory::Impl>(
//         getEnv()->rank,
//         (char *)lib_buffer_->devicePtr() + last_index * granularity,
//         (char *)lib_buffer_->hostPtr() + last_index * granularity, size,
//         isHostMemory ? BufferType::HOST : BufferType::DEVICE, tag,
//         isHostMemory, ConnectionContext::getAvailableTransports());

//     return MemoryContext::createRegisteredMemory(impl_ptr);
//   }

//   RegisteredMemory registerAsWorkSpace(void *buffer, bool isHostMemory, int
//   tag,
//                                        Transport transport) {
//     std::lock_guard<std::mutex> lock(mutex_);
//     auto impl_ptr = std::make_shared<RegisteredMemory::Impl>(
//         getEnv()->rank, buffer, isHostMemory ? buffer : nullptr, 0,
//         isHostMemory ? BufferType::HOST : BufferType::DEVICE, tag,
//         isHostMemory, TransportFlags(transport));
//     return MemoryContext::createRegisteredMemory(impl_ptr);
//   }

//   void unregister(RegisteredMemory memory) {
//     std::lock_guard<std::mutex> lock(mutex_);
//     // 释放内存槽位
//     size_t granularity = Config::HOST_BUFFER_SIZE / Config::NUM_SLOT;
//     if (memory.devicePtr() >= lib_buffer_->devicePtr() &&
//         memory.devicePtr() <
//             (char *)lib_buffer_->devicePtr() + Config::LIB_BUFFER_SIZE) {
//       size_t index =
//           (char *)memory.devicePtr() - (char *)lib_buffer_->devicePtr();
//       size_t slot_index = index / granularity;
//       size_t allocated_slot_count =
//           (memory.size() + granularity - 1) / granularity;

//       for (size_t i = 0; i < allocated_slot_count; i++) {
//         lib->predefined_slot[slot_index + i].status = SlotStatus::FREE;
//         lib->predefined_slot[slot_index + i].tag = 0;
//       }
//     }
//   }

//   std::vector<RegisteredMemory> waitWorkSpaceReady(std::vector<int> &ranks,
//                                                    int tag) {
//     std::vector<RegisteredMemory> result;
//     for (int rank : ranks) {
//       // 等待远程工作空间就绪的逻辑
//       // 这里简化实现，返回空的RegisteredMemory
//       result.push_back(MemoryContext::createRegisteredMemory(nullptr));
//     }
//     return result;
//   }

//   std::shared_ptr<Endpoint> getEndpoint(int rank) {
//     if (auto comm = communicator_.lock()) {
//       return comm->getEndpoint(rank);
//     }
//     return nullptr;
//   }

//   void registerEndpoint(std::shared_ptr<Endpoint> endpoint, int remoteRank) {
//     if (auto comm = communicator_.lock()) {
//       comm->registerEndpoint(endpoint, remoteRank);
//     }
//   }

// private:
//   std::weak_ptr<Communicator> communicator_;
//   std::function<void(void *)> free_lib_func_;
//   std::function<void(void *)> free_host_func_;
//   std::function<void(void *)> free_device_func_;
//   std::unique_ptr<RegisteredMemory> lib_buffer_;
//   std::unique_ptr<RegisteredMemory> host_buffer_;
//   std::unique_ptr<RegisteredMemory> device_buffer_;
//   LibBufferSlot *lib;
//   std::mutex mutex_;
// };

// MemoryContext::MemoryContext(std::shared_ptr<Communicator> communicator)
//     : pimpl_(std::make_unique<Impl>(communicator)) {}

// MemoryContext::~MemoryContext() {}

// // 添加静态工厂方法来创建RegisteredMemory
// RegisteredMemory MemoryContext::createRegisteredMemory(
//     std::shared_ptr<RegisteredMemory::Impl> impl) {
//   return RegisteredMemory(impl);
// }

// std::shared_ptr<Endpoint> MemoryContext::getEndpoint(int rank) {
//   return pimpl_->getEndpoint(rank);
// }

// void MemoryContext::registerEndpoint(std::shared_ptr<Endpoint> endpoint,
//                                      int remoteRank) {
//   pimpl_->registerEndpoint(endpoint, remoteRank);
// }

// RegisteredMemory MemoryContext::getPredefinedMemory(BufferType type) {
//   return pimpl_->getPredefinedMemory(type);
// }

// RegisteredMemory MemoryContext::allocateWorkSpace(size_t size,
//                                                   bool isHostMemory, int tag)
//                                                   {
//   return pimpl_->allocateWorkSpace(size, isHostMemory, tag);
// }

// RegisteredMemory MemoryContext::registerAsWorkSpace(void *buffer,
//                                                     bool isHostMemory, int
//                                                     tag, Transport transport)
//                                                     {
//   return pimpl_->registerAsWorkSpace(buffer, isHostMemory, tag, transport);
// }

// void MemoryContext::unregister(RegisteredMemory memory) {
//   pimpl_->unregister(memory);
// }

// std::vector<RegisteredMemory>
// MemoryContext::waitWorkSpaceReady(std::vector<int> &ranks, int tag) {
//   return pimpl_->waitWorkSpaceReady(ranks, tag);
// }

// // Endpoint::Impl 实现
// struct Endpoint::Impl {
//   int rank_;
//   Transport transport_;
//   int maxWriteQueueSize_;
//   int maxCompleteQueueSize_;
//   uint64_t latency_;
//   uint64_t bandwidth_;
//   NetConfEntry networkAddress_;
//   std::weak_ptr<Communicator> communicator_;

//   Impl(std::shared_ptr<Communicator> communicator)
//       : rank_(0), transport_(Transport::Unknown), maxWriteQueueSize_(0),
//         maxCompleteQueueSize_(0), latency_(0), bandwidth_(0),
//         communicator_(communicator) {}
// };

// Endpoint::Endpoint(std::shared_ptr<Communicator> communicator)
//     : pimpl_(std::make_shared<Impl>(communicator)) {}

// Endpoint::Endpoint(std::shared_ptr<Impl> pimpl) : pimpl_(pimpl) {}

// int Endpoint::rank() const { return pimpl_ ? pimpl_->rank_ : 0; }

// Transport Endpoint::transport() const {
//   return pimpl_ ? pimpl_->transport_ : Transport::Unknown;
// }

// RegisteredMemory Endpoint::getRegisteredMemory(BufferType type) {
//   if (auto comm = pimpl_->communicator_.lock()) {
//     auto memCtx = comm->memoryContext();
//     return memCtx->getPredefinedMemory(type);
//   }
//   return RegisteredMemory::deserialize({});
// }

// int Endpoint::maxWriteQueueSize() const {
//   return pimpl_ ? pimpl_->maxWriteQueueSize_ : 0;
// }

// int Endpoint::maxCompleteQueueSize() const {
//   return pimpl_ ? pimpl_->maxCompleteQueueSize_ : 0;
// }

// uint64_t Endpoint::latency() const { return pimpl_ ? pimpl_->latency_ : 0; }

// uint64_t Endpoint::bandwidth() const { return pimpl_ ? pimpl_->bandwidth_ :
// 0; }

// std::vector<char> Endpoint::serialize() const {
//   nlohmann::json j;
//   if (pimpl_) {
//     j["rank"] = pimpl_->rank_;
//     j["transport"] = static_cast<int>(pimpl_->transport_);
//     j["maxWriteQueueSize"] = pimpl_->maxWriteQueueSize_;
//     j["maxCompleteQueueSize"] = pimpl_->maxCompleteQueueSize_;
//     j["latency"] = pimpl_->latency_;
//     j["bandwidth"] = pimpl_->bandwidth_;
//   }
//   std::string s = j.dump();
//   return std::vector<char>(s.begin(), s.end());
// }

// Endpoint Endpoint::deserialize(const std::vector<char> &data) {
//   std::string s(data.begin(), data.end());
//   nlohmann::json j = nlohmann::json::parse(s);

//   auto impl = std::make_shared<Impl>(nullptr);
//   impl->rank_ = j.value("rank", 0);
//   impl->transport_ = static_cast<Transport>(j.value("transport", 0));
//   impl->maxWriteQueueSize_ = j.value("maxWriteQueueSize", 0);
//   impl->maxCompleteQueueSize_ = j.value("maxCompleteQueueSize", 0);
//   impl->latency_ = j.value("latency", static_cast<uint64_t>(0));
//   impl->bandwidth_ = j.value("bandwidth", static_cast<uint64_t>(0));

//   return Endpoint(impl);
// }

// // Device::Impl 实现
// struct Device::Impl {
//   int id_;
//   int rank_;
//   std::vector<std::tuple<TransportFlags, NetConfEntry>> endpoint_infos_;

//   Impl(int id, int rank,
//        std::vector<std::tuple<TransportFlags, NetConfEntry>> infos)
//       : id_(id), rank_(rank), endpoint_infos_(infos) {}
// };

// Device::Device(
//     int id, int rank,
//     std::vector<std::tuple<TransportFlags, NetConfEntry>> endpoint_infos)
//     : pimpl_(std::make_unique<Impl>(id, rank, endpoint_infos)) {}

// NetConfEntry Device::getConfEntry(TransportFlags transport) {
//   if (pimpl_) {
//     for (const auto &[flags, entry] : pimpl_->endpoint_infos_) {
//       if ((flags & transport).any()) {
//         return entry;
//       }
//     }
//   }
//   return NetConfEntry{};
// }

// int Device::uid() const { return pimpl_ ? pimpl_->id_ : 0; }

// // Switch::Impl 实现
// struct Switch::Impl {
//   int id_;
//   std::string name_;
//   NetConfEntry network_address_;

//   Impl(int id, const std::string &name, NetConfEntry addr)
//       : id_(id), name_(name), network_address_(addr) {}
// };

// Switch::Switch(int id, const std::string &name, NetConfEntry network_address)
//     : pimpl_(std::make_unique<Impl>(id, name, network_address)) {}

// NetConfEntry Switch::getConfEntry() const {
//   return pimpl_ ? pimpl_->network_address_ : NetConfEntry{};
// }

// int Switch::uid() const { return pimpl_ ? pimpl_->id_ : 0; }

// // OpticalSwitch::Impl 实现
// struct OpticalSwitch::Impl {
//   int id_;
//   std::string name_;
//   NetConfEntry network_address_;

//   Impl(int id, const std::string &name, NetConfEntry addr)
//       : id_(id), name_(name), network_address_(addr) {}
// };

// OpticalSwitch::OpticalSwitch(int id, const std::string &name,
//                              NetConfEntry network_address)
//     : pimpl_(std::make_unique<Impl>(id, name, network_address)) {}

// NetConfEntry OpticalSwitch::getConfEntry() const {
//   return pimpl_ ? pimpl_->network_address_ : NetConfEntry{};
// }

// int OpticalSwitch::uid() const { return pimpl_ ? pimpl_->id_ : 0; }

// // ClusterContext::Impl 实现
// class ClusterContext::Impl {
// public:
//   Impl() : current_phase_(0) {}
//   ~Impl() {}

//   NetConfEntry getConfEntry() const { return cluster_controller_addr_; }

//   void registerPhase(int phase, std::vector<NetConfConnection> &connections)
//   {
//     phase_connections_[phase] = connections;
//   }

//   int getPhase() const { return current_phase_; }

//   void registerPhaseTransform(
//       int prevPhase, int nextPhase,
//       std::vector<std::tuple<int, NetConfConnection, std::string>> &commands)
//       {
//     phase_transforms_[{prevPhase, nextPhase}] = commands;
//   }

//   void preCommit(int nextPhase) {
//     next_phase_ = nextPhase;
//     // 准备阶段转换配置但不激活
//   }

//   void commit() {
//     current_phase_ = next_phase_;
//     // 提交网络配置更改
//   }

// private:
//   int current_phase_;
//   int next_phase_;
//   NetConfEntry cluster_controller_addr_;
//   std::map<int, std::vector<NetConfConnection>> phase_connections_;
//   std::map<std::pair<int, int>,
//            std::vector<std::tuple<int, NetConfConnection, std::string>>>
//       phase_transforms_;
// };

// ClusterContext::ClusterContext() : pimpl_(std::make_unique<Impl>()) {}

// ClusterContext::~ClusterContext() {}

// NetConfEntry ClusterContext::getConfEntry() const {
//   return pimpl_->getConfEntry();
// }

// void ClusterContext::registerPhase(
//     int phase, std::vector<NetConfConnection> &connections) {
//   pimpl_->registerPhase(phase, connections);
// }

// int ClusterContext::getPhase() const { return pimpl_->getPhase(); }

// void ClusterContext::registerPhaseTransform(
//     int prevPhase, int nextPhase,
//     std::vector<std::tuple<int, NetConfConnection, std::string>> &commands) {
//   pimpl_->registerPhaseTransform(prevPhase, nextPhase, commands);
// }

// void ClusterContext::preCommit(int nextPhase) { pimpl_->preCommit(nextPhase);
// }

// void ClusterContext::commit() { pimpl_->commit(); }

// // ConnectionContext::Impl 实现
// struct ConnectionContext::Impl {
//   std::shared_ptr<ClusterContext> cluster_context_;
//   std::map<int, std::shared_ptr<Endpoint>> endpoints_;
//   std::map<std::pair<int, int>, std::shared_ptr<Connection>> connections_;
//   std::weak_ptr<Communicator> communicator_;

//   Impl(std::shared_ptr<Communicator> communicator)
//       : cluster_context_(std::make_shared<ClusterContext>()),
//         communicator_(communicator) {}

//   std::shared_ptr<Endpoint> getEndpoint(int rank) {
//     auto it = endpoints_.find(rank);
//     if (it != endpoints_.end()) {
//       return it->second;
//     }
//     return nullptr;
//   }

//   void registerEndpoint(std::shared_ptr<Endpoint> endpoint, int remoteRank) {
//     endpoints_[remoteRank] = endpoint;
//   }

//   TransportFlags getChannelTypes(std::shared_ptr<Connection> connection) {
//     return connection ? TransportFlags(connection->transport())
//                       : TransportFlags();
//   }

//   int remoteRankOf(std::shared_ptr<Connection> connection) {
//     return connection ? connection->remoteRank() : 0;
//   }

//   // getConnection方法在connection.cc中定义
//   std::shared_ptr<Connection> getConnection(Endpoint localEndpoint,
//                                             Endpoint remoteEndpoint,
//                                             TransportFlags transport);

//   void connect(Endpoint localEndpoint, Endpoint remoteEndpoint) {
//     // TODO: 建立实际连接
//   }

//   void disconnect(Endpoint localEndpoint, Endpoint remoteEndpoint) {
//     auto key = std::make_pair(localEndpoint.rank(), remoteEndpoint.rank());
//     connections_.erase(key);
//   }

//   bool isConnected(Endpoint localEndpoint, Endpoint remoteEndpoint) {
//     auto key = std::make_pair(localEndpoint.rank(), remoteEndpoint.rank());
//     return connections_.find(key) != connections_.end();
//   }

//   bool notifyOperator(RegisteredMemory mem, std::vector<Endpoint> &endpoints)
//   {
//     // 通知其他进程关于操作符的内存分配
//     return true; // 简化实现
//   }

//   TransportFlags getTransportFlags(Endpoint localEndpoint,
//                                    Endpoint remoteEndpoint) {
//     // 获取两个端点之间可用的传输类型
//     return getAvailableTransports();
//   }

//   std::shared_ptr<ClusterContext> clusterContext() { return cluster_context_;
//   }
// };

// ConnectionContext::ConnectionContext(std::shared_ptr<Communicator>
// communicator)
//     : pimpl_(std::make_unique<Impl>(communicator)) {}

// ConnectionContext::~ConnectionContext() {}

// std::shared_ptr<Endpoint> ConnectionContext::getEndpoint(int rank) {
//   return pimpl_->getEndpoint(rank);
// }

// void ConnectionContext::registerEndpoint(std::shared_ptr<Endpoint> endpoint,
//                                          int remoteRank) {
//   pimpl_->registerEndpoint(endpoint, remoteRank);
// }

// TransportFlags
// ConnectionContext::getChannelTypes(std::shared_ptr<Connection> connection) {
//   return pimpl_->getChannelTypes(connection);
// }

// int ConnectionContext::remoteRankOf(std::shared_ptr<Connection> connection) {
//   return pimpl_->remoteRankOf(connection);
// }

// std::shared_ptr<Connection> ConnectionContext::getConnection(
//     Endpoint localEndpoint, Endpoint remoteEndpoint, TransportFlags
//     transport) {
//   return pimpl_->getConnection(localEndpoint, remoteEndpoint, transport);
// }

// void ConnectionContext::connect(Endpoint localEndpoint,
//                                 Endpoint remoteEndpoint) {
//   pimpl_->connect(localEndpoint, remoteEndpoint);
// }

// void ConnectionContext::disconnect(Endpoint localEndpoint,
//                                    Endpoint remoteEndpoint) {
//   pimpl_->disconnect(localEndpoint, remoteEndpoint);
// }

// bool ConnectionContext::isConnected(Endpoint localEndpoint,
//                                     Endpoint remoteEndpoint) {
//   return pimpl_->isConnected(localEndpoint, remoteEndpoint);
// }

// bool ConnectionContext::notifyOperator(RegisteredMemory mem,
//                                        std::vector<Endpoint> &endpoints) {
//   return pimpl_->notifyOperator(mem, endpoints);
// }

// TransportFlags ConnectionContext::getTransportFlags(Endpoint localEndpoint,
//                                                     Endpoint remoteEndpoint)
//                                                     {
//   return pimpl_->getTransportFlags(localEndpoint, remoteEndpoint);
// }

// TransportFlags ConnectionContext::getAvailableTransports() {
//   // 返回当前系统支持的所有传输类型
//   return TransportFlags(Transport::HostIpc) | Transport::CudaIpc |
//          Transport::IB | Transport::Ethernet | Transport::NVLS;
// }

// std::shared_ptr<ClusterContext> ConnectionContext::clusterContext() {
//   return pimpl_->clusterContext();
// }

// // Operator::Impl 实现
// class Operator::Impl {
// public:
//   Impl(std::string &path, std::shared_ptr<Communicator> comm)
//       : communicator_(comm), path_(path), is_inplace_(false),
//         is_configurable_(true) {
//     // 从路径解析操作类型
//     size_t pos = path.find_last_of('/');
//     std::string filename =
//         (pos != std::string::npos) ? path.substr(pos + 1) : path;

//     // 根据文件名推断操作类型
//     if (filename.find("allreduce") != std::string::npos) {
//       collective_ = "allreduce";
//     } else if (filename.find("broadcast") != std::string::npos) {
//       collective_ = "broadcast";
//     } else if (filename.find("allgather") != std::string::npos) {
//       collective_ = "allgather";
//     } else if (filename.find("reduce") != std::string::npos) {
//       collective_ = "reduce";
//     } else {
//       collective_ = "custom";
//     }

//     name_ = "operator_" + collective_ + "_from_" + filename;

//     // 某些操作是原地的
//     is_inplace_ = (collective_ == "allreduce" || collective_ == "broadcast");
//   }

//   ~Impl() {}

//   std::string name() const { return name_; }
//   std::string collective() const { return collective_; }
//   bool isInplace() const { return is_inplace_; }
//   bool isConfigurable() const { return is_configurable_; }

//   Event execute(int rank, void *input, void *output, DataType dtype,
//                 size_t inputSize, size_t outputSize, Event &event, bool
//                 flush, uint64_t tag) {
//     // 计算元素数量
//     size_t dtype_size = getDtypeSize(dtype);
//     size_t count = inputSize / dtype_size;

//     // 创建或获取集合通信操作
//     if (!collective_op_) {
//       collective_op_ = createCollectiveOp(collective_, communicator_.lock());
//       if (!collective_op_) {
//         LOG_ERROR << "Failed to create collective operation: " <<
//         collective_; return event;
//       }
//     }

//     // 执行集合通信
//     Event result;
//     if (collective_op_) {
//       result = collective_op_->execute(input, output, count, dtype, event);
//     }

//     // 设置事件回调
//     result.flush = [flush]() {
//       if (flush) {
//         // 刷新所有待处理的操作
//         // TODO: 实现具体的刷新逻辑
//       }
//     };

//     result.wait = []() {
//       // 等待操作完成
//       // TODO: 实现具体的等待逻辑
//     };

//     result.record = []() {
//       // 记录时间戳或其他性能指标
//       // TODO: 实现具体的记录逻辑
//     };

//     return result;
//   }

// private:
//   std::weak_ptr<Communicator> communicator_;
//   std::string path_;
//   std::string name_;
//   std::string collective_;
//   bool is_inplace_;
//   bool is_configurable_;
//   std::shared_ptr<CollectiveOp> collective_op_;

//   // 获取数据类型大小
//   size_t getDtypeSize(DataType dtype) {
//     switch (dtype) {
//     case DataType::I8:
//     case DataType::U8:
//     case DataType::FP8_E4M3:
//     case DataType::FP8_E5M2:
//       return 1;
//     case DataType::I16:
//     case DataType::U16:
//     case DataType::FP16:
//     case DataType::BF16:
//       return 2;
//     case DataType::I32:
//     case DataType::U32:
//     case DataType::FP32:
//       return 4;
//     case DataType::I64:
//     case DataType::U64:
//       return 8;
//     default:
//       return 0;
//     }
//   }
// };

// // 前向声明（在collective_ops.cc中定义）
// std::shared_ptr<CollectiveOp>
// createCollectiveOp(const std::string &op_type,
//                    std::shared_ptr<Communicator> comm);

// // Communicator::Impl 实现
// class Communicator::Impl {
// public:
//   Impl() {}
//   ~Impl() {}

//   void initialize() {
//     mem_context_ =
//         std::make_shared<MemoryContext>(shared_from_this()->shared_from_this());
//     conn_context_ = std::make_shared<ConnectionContext>(
//         shared_from_this()->shared_from_this());
//     cluster_context_ = conn_context_->clusterContext();
//   }

//   std::shared_ptr<Endpoint> getEndpoint(int rank) {
//     auto it = endpoints_.find(rank);
//     if (it != endpoints_.end()) {
//       return it->second;
//     }
//     return nullptr;
//   }

//   void registerEndpoint(std::shared_ptr<Endpoint> endpoint, int remoteRank) {
//     endpoints_[remoteRank] = endpoint;
//   }

//   std::shared_ptr<MemoryContext> memoryContext() { return mem_context_; }

//   std::shared_ptr<ConnectionContext> connectionContext() {
//     return conn_context_;
//   }

//   std::shared_ptr<ClusterContext> clusterContext() { return cluster_context_;
//   }

//   std::shared_ptr<Operator> registerOperator(const std::string
//   &operator_path) {
//     std::string path = operator_path;
//     return std::make_shared<Operator>(path,
//                                       shared_from_this()->shared_from_this());
//   }

//   void registerClusterContext(const std::string &cluster_config_path) {
//     // 从配置文件加载集群上下文
//     // TODO: 实现配置文件解析逻辑
//   }

//   std::shared_ptr<Communicator> shared_from_this() {
//     return
//     std::static_pointer_cast<Communicator>(parent_->shared_from_this());
//   }

//   void setParent(Communicator *parent) { parent_ = parent; }

// private:
//   std::shared_ptr<MemoryContext> mem_context_;
//   std::shared_ptr<ConnectionContext> conn_context_;
//   std::shared_ptr<ClusterContext> cluster_context_;
//   std::map<int, std::shared_ptr<Endpoint>> endpoints_;
//   Communicator *parent_;
// };

// Communicator::Communicator() : pimpl_(std::make_unique<Impl>()) {
//   pimpl_->setParent(this);
//   pimpl_->initialize();
// }

// Communicator::~Communicator() {}

// std::shared_ptr<Endpoint> Communicator::getEndpoint(int rank) {
//   return pimpl_->getEndpoint(rank);
// }

// void Communicator::registerEndpoint(std::shared_ptr<Endpoint> endpoint,
//                                     int remoteRank) {
//   pimpl_->registerEndpoint(endpoint, remoteRank);
// }

// std::shared_ptr<MemoryContext> Communicator::memoryContext() {
//   return pimpl_->memoryContext();
// }

// std::shared_ptr<ConnectionContext> Communicator::connectionContext() {
//   return pimpl_->connectionContext();
// }

// std::shared_ptr<ClusterContext> Communicator::clusterContext() {
//   return pimpl_->clusterContext();
// }

// std::shared_ptr<Operator>
// Communicator::registerOperator(const std::string &operator_path) {
//   return pimpl_->registerOperator(operator_path);
// }

// void Communicator::registerClusterContext(
//     const std::string &cluster_config_path) {
//   pimpl_->registerClusterContext(cluster_config_path);
// }

// }; // namespace pccl

// // Connection相关的实现（在connection.cc中定义，这里声明）
// namespace pccl {
// class ConnectionFactory {
// public:
//   static std::shared_ptr<Connection>
//   createConnection(int localRank, int remoteRank, Transport transport);
//   static Transport selectBestTransport(TransportFlags available, int
//   localRank,
//                                        int remoteRank);
// };

// // ConnectionContext::Impl的getConnection方法实现
// std::shared_ptr<Connection> ConnectionContext::Impl::getConnection(
//     Endpoint localEndpoint, Endpoint remoteEndpoint, TransportFlags
//     transport) {
//   auto key = std::make_pair(localEndpoint.rank(), remoteEndpoint.rank());
//   auto it = connections_.find(key);
//   if (it != connections_.end()) {
//     return it->second;
//   }

//   // 选择最优的传输协议
//   Transport selectedTransport = ConnectionFactory::selectBestTransport(
//       transport, localEndpoint.rank(), remoteEndpoint.rank());

//   if (selectedTransport != Transport::Unknown) {
//     auto conn = ConnectionFactory::createConnection(
//         localEndpoint.rank(), remoteEndpoint.rank(), selectedTransport);
//     if (conn) {
//       connections_[key] = conn;
//       return conn;
//     }
//   }

//   return nullptr;
// }
} // namespace pccl