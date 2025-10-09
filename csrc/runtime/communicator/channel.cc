#include "runtime/communicator/channel.h"
#include "utils/logging.h"
#include <algorithm>

namespace pccl::communicator {

bool Endpoint::operator==(const Endpoint& other) const {
  return attributes_ == other.attributes_;
}

bool Endpoint::operator<(const Endpoint& other) const {
  return attributes_ < other.attributes_;
}

std::size_t Endpoint::hash() const {
  std::size_t seed = 0;
  for (const auto& pair : attributes_) {
    seed ^= std::hash<std::string>{}(pair.first) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    seed ^= std::hash<std::string>{}(pair.second) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  }
  return seed;
}

nlohmann::json Endpoint::toJson() const {
  nlohmann::json j;
  for (const auto& pair : attributes_) {
    j[pair.first] = pair.second;
  }
  return j;
}

Endpoint Endpoint::fromJson(const nlohmann::json& json_data) {
  Endpoint endpoint;
  for (auto it = json_data.begin(); it != json_data.end(); ++it) {
    endpoint.attributes_[it.key()] = it.value();
  }
  return endpoint;
}

std::string Endpoint::toString() const {
  return toJson().dump();
}

float NetMetrics::effectiveBandwidth(size_t data_size) const {
  if (latency_ == 0) return bandwidth_;
  double total_time = latency_ + (data_size / bandwidth_);
  return data_size / total_time;
}

CommunicationChannel::CommunicationChannel(const Endpoint& self_endpoint, const Endpoint& peer_endpoint)
  : self_endpoint_(self_endpoint), peer_endpoint_(peer_endpoint),
    state_(ConnectionState::DISCONNECTED), next_tx_id_(1) {
}

CommunicationChannel::~CommunicationChannel() {
  shutdown();
}

bool CommunicationChannel::initialize() {
  if (state_ != ConnectionState::DISCONNECTED) {
    PCCL_LOG_ERROR("Channel already initialized");
    return false;
  }
  
  state_ = ConnectionState::CONNECTING;
  
  bool any_connected = false;
  for (auto& [type, engine_info] : engines_) {
    if (engine_info.enabled_ && engine_info.engine_->connect(self_endpoint_, peer_endpoint_)) {
      any_connected = true;
      PCCL_LOG_INFO("{} engine connected successfully", 
                   type == ChannelType::RDMA ? "RDMA" : "TCP");
    } else {
      engine_info.enabled_ = false;
      PCCL_LOG_WARN("{} engine failed to connect", 
                   type == ChannelType::RDMA ? "RDMA" : "TCP");
    }
  }
  
  if (any_connected) {
    state_ = ConnectionState::CONNECTED;
    PCCL_LOG_INFO("Communication channel established with peer");
    return true;
  } else {
    state_ = ConnectionState::ERROR;
    PCCL_LOG_ERROR("All communication engines failed to connect");
    return false;
  }
}

void CommunicationChannel::shutdown() {
  state_ = ConnectionState::DISCONNECTED;
  
  for (auto& [type, engine_info] : engines_) {
    if (engine_info.engine_) {
      engine_info.engine_->disconnect();
    }
  }
  
  engines_.clear();
  transactions_.clear();
  registered_regions_.clear();
}

uint64_t CommunicationChannel::prepSend(const MemRegion& dst, const MemRegion& src) {
  if (state_ != ConnectionState::CONNECTED) {
    PCCL_LOG_ERROR("Channel not connected");
    return 0;
  }
  
  ChannelType best_engine = selectBestEngine(src.size_);
  auto it = engines_.find(best_engine);
  if (it == engines_.end() || !it->second.enabled_) {
    PCCL_LOG_ERROR("Best engine not available");
    return 0;
  }
  
  uint64_t tx_id = next_tx_id_.fetch_add(1);
  uint64_t fragment_id = it->second.engine_->prepSend(dst, src);
  
  if (fragment_id == 0) {
    PCCL_LOG_ERROR("Failed to prepare send operation");
    return 0;
  }
  
  auto transaction = std::make_shared<Transaction>(tx_id, src.size_);
  transaction->fragments_.push_back({best_engine, fragment_id});
  
  {
    std::lock_guard lock(tx_mutex_);
    transactions_[tx_id] = transaction;
  }
  
  return tx_id;
}

bool CommunicationChannel::postSend() {
  if (state_ != ConnectionState::CONNECTED) {
    return false;
  }
  
  bool all_success = true;
  for (auto& [type, engine_info] : engines_) {
    if (engine_info.enabled_) {
      engine_info.engine_->postSend();
    }
  }
  
  return all_success;
}

void CommunicationChannel::signal(uint64_t tx_mask) {
  for (auto& [type, engine_info] : engines_) {
    if (engine_info.enabled_) {
      engine_info.engine_->signal(tx_mask);
    }
  }
}

bool CommunicationChannel::checkSignal(uint64_t tx_mask) {
  for (auto& [type, engine_info] : engines_) {
    if (engine_info.enabled_ && engine_info.engine_->checkSignal(tx_mask)) {
      return true;
    }
  }
  return false;
}

bool CommunicationChannel::waitTx(uint64_t tx_id) {
  if (state_ != ConnectionState::CONNECTED) {
    return false;
  }
  
  std::shared_ptr<Transaction> transaction;
  {
    std::lock_guard lock(tx_mutex_);
    auto it = transactions_.find(tx_id);
    if (it == transactions_.end()) {
      PCCL_LOG_ERROR("Transaction {} not found", tx_id);
      return false;
    }
    transaction = it->second;
  }
  
  for (const auto& [engine_type, fragment_id] : transaction->fragments_) {
    auto it = engines_.find(engine_type);
    if (it != engines_.end() && it->second.enabled_) {
      if (!it->second.engine_->waitTx(fragment_id)) {
        PCCL_LOG_ERROR("Fragment wait failed for transaction {}", tx_id);
        return false;
      }
    }
  }
  
  transaction->completed_ = true;
  return true;
}

bool CommunicationChannel::flush() {
  if (state_ != ConnectionState::CONNECTED) {
    return false;
  }
  
  bool all_success = true;
  for (auto& [type, engine_info] : engines_) {
    if (engine_info.enabled_) {
      if (!engine_info.engine_->flush()) {
        all_success = false;
      }
    }
  }
  
  cleanupCompletedTransactions();
  return all_success;
}

bool CommunicationChannel::connect() {
  return initialize();
}

void CommunicationChannel::disconnect() {
  shutdown();
}

bool CommunicationChannel::connected() const {
  return state_ == ConnectionState::CONNECTED;
}

NetMetrics CommunicationChannel::getStats() const {
  NetMetrics overall;
  for (const auto& [type, engine_info] : engines_) {
    if (engine_info.enabled_) {
      auto stats = engine_info.engine_->getStats();
      overall.bandwidth_ += stats.bandwidth_;
      overall.latency_ = std::max(overall.latency_, stats.latency_);
      overall.total_bytes_ += stats.total_bytes_;
      overall.total_operations_ += stats.total_operations_;
    }
  }
  return overall;
}

void CommunicationChannel::updateStats(const NetMetrics& stats) {
  for (auto& [type, engine_info] : engines_) {
    if (engine_info.enabled_) {
      engine_info.engine_->updateStats(stats);
    }
  }
}

bool CommunicationChannel::addEngine(std::unique_ptr<CommEngine> engine) {
  if (!engine) {
    return false;
  }
  
  ChannelType type = engine->getType();
  std::lock_guard lock(engines_mutex_);
  engines_[type] = EngineInfo(std::move(engine));
  return true;
}

bool CommunicationChannel::removeEngine(ChannelType type) {
  std::lock_guard lock(engines_mutex_);
  auto it = engines_.find(type);
  if (it != engines_.end()) {
    it->second.engine_->disconnect();
    engines_.erase(it);
    return true;
  }
  return false;
}

bool CommunicationChannel::registerMemoryRegion(MemRegion& region) {
  std::lock_guard lock(regions_mutex_);
  registered_regions_[region.ptr_] = region;
  
  bool all_success = true;
  for (auto& [type, engine_info] : engines_) {
    if (engine_info.enabled_) {
      if (!engine_info.engine_->registerMemoryRegion(region)) {
        all_success = false;
      }
    }
  }
  
  return all_success;
}

bool CommunicationChannel::deregisterMemoryRegion(MemRegion& region) {
  std::lock_guard lock(regions_mutex_);
  auto it = registered_regions_.find(region.ptr_);
  if (it != registered_regions_.end()) {
    registered_regions_.erase(it);
  }
  
  bool all_success = true;
  for (auto& [type, engine_info] : engines_) {
    if (engine_info.enabled_) {
      if (!engine_info.engine_->deregisterMemoryRegion(region)) {
        all_success = false;
      }
    }
  }
  
  return all_success;
}

const Endpoint& CommunicationChannel::getSelfEndpoint() const {
  return self_endpoint_;
}

const Endpoint& CommunicationChannel::getPeerEndpoint() const {
  return peer_endpoint_;
}

ChannelType CommunicationChannel::getBestEngineType() const {
  return selectBestEngine(4096);
}

ChannelType CommunicationChannel::selectBestEngine(size_t data_size) const {
  if (engines_.count(ChannelType::RDMA) > 0) {
    auto it = engines_.find(ChannelType::RDMA);
    if (it != engines_.end() && it->second.enabled_) {
      return ChannelType::RDMA;
    }
  }
  
  if (engines_.count(ChannelType::TCP) > 0) {
    auto it = engines_.find(ChannelType::TCP);
    if (it != engines_.end() && it->second.enabled_) {
      return ChannelType::TCP;
    }
  }
  
  PCCL_LOG_ERROR("No available communication engine");
  return ChannelType::TCP;
}

void CommunicationChannel::updateEngineMetrics(ChannelType type, size_t data_size, double latency) {
  auto it = engines_.find(type);
  if (it != engines_.end()) {
    it->second.metrics_.total_bytes_ += data_size;
    it->second.metrics_.total_operations_++;
    it->second.metrics_.latency_ = (it->second.metrics_.latency_ + latency) / 2;
    it->second.metrics_.bandwidth_ = it->second.metrics_.total_bytes_ / 
                                   (it->second.metrics_.total_operations_ * it->second.metrics_.latency_);
  }
}

void CommunicationChannel::handleFragmentCompletion(ChannelType engine_type, uint64_t fragment_id) {
}

void CommunicationChannel::cleanupCompletedTransactions() {
  std::lock_guard lock(tx_mutex_);
  for (auto it = transactions_.begin(); it != transactions_.end();) {
    if (it->second->completed_) {
      it = transactions_.erase(it);
    } else {
      ++it;
    }
  }
}

ChannelManager::ChannelManager() {

}

ChannelManager::~ChannelManager() {
  shutdown();
}

bool ChannelManager::initialize(const Endpoint& self_endpoint) {
  self_endpoint_ = self_endpoint;
  return true;
}

void ChannelManager::shutdown() {
  std::lock_guard lock(channels_mutex_);
  for (auto& [endpoint, channel] : channels_) {
    channel->shutdown();
  }
  channels_.clear();
  engine_factories_.clear();
}

bool ChannelManager::registerEngineFactory(ChannelType type, 
  std::function<std::unique_ptr<CommEngine>(const Endpoint&, const Endpoint&)> factory) {
  
  engine_factories_[type] = factory;
  return true;
}

std::shared_ptr<CommunicationChannel> ChannelManager::getOrCreateChannel(const Endpoint& peer_endpoint) {
  std::lock_guard lock(channels_mutex_);
  
  auto it = channels_.find(peer_endpoint);
  if (it != channels_.end()) {
    return it->second;
  }
  
  auto channel = std::make_shared<CommunicationChannel>(self_endpoint_, peer_endpoint);
  
  for (const auto& [type, factory] : engine_factories_) {
    auto engine = factory(self_endpoint_, peer_endpoint);
    if (engine) {
      channel->addEngine(std::move(engine));
    }
  }
  
  if (!channel->initialize()) {
    PCCL_LOG_ERROR("Failed to initialize channel to peer");
    return nullptr;
  }
  
  channels_[peer_endpoint] = channel;
  PCCL_LOG_INFO("Created new channel to peer");
  return channel;
}

bool ChannelManager::removeChannel(const Endpoint& peer_endpoint) {
  std::lock_guard lock(channels_mutex_);
  auto it = channels_.find(peer_endpoint);
  if (it != channels_.end()) {
    it->second->shutdown();
    channels_.erase(it);
    return true;
  }
  return false;
}

std::shared_ptr<CommunicationChannel> ChannelManager::getChannel(const Endpoint& peer_endpoint) const {
  std::lock_guard lock(channels_mutex_);
  auto it = channels_.find(peer_endpoint);
  return it != channels_.end() ? it->second : nullptr;
}

std::vector<Endpoint> ChannelManager::getConnectedEndpoints() const {
  std::lock_guard lock(channels_mutex_);
  std::vector<Endpoint> endpoints;
  for (const auto& pair : channels_) {
    endpoints.push_back(pair.first);
  }
  return endpoints;
}

bool ChannelManager::isEndpointConnected(const Endpoint& endpoint) const {
  std::lock_guard lock(channels_mutex_);
  return channels_.find(endpoint) != channels_.end();
}

void ChannelManager::updateClusterConfigs(const std::map<int, nlohmann::json>& cluster_configs) {
  for (const auto& [rank, config] : cluster_configs) {
    Endpoint peer_endpoint = Endpoint::fromJson(config);
    
    if (peer_endpoint == self_endpoint_) {
      continue;
    }
    
    if (!isEndpointConnected(peer_endpoint)) {
      establishChannel(peer_endpoint);
    }
  }
}

void ChannelManager::establishChannel(const Endpoint& peer_endpoint) {
  getOrCreateChannel(peer_endpoint);
}

void ChannelManager::reconnectChannel(const Endpoint& peer_endpoint) {
  removeChannel(peer_endpoint);
  establishChannel(peer_endpoint);
}

} // namespace pccl::communicator
