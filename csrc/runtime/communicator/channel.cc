#include "runtime/communicator/channel.h"
#include "plugins/aroce/roce_adapter.h"
#include "plugins/atcp/tcp_adapter.h"
#include "utils/logging.h"
#include <algorithm>
#include <cmath>
#include <string>

namespace pccl::communicator {

static constexpr std::string RDMA_CHANNEL_KEY = "RDMA";
static constexpr std::string RDMA_QP_NUM = "RDMA_QP_NUM";
static constexpr std::string RDMA_GID = "RDMA_GID";
static constexpr std::string RDMA_CHANNEL_KEY = "RDMA";
static constexpr std::string RDMA_CHANNEL_KEY = "RDMA";

static std::vector<ChannelType> get_shared_channel_type(Endpoint node0, Endpoint Node1) {

}

bool Endpoint::operator==(const Endpoint& other) const {
  return attributes_ == other.attributes_;
}

bool Endpoint::operator<(const Endpoint& other) const {
  return attributes_ < other.attributes_;
}

std::size_t Endpoint::hash() const {
  std::size_t seed = 0;
  for (const auto& attr : attributes_) {
    seed ^= std::hash<std::string>{}(attr.first) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    seed ^= std::hash<std::string>{}(attr.second) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  }
  return seed;
}

nlohmann::json Endpoint::toJson() const {
  return nlohmann::json(attributes_);
}

Endpoint Endpoint::fromJson(const nlohmann::json& j) {
  Endpoint ep;
  if (j.is_object()) {
    for (auto it = j.begin(); it != j.end(); ++it) {
      ep.attributes_[it.key()] = it.value();
    }
  }
  return ep;
}

float NetMetrics::effectiveBandwidth(size_t data_size) const {
  if (data_size == 0) {
    return 0;
  }
  return (float)data_size / ((float)data_size / bandwidth_ + latency_);
}

nlohmann::json OobMessage::toJson() const {
  nlohmann::json j;
  j["type"] = static_cast<int>(type_);
  j["seq_id"] = seq_id_;
  j["timestamp"] = timestamp_;
  j["src_rank"] = src_rank;
  j["payload"] = payload_;
  return j;
}

OobMessage OobMessage::fromJson(const nlohmann::json& j) {
  OobMessage msg;
  msg.type_ = static_cast<OobMsgType>(j.value("type", 0));
  msg.seq_id_ = j.value("seq_id", 0);
  msg.timestamp_ = j.value("timestamp", 0);
  msg.src_rank = j.value("src_rank", -1);
  
  auto payload_json = j.find("payload");
  if (payload_json != j.end() && payload_json->is_array()) {
    msg.payload_.assign(payload_json->begin(), payload_json->end());
  }
  
  return msg;
}

std::unique_ptr<CommEngine> CommEngine::create(ChannelType type,
                                               const Endpoint& local_endpoint,
                                               const Endpoint& remote_endpoint) {
  return nullptr;
}

CommunicationChannel::CommunicationChannel(const Endpoint& self_endpoint, const Endpoint& peer_endpoint)
  : self_endpoint_(self_endpoint), peer_endpoint_(peer_endpoint),
    state_(ConnectionState::DISCONNECTED), next_tx_id_(1) {}

CommunicationChannel::~CommunicationChannel() {
  shutdown();
}

bool CommunicationChannel::initialize() {
  if (state_ != ConnectionState::DISCONNECTED) {
    PCCL_LOG_WARN("Channel already initialized");
    return false;
  }
  bool connected = false;
  
  
  state_ = ConnectionState::CONNECTED;
  PCCL_LOG_INFO("Communication channel initializing");
  return true;
}

void CommunicationChannel::shutdown() {
  if (state_ == ConnectionState::DISCONNECTED) return;
  
  disconnect();
  
  std::lock_guard<std::mutex> lock(engines_mutex_);
  engines_.clear();
  
  state_ = ConnectionState::DISCONNECTED;
  PCCL_LOG_INFO("Communication channel shutdown");
}

bool CommunicationChannel::prepSend(const MemRegion& dst, const MemRegion& src, uint64_t tx_id) {
  if (state_ != ConnectionState::CONNECTED) {
    PCCL_LOG_ERROR("Channel not connected");
    return false;
  }
  
  auto* engine = selectBestEngine(src.size_);
  if (!engine) {
    PCCL_LOG_ERROR("No suitable engine found for data size {}", src.size_);
    return false;
  }
  
  if ((int)src.size_ > engine->getStats().max_frag_) {
    return setupFragmentedSend(dst, src, tx_id);
  }
  
  std::lock_guard<std::mutex> lock(tx_mutex_);
  return engine->prepSend(dst, src, tx_id);
}

bool CommunicationChannel::postSend(uint64_t tx_id) {
  if (state_ != ConnectionState::CONNECTED) return false;
  
  std::lock_guard<std::mutex> lock(tx_mutex_);
  
  auto frag_it = fragmented_tx_.find(tx_id);
  if (frag_it != fragmented_tx_.end()) {
    bool success = true;
    for (const auto& fragment : frag_it->second.fragment_tx_ids_) {
      auto engine_it = engines_.find(fragment.first);
      if (engine_it != engines_.end()) {
        if (!engine_it->second.engine_->postSend(fragment.second)) {
          success = false;
        }
      }
    }
    return success;
  }
  
  for (auto& engine_pair : engines_) {
    if (engine_pair.second.engine_->postSend(tx_id)) {
      return true;
    }
  }
  
  return false;
}

void CommunicationChannel::signal(uint64_t tx_mask) {
  std::lock_guard<std::mutex> lock(engines_mutex_);
  for (auto& engine_pair : engines_) {
    engine_pair.second.engine_->signal(tx_mask);
  }
}

uint64_t CommunicationChannel::checkSignals() {
  uint64_t completed_mask = 0;
  
  std::lock_guard<std::mutex> lock(engines_mutex_);
  for (auto& engine_pair : engines_) {
    completed_mask |= engine_pair.second.engine_->checkSignals();
  }
  
  return completed_mask;
}

bool CommunicationChannel::waitTx(uint64_t tx_id, uint32_t timeout) {
  auto frag_it = fragmented_tx_.find(tx_id);
  if (frag_it != fragmented_tx_.end()) {
    auto start_time = std::chrono::steady_clock::now();
    auto timeout_time = start_time + std::chrono::milliseconds(timeout);
    
    while (std::chrono::steady_clock::now() < timeout_time) {
      if (frag_it->second.completed_fragments_ >= frag_it->second.fragment_tx_ids_.size()) {
        fragmented_tx_.erase(frag_it);
        return true;
      }
      
      uint64_t completed = checkSignals();
      for (const auto& fragment : frag_it->second.fragment_tx_ids_) {
        if (completed & (1ULL << fragment.second)) {
          frag_it->second.completed_fragments_++;
          handleFragmentCompletion(tx_id, fragment.first, fragment.second);
        }
      }
      
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    
    return false;
  }
  
  std::lock_guard<std::mutex> lock(engines_mutex_);
  for (auto& engine_pair : engines_) {
    if (engine_pair.second.engine_->waitTx(tx_id, timeout)) {
      return true;
    }
  }
  
  return false;
}

bool CommunicationChannel::flush(uint64_t tx_mask) {
  auto start_time = std::chrono::steady_clock::now();
  constexpr uint32_t timeout_ms = 5000;
  
  while (std::chrono::steady_clock::now() - start_time < std::chrono::milliseconds(timeout_ms)) {
    uint64_t completed = checkSignals();
    if ((completed & tx_mask) == tx_mask) {
      return true;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  
  return false;
}

bool CommunicationChannel::connect(const Endpoint& self, const Endpoint& peer) {
  if (state_ == ConnectionState::CONNECTED) {
    disconnect();
  }
  
  self_endpoint_ = self;
  peer_endpoint_ = peer;
  
  std::lock_guard<std::mutex> lock(engines_mutex_);
  bool any_connected = false;
  
  for (auto& engine_pair : engines_) {
    if (engine_pair.second.engine_->connect(self, peer)) {
      any_connected = true;
    }
  }
  
  state_ = any_connected ? ConnectionState::CONNECTED : ConnectionState::ERROR;
  return any_connected;
}

void CommunicationChannel::disconnect() {
  std::lock_guard<std::mutex> lock(engines_mutex_);
  
  for (auto& engine_pair : engines_) {
    engine_pair.second.engine_->disconnect();
  }
  
  state_ = ConnectionState::DISCONNECTED;
}

bool CommunicationChannel::connected() const {
  return state_ == ConnectionState::CONNECTED;
}

NetMetrics CommunicationChannel::getStats() const {
  std::lock_guard<std::mutex> lock(engines_mutex_);
  
  NetMetrics best_metrics{};
  double best_score = -1.0;
  
  for (const auto& engine_pair : engines_) {
    double score = engine_pair.second.current_score_;
    if (score > best_score) {
      best_metrics = engine_pair.second.metrics_;
      best_score = score;
    }
  }
  
  return best_metrics;
}

void CommunicationChannel::updateStats(const NetMetrics& stats) {
}

bool CommunicationChannel::supportsMemType(uint32_t mem_type) const {
  std::lock_guard<std::mutex> lock(engines_mutex_);
  
  for (const auto& engine_pair : engines_) {
    if (engine_pair.second.engine_->supportsMemType(mem_type)) {
      return true;
    }
  }
  
  return false;
}

uint32_t CommunicationChannel::maxConcurrentTx() const {
  std::lock_guard<std::mutex> lock(engines_mutex_);
  
  uint32_t max_tx = 0;
  for (const auto& engine_pair : engines_) {
    max_tx = std::max(max_tx, engine_pair.second.engine_->maxConcurrentTx());
  }
  
  return max_tx;
}

bool CommunicationChannel::addEngine(std::unique_ptr<CommEngine> engine) {
  if (!engine) return false;
  
  std::lock_guard<std::mutex> lock(engines_mutex_);
  ChannelType type = engine->getType();
  
  EngineInfo info;
  info.engine_ = std::move(engine);
  info.metrics_ = info.engine_->getStats();
  info.current_score_ = calculateEngineScore(info, 1024);
  
  engines_[type] = std::move(info);
  PCCL_LOG_INFO("Added engine type {} to channel", static_cast<int>(type));
  
  return true;
}

bool CommunicationChannel::addEngine(ChannelType type) {
  return false;
}

bool CommunicationChannel::removeEngine(ChannelType type) {
  std::lock_guard<std::mutex> lock(engines_mutex_);
  return engines_.erase(type) > 0;
}

bool CommunicationChannel::hasEngine(ChannelType type) const {
  std::lock_guard<std::mutex> lock(engines_mutex_);
  return engines_.find(type) != engines_.end();
}

const Endpoint& CommunicationChannel::getSelfEndpoint() const {
  return self_endpoint_;
}

const Endpoint& CommunicationChannel::getPeerEndpoint() const {
  return peer_endpoint_;
}

std::vector<ChannelType> CommunicationChannel::getAvailableEngineTypes() const {
  std::lock_guard<std::mutex> lock(engines_mutex_);
  
  std::vector<ChannelType> types;
  for (const auto& engine_pair : engines_) {
    types.push_back(engine_pair.first);
  }
  
  return types;
}

void CommunicationChannel::updateEngineMetrics(ChannelType type, const NetMetrics& metrics) {
  std::lock_guard<std::mutex> lock(engines_mutex_);
  
  auto it = engines_.find(type);
  if (it != engines_.end()) {
    it->second.metrics_ = metrics;
    it->second.current_score_ = calculateEngineScore(it->second, 1024);
  }
}

CommEngine* CommunicationChannel::selectBestEngine(size_t data_size) const {
  std::lock_guard<std::mutex> lock(engines_mutex_);
  
  CommEngine* best_engine = nullptr;
  double best_score = -1.0;
  
  for (const auto& engine_pair : engines_) {
    double score = calculateEngineScore(engine_pair.second, data_size);
    if (score > best_score) {
      best_score = score;
      best_engine = engine_pair.second.engine_.get();
    }
  }
  
  return best_engine;
}

bool CommunicationChannel::setupFragmentedSend(const MemRegion& dst, const MemRegion& src, uint64_t tx_id) {
  return false;
}

void CommunicationChannel::handleFragmentCompletion(uint64_t parent_tx_id, ChannelType engine_type, uint64_t fragment_tx_id) {
}

double CommunicationChannel::calculateEngineScore(const EngineInfo& engine, size_t data_size) const {
  return engine.metrics_.scoreForDataSize(data_size);
}

ChannelManager::ChannelManager(std::unique_ptr<OobChannel> oob_channel)
  : oob_channel_(std::move(oob_channel)) {}

ChannelManager::~ChannelManager() {
  shutdown();
}

bool ChannelManager::initialize(const Endpoint& self_endpoint) {
  if (!oob_channel_) return false;
  
  self_endpoint_ = self_endpoint;
  
  if (!oob_channel_->init(self_endpoint)) {
    return false;
  }
  
  oob_channel_->registerHandler(OobMsgType::NODE_JOIN,
                               [this](const OobMessage& msg) {
                                 handleOobMessage(msg);
                               });
  
  oob_channel_->registerHandler(OobMsgType::NODE_LEAVE,
                               [this](const OobMessage& msg) {
                                 handleOobMessage(msg);
                               });
  
  oob_channel_->registerHandler(OobMsgType::TOPOLOGY_UPDATE,
                               [this](const OobMessage& msg) {
                                 handleOobMessage(msg);
                               });
  
  PCCL_LOG_INFO("Channel manager initialized");
  return true;
}

void ChannelManager::shutdown() {
  if (oob_channel_) {
    oob_channel_->shutdown();
  }
  
  std::lock_guard<std::mutex> lock(channels_mutex_);
  channels_.clear();
  
  std::lock_guard<std::mutex> node_lock(node_map_mutex_);
  node_map_.clear();
}

bool ChannelManager::registerEngineFactory(ChannelType type,
                         std::function<std::unique_ptr<CommEngine>(const Endpoint&, const Endpoint&)> factory) {
  engine_factories_[type] = factory;
  return true;
}

std::shared_ptr<CommunicationChannel> ChannelManager::getOrCreateChannel(const Endpoint& peer_endpoint) {
  std::lock_guard<std::mutex> lock(channels_mutex_);
  
  auto it = channels_.find(peer_endpoint);
  if (it != channels_.end()) {
    return it->second;
  }
  
  auto channel = std::make_shared<CommunicationChannel>(self_endpoint_, peer_endpoint);
  if (!channel->initialize()) {
    return nullptr;
  }
  
  establishOptimalEngines(peer_endpoint);
  
  channels_[peer_endpoint] = channel;
  return channel;
}

bool ChannelManager::removeChannel(const Endpoint& peer_endpoint) {
  std::lock_guard<std::mutex> lock(channels_mutex_);
  return channels_.erase(peer_endpoint) > 0;
}

std::shared_ptr<CommunicationChannel> ChannelManager::getChannel(const Endpoint& peer_endpoint) const {
  std::lock_guard<std::mutex> lock(channels_mutex_);
  auto it = channels_.find(peer_endpoint);
  return it != channels_.end() ? it->second : nullptr;
}

bool ChannelManager::addEngineToChannel(const Endpoint& peer_endpoint, ChannelType type) {
  auto channel = getChannel(peer_endpoint);
  if (!channel) return false;
  
  auto factory_it = engine_factories_.find(type);
  if (factory_it == engine_factories_.end()) return false;
  
  auto engine = factory_it->second(self_endpoint_, peer_endpoint);
  if (!engine) return false;
  
  return channel->addEngine(std::move(engine));
}

bool ChannelManager::removeEngineFromChannel(const Endpoint& peer_endpoint, ChannelType type) {
  auto channel = getChannel(peer_endpoint);
  return channel ? channel->removeEngine(type) : false;
}

void ChannelManager::updateNodeMapping(const std::string& node_id, const Endpoint& endpoint) {
  std::lock_guard<std::mutex> lock(node_map_mutex_);
  
  NodeInfo info;
  info.node_id_ = node_id;
  info.endpoint_ = endpoint;
  info.last_updated_ = std::chrono::duration_cast<std::chrono::milliseconds>(
    std::chrono::system_clock::now().time_since_epoch()).count();
  
  node_map_[node_id] = info;
}

void ChannelManager::removeNodeMapping(const std::string& node_id) {
  std::lock_guard<std::mutex> lock(node_map_mutex_);
  node_map_.erase(node_id);
}

Endpoint ChannelManager::resolveNodeId(const std::string& node_id) const {
  std::lock_guard<std::mutex> lock(node_map_mutex_);
  auto it = node_map_.find(node_id);
  return it != node_map_.end() ? it->second.endpoint_ : Endpoint{};
}

std::vector<Endpoint> ChannelManager::getConnectedPeers() const {
  if (!oob_channel_) return {};
  return oob_channel_->getConnectedNodes();
}

std::set<ChannelType> ChannelManager::getAvailableChannelTypes(const Endpoint& peer_endpoint) const {
  std::set<ChannelType> types;
  
  for (const auto& factory_pair : engine_factories_) {
    types.insert(factory_pair.first);
  }
  
  return types;
}

void ChannelManager::handleOobMessage(const OobMessage& msg) {
  switch (msg.type_) {
    case OobMsgType::NODE_JOIN: {
      break;
    }
    case OobMsgType::NODE_LEAVE: {
      break;
    }
    case OobMsgType::TOPOLOGY_UPDATE: {
      break;
    }
    default:
      break;
  }
}

void ChannelManager::establishOptimalEngines(const Endpoint& peer_endpoint) {
  auto available_types = getAvailableChannelTypes(peer_endpoint);
  
  for (ChannelType type : available_types) {
    addEngineToChannel(peer_endpoint, type);
  }
}

void ChannelManager::reconnectNode(const std::string& node_id, const Endpoint& new_endpoint) {
  auto old_endpoint = resolveNodeId(node_id);
  if (old_endpoint == new_endpoint) return;
  
  updateNodeMapping(node_id, new_endpoint);
  
  std::lock_guard<std::mutex> lock(channels_mutex_);
  auto it = channels_.find(old_endpoint);
  if (it != channels_.end()) {
    auto channel = it->second;
    channels_.erase(it);
    channels_[new_endpoint] = channel;
    
    channel->connect(self_endpoint_, new_endpoint);
  }
}

} // namespace pccl::communicator
