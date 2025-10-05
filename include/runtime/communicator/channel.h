#pragma once

#include <string>
#include <cstdint>
#include <vector>
#include <functional>
#include <memory>
#include <unordered_map>
#include <atomic>
#include <mutex>
#include <map>
#include <set>
#include <nlohmann/json.hpp>

namespace pccl::communicator {

enum class ChannelType {
  RDMA,
  TCP,
};

enum class ConnectionState {
  DISCONNECTED,
  CONNECTING,
  CONNECTED,
  ERROR
};

enum class OobMsgType {
  HEARTBEAT,
  TOPOLOGY_UPDATE,
  NODE_JOIN,
  NODE_LEAVE,
  CHANNEL_REQUEST,
  CHANNEL_RESPONSE,
  METRICS_UPDATE,
  HEALTH_CHECK
};

struct Endpoint {
  std::map<std::string, std::string> attributes_;
  
  bool operator==(const Endpoint& other) const;
  bool operator<(const Endpoint& other) const;
  std::size_t hash() const;

  nlohmann::json toJson() const;
  static Endpoint fromJson(const nlohmann::json& json_data);
};

struct MemRegion {
  void* ptr_;
  size_t size_;
  uint32_t type_;
};

struct NetMetrics {
  float bandwidth_;
  float latency_;
  float loss_rate_;
  int max_frag_;
  int best_frag_;
  long long updated_;
  
  float effectiveBandwidth(size_t data_size) const;
  float scoreForDataSize(size_t data_size) const;
};

struct SendConfig {
  size_t frag_size_;
  int parallel_;
  uint32_t timeout_;
  uint32_t retries_;
};

struct OobMessage {
  OobMsgType type_;
  uint64_t seq_id_;
  uint64_t timestamp_;
  int src_rank;
  std::vector<uint8_t> payload_;

  nlohmann::json toJson() const;
  static OobMessage fromJson(const nlohmann::json& json_data);
};

class OobChannel {
public:
  virtual ~OobChannel() = default;
  
  virtual bool init(const Endpoint& self_endpoint) = 0;
  virtual void shutdown() = 0;
  virtual bool send(const OobMessage& msg, const Endpoint& dst) = 0;
  virtual bool broadcast(const OobMessage& msg, const std::vector<Endpoint>& targets) = 0;
  virtual bool poll(OobMessage* msg, uint32_t timeout_ms) = 0;
  virtual bool registerHandler(OobMsgType type, std::function<void(const OobMessage&)> handler) = 0;
  virtual std::vector<Endpoint> getConnectedNodes() const = 0;
  virtual bool isConnected(const Endpoint& endpoint) const = 0;
};

class CommEngine {
public:
  virtual ~CommEngine() = default;
  
  virtual bool prepSend(const MemRegion& dst, const MemRegion& src, uint64_t tx_id) = 0;
  virtual bool postSend(uint64_t tx_id) = 0;
  virtual void signal(uint64_t tx_mask) = 0;
  virtual uint64_t checkSignals() = 0;
  virtual bool waitTx(uint64_t tx_id, uint32_t timeout) = 0;
  virtual bool flush(uint64_t tx_mask) = 0;
  
  virtual bool connect(const Endpoint& self, const Endpoint& peer) = 0;
  virtual void disconnect() = 0;
  virtual bool connected() const = 0;

  virtual NetMetrics getStats() const = 0;
  virtual void updateStats(const NetMetrics& stats) = 0;
  virtual bool supportsMemType(uint32_t mem_type) const = 0;
  virtual uint32_t maxConcurrentTx() const = 0;
  
  virtual ChannelType getType() const = 0;
  virtual const Endpoint& getSelfEndpoint() const = 0;
  virtual const Endpoint& getPeerEndpoint() const = 0;

  static std::unique_ptr<CommEngine> create(ChannelType type, 
                                            const Endpoint& local_endpoint, 
                                            const Endpoint& remote_endpoint);
};

class CommunicationChannel {
public:
  CommunicationChannel(const Endpoint& self_endpoint, const Endpoint& peer_endpoint);
  ~CommunicationChannel();
  
  bool initialize();
  void shutdown();
  
  bool prepSend(const MemRegion& dst, const MemRegion& src, uint64_t tx_id);
  bool postSend(uint64_t tx_id);
  void signal(uint64_t tx_mask);
  uint64_t checkSignals();
  bool waitTx(uint64_t tx_id, uint32_t timeout);
  bool flush(uint64_t tx_mask);
  
  bool connect(const Endpoint& self, const Endpoint& peer);
  void disconnect();
  bool connected() const;

  NetMetrics getStats() const;
  void updateStats(const NetMetrics& stats);
  bool supportsMemType(uint32_t mem_type) const;
  uint32_t maxConcurrentTx() const;
  
  bool addEngine(std::unique_ptr<CommEngine> engine);
  bool addEngine(ChannelType type);
  bool removeEngine(ChannelType type);
  bool hasEngine(ChannelType type) const;
  
  const Endpoint& getSelfEndpoint() const;
  const Endpoint& getPeerEndpoint() const;
  
  std::vector<ChannelType> getAvailableEngineTypes() const;
  void updateEngineMetrics(ChannelType type, const NetMetrics& metrics);

private:
  struct EngineInfo {
    std::unique_ptr<CommEngine> engine_;
    NetMetrics metrics_;
    double current_score_;
  };
  
  Endpoint self_endpoint_;
  Endpoint peer_endpoint_;
  std::map<ChannelType, EngineInfo> engines_;
  mutable std::mutex engines_mutex_;
  std::atomic<ConnectionState> state_;
  
  struct FragmentedTransaction {
    uint64_t parent_tx_id_;
    std::vector<std::pair<ChannelType, uint64_t>> fragment_tx_ids_;
    size_t total_size_;
    size_t completed_fragments_;
  };
  
  std::unordered_map<uint64_t, FragmentedTransaction> fragmented_tx_;
  uint64_t next_tx_id_;
  std::mutex tx_mutex_;
  
  CommEngine* selectBestEngine(size_t data_size) const;
  bool setupFragmentedSend(const MemRegion& dst, const MemRegion& src, uint64_t tx_id);
  void handleFragmentCompletion(uint64_t parent_tx_id, ChannelType engine_type, uint64_t fragment_tx_id);
  double calculateEngineScore(const EngineInfo& engine, size_t data_size) const;
};

class ChannelManager {
public:
  ChannelManager(std::unique_ptr<OobChannel> oob_channel);
  ~ChannelManager();
  
  bool initialize(const Endpoint& self_endpoint);
  void shutdown();
  
  bool registerEngineFactory(ChannelType type, 
                           std::function<std::unique_ptr<CommEngine>(const Endpoint&, const Endpoint&)> factory);
  
  std::shared_ptr<CommunicationChannel> getOrCreateChannel(const Endpoint& peer_endpoint);
  bool removeChannel(const Endpoint& peer_endpoint);
  std::shared_ptr<CommunicationChannel> getChannel(const Endpoint& peer_endpoint) const;
  
  bool addEngineToChannel(const Endpoint& peer_endpoint, ChannelType type);
  bool removeEngineFromChannel(const Endpoint& peer_endpoint, ChannelType type);
  
  void updateNodeMapping(const std::string& node_id, const Endpoint& endpoint);
  void removeNodeMapping(const std::string& node_id);
  Endpoint resolveNodeId(const std::string& node_id) const;
  
  std::vector<Endpoint> getConnectedPeers() const;
  std::set<ChannelType> getAvailableChannelTypes(const Endpoint& peer_endpoint) const;

private:
  struct NodeInfo {
    std::string node_id_;
    Endpoint endpoint_;
    uint64_t last_updated_;
    std::set<ChannelType> supported_channels_;
  };
  
  std::unique_ptr<OobChannel> oob_channel_;
  Endpoint self_endpoint_;
  
  std::unordered_map<ChannelType, 
                    std::function<std::unique_ptr<CommEngine>(const Endpoint&, const Endpoint&)>> engine_factories_;
  std::map<Endpoint, std::shared_ptr<CommunicationChannel>> channels_;
  std::unordered_map<std::string, NodeInfo> node_map_;
  
  mutable std::mutex channels_mutex_;
  mutable std::mutex node_map_mutex_;
  
  void handleOobMessage(const OobMessage& msg);
  void establishOptimalEngines(const Endpoint& peer_endpoint);
  void reconnectNode(const std::string& node_id, const Endpoint& new_endpoint);
};

} // namespace pccl::communicator
