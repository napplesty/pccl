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

struct Endpoint {
  std::map<std::string, std::string> attributes_;
  
  bool operator==(const Endpoint& other) const;
  bool operator<(const Endpoint& other) const;
  std::size_t hash() const;

  nlohmann::json toJson() const;
  static Endpoint fromJson(const nlohmann::json& json_data);
  
  std::string toString() const;
};

struct MemRegion {
  void* ptr_;
  size_t size_;
  uint32_t lkey_;
  uint32_t rkey_;
  
  MemRegion() : ptr_(nullptr), size_(0), lkey_(0), rkey_(0) {}
  MemRegion(void* ptr, size_t size, uint32_t lkey = 0, uint32_t rkey = 0) 
    : ptr_(ptr), size_(size), lkey_(lkey), rkey_(rkey) {}
};

struct NetMetrics {
  float bandwidth_;
  float latency_;
  uint64_t total_bytes_;
  uint64_t total_operations_;
  
  NetMetrics() : bandwidth_(0), latency_(0), total_bytes_(0), total_operations_(0) {}
  float effectiveBandwidth(size_t data_size) const;
};

class CommEngine {
public:
  virtual ~CommEngine() = default;
  
  virtual uint64_t prepSend(const MemRegion& dst, const MemRegion& src) = 0;
  virtual void postSend() = 0;
  virtual void signal(uint64_t tx_mask) = 0;
  virtual bool checkSignal(uint64_t tx_mask) = 0;
  virtual bool waitTx(uint64_t tx_id) = 0;
  virtual bool flush() = 0;
  
  virtual bool connect(const Endpoint& self, const Endpoint& peer) = 0;
  virtual void disconnect() = 0;
  virtual bool connected() const = 0;

  virtual NetMetrics getStats() const = 0;
  virtual void updateStats(const NetMetrics& stats) = 0;
  
  virtual ChannelType getType() const = 0;
  virtual const Endpoint& getSelfEndpoint() const = 0;
  virtual const Endpoint& getPeerEndpoint() const = 0;
  
  virtual bool registerMemoryRegion(const MemRegion& region) = 0;
  virtual bool deregisterMemoryRegion(const MemRegion& region) = 0;
};

class CommunicationChannel {
public:
  CommunicationChannel(const Endpoint& self_endpoint, const Endpoint& peer_endpoint);
  ~CommunicationChannel();
  
  bool initialize();
  void shutdown();
  
  uint64_t prepSend(const MemRegion& dst, const MemRegion& src);
  bool postSend();
  void signal(uint64_t tx_mask);
  bool checkSignal(uint64_t tx_mask);
  bool waitTx(uint64_t tx_id);
  bool flush();
  
  bool connect();
  void disconnect();
  bool connected() const;

  NetMetrics getStats() const;
  void updateStats(const NetMetrics& stats);
  
  bool addEngine(std::unique_ptr<CommEngine> engine);
  bool removeEngine(ChannelType type);
  
  bool registerMemoryRegion(const MemRegion& region);
  bool deregisterMemoryRegion(const MemRegion& region);
  
  const Endpoint& getSelfEndpoint() const;
  const Endpoint& getPeerEndpoint() const;
  
  ChannelType getBestEngineType() const;
  
private:
  struct EngineInfo {
    std::unique_ptr<CommEngine> engine_;
    NetMetrics metrics_;
    bool enabled_;
    EngineInfo() : engine_(nullptr), enabled_(false) {}
    EngineInfo(std::unique_ptr<CommEngine> engine) 
      : engine_(std::move(engine)), enabled_(true) {}
  };
  
  struct Transaction {
    uint64_t id_;
    std::vector<std::pair<ChannelType, uint64_t>> fragments_;
    size_t total_size_;
    std::atomic<size_t> completed_fragments_;
    std::atomic<bool> completed_;
    
    Transaction(uint64_t id, size_t size) 
      : id_(id), total_size_(size), completed_fragments_(0), completed_(false) {}
  };
  
  Endpoint self_endpoint_;
  Endpoint peer_endpoint_;
  std::map<ChannelType, EngineInfo> engines_;
  mutable std::mutex engines_mutex_;
  std::atomic<ConnectionState> state_;
  
  std::unordered_map<uint64_t, std::shared_ptr<Transaction>> transactions_;
  std::atomic<uint64_t> next_tx_id_;
  std::mutex tx_mutex_;
  
  std::unordered_map<void*, MemRegion> registered_regions_;
  std::mutex regions_mutex_;
  
  ChannelType selectBestEngine(size_t data_size) const;
  void updateEngineMetrics(ChannelType type, size_t data_size, double latency);
  void handleFragmentCompletion(ChannelType engine_type, uint64_t fragment_id);
  void cleanupCompletedTransactions();
};

class ChannelManager {
public:
  ChannelManager();
  ~ChannelManager();
  
  bool initialize(const Endpoint& self_endpoint);
  void shutdown();
  
  bool registerEngineFactory(ChannelType type, 
    std::function<std::unique_ptr<CommEngine>(const Endpoint&, const Endpoint&)> factory);
  
  std::shared_ptr<CommunicationChannel> getOrCreateChannel(const Endpoint& peer_endpoint);
  bool removeChannel(const Endpoint& peer_endpoint);
  std::shared_ptr<CommunicationChannel> getChannel(const Endpoint& peer_endpoint) const;
  
  std::vector<Endpoint> getConnectedEndpoints() const;
  bool isEndpointConnected(const Endpoint& endpoint) const;
  
  void updateClusterConfigs(const std::map<int, nlohmann::json>& cluster_configs);
  
private:
  Endpoint self_endpoint_;
  std::map<Endpoint, std::shared_ptr<CommunicationChannel>> channels_;
  std::unordered_map<ChannelType, 
    std::function<std::unique_ptr<CommEngine>(const Endpoint&, const Endpoint&)>> engine_factories_;
  
  mutable std::mutex channels_mutex_;
  
  void establishChannel(const Endpoint& peer_endpoint);
  void reconnectChannel(const Endpoint& peer_endpoint);
};

} // namespace pccl::communicator
