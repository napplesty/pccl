#pragma once

#include "plugins/aroce/roce_utils.h"
#include "runtime/communicator/channel.h"
#include <atomic>
#include <mutex>
#include <unordered_map>
#include <queue>

namespace pccl::communicator {

class RoCEAdapter : public CommEngine {
public:
  RoCEAdapter(const Endpoint& local_endpoint, const Endpoint& remote_endpoint);
  ~RoCEAdapter();

  bool prepSend(const MemRegion& dst, const MemRegion& src, uint64_t tx_id) override;
  bool postSend(uint64_t tx_id) override;
  void signal(uint64_t tx_mask) override;
  uint64_t checkSignals() override;
  bool waitTx(uint64_t tx_id, uint32_t timeout) override;
  bool flush(uint64_t tx_mask) override;
  
  bool connect(const Endpoint& self, const Endpoint& peer) override;
  void disconnect() override;
  bool connected() const override;

  NetMetrics getStats() const override;
  void updateStats(const NetMetrics& stats) override;
  bool supportsMemType(uint32_t mem_type) const override;
  uint32_t maxConcurrentTx() const override;
  
  ChannelType getType() const override;
  const Endpoint& getSelfEndpoint() const override;
  const Endpoint& getPeerEndpoint() const override;

private:
  struct RoCETransaction {
    uint64_t tx_id;
    bool completed;
    void *src_addr;
    void *dst_addr;
    size_t data_size;
    uint64_t start_time;
  };

  bool initializeVerbs();
  bool setupConnection();
  bool registerMemoryRegions();
  void cleanup();

  Endpoint self_endpoint_;
  Endpoint peer_endpoint_;
  std::unique_ptr<VerbsManager> verbs_manager_;
  VerbsManager::ConnectionId connection_id_;
  VerbsManager::QPId qp_id_;
  
  std::unordered_map<uint64_t, RoCETransaction> active_transactions_;
  std::queue<uint64_t> completed_transactions_;
  std::atomic<uint64_t> signal_mask_{0};
  std::atomic<bool> connected_{false};
  
  mutable std::mutex tx_mutex_;
  mutable std::mutex stats_mutex_;
  NetMetrics current_stats_;
  
  std::unordered_map<void*, std::shared_ptr<VerbsMemoryRegion>> registered_mrs_;
  uint64_t next_tx_id_{1};
};

} // namespace pccl::communicator
