#pragma once

#include "plugins/atcp/tcp_utils.h"
#include "runtime/communicator/channel.h"
#include <atomic>
#include <mutex>
#include <unordered_map>
#include <queue>

namespace pccl::communicator {

class TCPAdapter : public CommEngine {
public:
  TCPAdapter(const Endpoint& local_endpoint, const Endpoint& remote_endpoint);
  ~TCPAdapter();

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
  struct TCPTransaction {
    uint64_t tx_id;
    bool completed;
    void *src_addr;
    void *dst_addr;
    size_t data_size;
    uint64_t start_time;
  };

  bool initializeTCP();
  bool setupConnection();
  void processCompletions();
  void cleanup();

  Endpoint self_endpoint_;
  Endpoint peer_endpoint_;
  std::unique_ptr<pccl::TcpManager> tcp_manager_;
  pccl::TcpManager::ConnectionId connection_id_;
  pccl::TcpManager::QPId qp_id_;
  
  std::unordered_map<uint64_t, TCPTransaction> active_transactions_;
  std::queue<uint64_t> completed_transactions_;
  std::atomic<uint64_t> signal_mask_{0};
  std::atomic<bool> connected_{false};
  
  mutable std::mutex tx_mutex_;
  mutable std::mutex stats_mutex_;
  NetMetrics current_stats_;
  
  uint64_t next_tx_id_{1};
};

} // namespace pccl::communicator
