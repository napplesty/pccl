#pragma once

#include <runtime/api/repr.h>
#include <runtime/communicator/channel.h>
#include <plugins/atcp/tcp_utils.h>
#include <memory>
#include <unordered_map>
#include <vector>

namespace pccl::communicator {

class TCPAdapter : public CommEngine {
public:
  TCPAdapter(const Endpoint& self, const Endpoint& peer);
  ~TCPAdapter() override;

  uint64_t prepSend(const MemRegion& dst, const MemRegion& src) override;
  void postSend() override;
  void signal(uint64_t tx_mask) override;
  bool checkSignal(uint64_t tx_mask) override;
  bool waitTx(uint64_t tx_id) override;
  bool flush() override;

  bool connect(const Endpoint& self, const Endpoint& peer) override;
  void disconnect() override;
  bool connected() const override;

  NetMetrics getStats() const override;
  void updateStats(const NetMetrics& stats) override;

  ChannelType getType() const override;
  const Endpoint& getSelfEndpoint() const override;
  const Endpoint& getPeerEndpoint() const override;

  bool registerMemoryRegion(MemRegion& region) override;
  bool deregisterMemoryRegion(MemRegion& region) override;

private:
  struct TCPMemoryRegion {
    void* addr;
    size_t length;
    uint32_t lkey;
    uint32_t rkey;
    runtime::ExecutorType executor_type;
  };

  struct TCPTransaction {
    uint64_t id;
    MemRegion src;
    MemRegion dst;
    std::vector<char> staging_buffer;
    bool completed;
  };

  pccl::TcpSendWR buildSendWR(const MemRegion& dst, const MemRegion& src);
  bool prepareDataForTransfer(const MemRegion& src, std::vector<char>& staging_buffer);
  bool transferDataToDestination(const MemRegion& dst, const std::vector<char>& staging_buffer);
  
  Endpoint self_endpoint_;
  Endpoint peer_endpoint_;
  std::unique_ptr<pccl::TcpManager> tcp_manager_;
  pccl::TcpManager::ConnectionId conn_id_{0};
  pccl::TcpManager::QPId qp_id_{0};
  std::atomic<uint64_t> next_tx_id_{1};
  
  std::unordered_map<void*, TCPMemoryRegion> registered_regions_;
  std::mutex regions_mutex_;
  std::atomic<uint32_t> next_key_{1};
  
  std::unordered_map<uint64_t, TCPTransaction> transactions_;
  std::mutex transactions_mutex_;
};

}
