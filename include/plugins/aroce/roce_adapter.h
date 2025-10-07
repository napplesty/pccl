#pragma once

#include <runtime/communicator/channel.h>
#include <plugins/aroce/roce_utils.h>
#include <memory>

namespace pccl::communicator {

class RoCEAdapter : public CommEngine {
public:
  RoCEAdapter(const Endpoint& self, const Endpoint& peer);
  ~RoCEAdapter() override;

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

private:
  ibv_send_wr buildSendWR(const MemRegion& dst, const MemRegion& src);
  ibv_send_wr buildWriteWR(const MemRegion& dst, const MemRegion& src);
  ibv_send_wr buildReadWR(const MemRegion& dst, const MemRegion& src);
  bool waitForCompletion(uint64_t tx_id);

  Endpoint self_endpoint_;
  Endpoint peer_endpoint_;
  std::unique_ptr<pccl::communicator::VerbsManager> verbs_manager_;
  pccl::communicator::VerbsManager::ConnectionId conn_id_{0};
  pccl::communicator::VerbsManager::QPId qp_id_{0};
  std::atomic<uint64_t> next_tx_id_{1};
  std::unordered_map<uint64_t, bool> completion_map_;
  std::mutex completion_mutex_;
};

} // namespace pccl::communicator
