#pragma once

#include <runtime/communicator/channel.h>
#include <plugins/aroce/roce_utils.h>
#include <memory>
#include <unordered_map>

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

  bool registerMemoryRegion(MemRegion& region) override;
  bool deregisterMemoryRegion(MemRegion& region) override;

private:
  ibv_send_wr buildSendWR(const MemRegion& dst, const MemRegion& src);
  ibv_send_wr buildWriteWR(const MemRegion& dst, const MemRegion& src);
  ibv_send_wr buildReadWR(const MemRegion& dst, const MemRegion& src);
  bool waitForCompletion(uint64_t tx_id);
  uint32_t getLocalKey(void* addr);

  Endpoint self_endpoint_;
  Endpoint peer_endpoint_;
  pccl::communicator::VerbsManager *verbs_manager_;
  std::shared_ptr<pccl::communicator::VerbsProtectionDomain> pd_;
  pccl::communicator::VerbsManager::ConnectionId conn_id_{0};
  pccl::communicator::VerbsManager::QPId qp_id_{0};
  std::atomic<uint64_t> next_tx_id_{1};
  
  std::unordered_map<void*, std::shared_ptr<pccl::communicator::VerbsMemoryRegion>> registered_regions_;
  std::mutex regions_mutex_;
};

} // namespace pccl::communicator
