#pragma once

#include <runtime/communicator/channel.h>
#include <plugins/atcp/tcp_utils.h>
#include <memory>

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

private:
  pccl::TcpSendWR buildSendWR(const MemRegion& dst, const MemRegion& src);
  pccl::TcpWriteWR buildWriteWR(const MemRegion& dst, const MemRegion& src);
  pccl::TcpReadWR buildReadWR(const MemRegion& dst, const MemRegion& src);

  Endpoint self_endpoint_;
  Endpoint peer_endpoint_;
  std::unique_ptr<pccl::TcpManager> tcp_manager_;
  pccl::TcpManager::ConnectionId conn_id_{0};
  pccl::TcpManager::QPId qp_id_{0};
  std::atomic<uint64_t> next_tx_id_{1};
};

} // namespace pccl::communicator
