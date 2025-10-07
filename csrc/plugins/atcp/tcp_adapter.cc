#include "plugins/atcp/tcp_adapter.h"
#include "plugins/atcp/tcp_utils.h"
#include "utils/logging.h"
#include <cstring>

namespace pccl::communicator {

TCPAdapter::TCPAdapter(const Endpoint& self, const Endpoint& peer)
  : self_endpoint_(self), peer_endpoint_(peer) {
  tcp_manager_ = std::make_unique<pccl::TcpManager>();
}

TCPAdapter::~TCPAdapter() {
  disconnect();
}

bool TCPAdapter::connect(const Endpoint& self, const Endpoint& peer) {
  try {
    std::string local_ip = self.attributes_.at("pccl.tcp.local_ip");
    std::string remote_ip = peer.attributes_.at("pccl.tcp.remote_ip");
    uint16_t remote_port = std::stoi(peer.attributes_.at("pccl.tcp.remote_port"));
    
    if (!tcp_manager_->initialize(local_ip, "pccl_token")) {
      PCCL_LOG_ERROR("Failed to initialize TCP manager");
      return false;
    }
    
    pccl::GID remote_gid = remote_ip + ":" + std::to_string(remote_port);
    tcp_manager_->registerGID(remote_gid, remote_ip, remote_port);
    
    conn_id_ = tcp_manager_->createConnection();
    if (conn_id_ == 0) {
      PCCL_LOG_ERROR("Failed to create TCP connection");
      return false;
    }
    
    qp_id_ = tcp_manager_->createQP(conn_id_);
    if (qp_id_ == 0) {
      PCCL_LOG_ERROR("Failed to create TCP QP");
      return false;
    }
    
    if (!tcp_manager_->modifyQPToInit(conn_id_, qp_id_)) {
      PCCL_LOG_ERROR("Failed to modify QP to INIT");
      return false;
    }
    
    if (!tcp_manager_->modifyQPToRTR(conn_id_, qp_id_, remote_gid)) {
      PCCL_LOG_ERROR("Failed to modify QP to RTR");
      return false;
    }
    
    if (!tcp_manager_->modifyQPToRTS(conn_id_, qp_id_)) {
      PCCL_LOG_ERROR("Failed to modify QP to RTS");
      return false;
    }
    
    PCCL_LOG_INFO("TCP adapter connected successfully");
    return true;
  } catch (const std::exception& e) {
    PCCL_LOG_ERROR("TCP connection failed: {}", e.what());
    return false;
  }
}

void TCPAdapter::disconnect() {
  if (tcp_manager_ && conn_id_ != 0) {
    tcp_manager_.reset();
    conn_id_ = 0;
    qp_id_ = 0;
  }
}

uint64_t TCPAdapter::prepSend(const MemRegion& dst, const MemRegion& src) {
  if (!connected()) {
    PCCL_LOG_ERROR("TCP adapter not connected");
    return 0;
  }
  
  auto wr = buildSendWR(dst, src);
  if (!tcp_manager_->postSend(conn_id_, qp_id_, &wr)) {
    PCCL_LOG_ERROR("Failed to post TCP send");
    return 0;
  }
  
  return next_tx_id_.fetch_add(1);
}

void TCPAdapter::postSend() {
}

void TCPAdapter::signal(uint64_t tx_mask) {
}

bool TCPAdapter::checkSignal(uint64_t tx_mask) {
  return true;
}

bool TCPAdapter::waitTx(uint64_t tx_id) {
  if (!connected()) {
    return false;
  }
  
  pccl::TcpWC wc;
  int retries = 0;
  const int max_retries = 1000;
  
  while (retries < max_retries) {
    int count = tcp_manager_->pollCQ(conn_id_, 1, &wc);
    if (count > 0) {
      return wc.status == pccl::TcpWCStatus::Success;
    }
    std::this_thread::sleep_for(std::chrono::microseconds(10));
    retries++;
  }
  
  PCCL_LOG_ERROR("TCP transaction timeout");
  return false;
}

bool TCPAdapter::flush() {
  return true;
}

bool TCPAdapter::connected() const {
  return conn_id_ != 0 && qp_id_ != 0;
}

NetMetrics TCPAdapter::getStats() const {
  return NetMetrics{};
}

void TCPAdapter::updateStats(const NetMetrics& stats) {
}

ChannelType TCPAdapter::getType() const {
  return ChannelType::TCP;
}

const Endpoint& TCPAdapter::getSelfEndpoint() const {
  return self_endpoint_;
}

const Endpoint& TCPAdapter::getPeerEndpoint() const {
  return peer_endpoint_;
}

pccl::TcpSendWR TCPAdapter::buildSendWR(const MemRegion& dst, const MemRegion& src) {
  pccl::TcpSendWR wr;
  auto sge = new pccl::TcpSGE[1];
  sge[0].addr = src.ptr_;
  sge[0].length = src.size_;
  sge[0].lkey = 0;
  
  wr.sg_list = sge;
  wr.num_sge = 1;
  wr.next = nullptr;
  
  return wr;
}

pccl::TcpWriteWR TCPAdapter::buildWriteWR(const MemRegion& dst, const MemRegion& src) {
  pccl::TcpWriteWR wr;
  auto sge = new pccl::TcpSGE[1];
  sge[0].addr = src.ptr_;
  sge[0].length = src.size_;
  sge[0].lkey = 0;
  
  wr.sg_list = sge;
  wr.num_sge = 1;
  wr.next = nullptr;
  
  pccl::TcpRemoteMRMetadata remote_mr;
  remote_mr.remote_gid = peer_endpoint_.attributes_.at("pccl.remote.gid");
  remote_mr.remote_addr = reinterpret_cast<uint64_t>(dst.ptr_);
  remote_mr.remote_rkey = std::stoul(peer_endpoint_.attributes_.at("pccl.remote.rkey"));
  wr.remote_mr = remote_mr;
  
  return wr;
}

pccl::TcpReadWR TCPAdapter::buildReadWR(const MemRegion& dst, const MemRegion& src) {
  pccl::TcpReadWR wr;
  auto sge = new pccl::TcpSGE[1];
  sge[0].addr = dst.ptr_;
  sge[0].length = dst.size_;
  sge[0].lkey = 0;
  
  wr.sg_list = sge;
  wr.num_sge = 1;
  wr.next = nullptr;
  
  pccl::TcpRemoteMRMetadata remote_mr;
  remote_mr.remote_gid = peer_endpoint_.attributes_.at("pccl.remote.gid");
  remote_mr.remote_addr = reinterpret_cast<uint64_t>(src.ptr_);
  remote_mr.remote_rkey = std::stoul(peer_endpoint_.attributes_.at("pccl.remote.rkey"));
  wr.remote_mr = remote_mr;
  
  return wr;
}

} // namespace pccl::communicator
