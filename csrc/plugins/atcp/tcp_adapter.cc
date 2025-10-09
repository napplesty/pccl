#include "plugins/atcp/tcp_adapter.h"
#include "plugins/atcp/tcp_utils.h"
#include "utils/allocator.h"
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
      return false;
    }
    
    pccl::GID remote_gid = remote_ip + ":" + std::to_string(remote_port);
    tcp_manager_->registerGID(remote_gid, remote_ip, remote_port);
    
    conn_id_ = tcp_manager_->createConnection();
    if (conn_id_ == 0) {
      return false;
    }
    
    qp_id_ = tcp_manager_->createQP(conn_id_);
    if (qp_id_ == 0) {
      return false;
    }
    
    if (!tcp_manager_->modifyQPToInit(conn_id_, qp_id_)) {
      return false;
    }
    
    if (!tcp_manager_->modifyQPToRTR(conn_id_, qp_id_, remote_gid)) {
      return false;
    }
    
    if (!tcp_manager_->modifyQPToRTS(conn_id_, qp_id_)) {
      return false;
    }
    
    return true;
  } catch (const std::exception& e) {
    return false;
  }
}

void TCPAdapter::disconnect() {
  if (tcp_manager_ && conn_id_ != 0) {
    tcp_manager_.reset();
    conn_id_ = 0;
    qp_id_ = 0;
  }
  
  std::lock_guard<std::mutex> lock(regions_mutex_);
  registered_regions_.clear();
  
  std::lock_guard<std::mutex> tx_lock(transactions_mutex_);
  transactions_.clear();
}

uint64_t TCPAdapter::prepSend(const MemRegion& dst, const MemRegion& src) {
  if (!connected()) {
    return 0;
  }
  
  uint64_t tx_id = next_tx_id_.fetch_add(1);
  
  TCPTransaction transaction;
  transaction.id = tx_id;
  transaction.src = src;
  transaction.dst = dst;
  transaction.completed = false;
  
  if (!prepareDataForTransfer(src, transaction.staging_buffer)) {
    return 0;
  }
  
  auto wr = buildSendWR(dst, src);
  if (!tcp_manager_->postSend(conn_id_, qp_id_, &wr)) {
    return 0;
  }
  
  {
    std::lock_guard<std::mutex> lock(transactions_mutex_);
    transactions_[tx_id] = transaction;
  }
  
  return tx_id;
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
      if (wc.status == pccl::TcpWCStatus::Success) {
        std::lock_guard<std::mutex> lock(transactions_mutex_);
        auto it = transactions_.find(tx_id);
        if (it != transactions_.end()) {
          if (transferDataToDestination(it->second.dst, it->second.staging_buffer)) {
            it->second.completed = true;
            return true;
          }
        }
      }
      return false;
    }
    std::this_thread::sleep_for(std::chrono::microseconds(10));
    retries++;
  }
  
  return false;
}

bool TCPAdapter::flush() {
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
  
  return false;
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

bool TCPAdapter::registerMemoryRegion(MemRegion& region) {
  std::lock_guard<std::mutex> lock(regions_mutex_);
  
  TCPMemoryRegion tcp_region;
  tcp_region.addr = region.ptr_;
  tcp_region.length = region.size_;
  tcp_region.lkey = next_key_.fetch_add(1);
  tcp_region.rkey = next_key_.fetch_add(1);
  tcp_region.executor_type = runtime::ExecutorType::CPU;
  
  registered_regions_[region.ptr_] = tcp_region;
  
  return true;
}

bool TCPAdapter::deregisterMemoryRegion(MemRegion& region) {
  std::lock_guard<std::mutex> lock(regions_mutex_);
  
  auto it = registered_regions_.find(region.ptr_);
  if (it != registered_regions_.end()) {
    registered_regions_.erase(it);
    return true;
  }
  
  return false;
}

pccl::TcpSendWR TCPAdapter::buildSendWR(const MemRegion& dst, const MemRegion& src) {
  pccl::TcpSendWR wr;
  auto sge = new pccl::TcpSGE[1];
  
  std::lock_guard<std::mutex> lock(regions_mutex_);
  auto it = registered_regions_.find(src.ptr_);
  if (it != registered_regions_.end()) {
    sge[0].lkey = it->second.lkey;
  } else {
    sge[0].lkey = 0;
  }
  
  sge[0].addr = src.ptr_;
  sge[0].length = src.size_;
  
  wr.sg_list = sge;
  wr.num_sge = 1;
  wr.next = nullptr;
  
  return wr;
}

bool TCPAdapter::prepareDataForTransfer(const MemRegion& src, std::vector<char>& staging_buffer) {
  runtime::ExecutorType src_executor_type = runtime::ExecutorType::CPU;
  
  std::lock_guard<std::mutex> lock(regions_mutex_);
  auto it = registered_regions_.find(src.ptr_);
  if (it != registered_regions_.end()) {
    src_executor_type = it->second.executor_type;
  }
  
  if (src_executor_type == runtime::ExecutorType::CPU) {
    return true;
  }
  
  staging_buffer.resize(src.size_);
  
  return utils::generic_memcpy(src_executor_type, runtime::ExecutorType::CPU, 
                              src.ptr_, staging_buffer.data(), src.size_);
}

bool TCPAdapter::transferDataToDestination(const MemRegion& dst, const std::vector<char>& staging_buffer) {
  runtime::ExecutorType dst_executor_type = runtime::ExecutorType::CPU;
  
  std::lock_guard<std::mutex> lock(regions_mutex_);
  auto it = registered_regions_.find(dst.ptr_);
  if (it != registered_regions_.end()) {
    dst_executor_type = it->second.executor_type;
  }
  
  if (dst_executor_type == runtime::ExecutorType::CPU) {
    if (!staging_buffer.empty()) {
      memcpy(dst.ptr_, staging_buffer.data(), dst.size_);
    }
    return true;
  }
  
  if (staging_buffer.empty()) {
    return true;
  } else {
    void *buffer_ptr = const_cast<char *>(staging_buffer.data());
    return utils::generic_memcpy(runtime::ExecutorType::CPU, dst_executor_type, 
                                 buffer_ptr, dst.ptr_, dst.size_);
  }
}

}
