#include "plugins/aroce/roce_adapter.h"
#include <thread>
#include <chrono>

namespace pccl::communicator {

RoCEAdapter::RoCEAdapter(const Endpoint& local_endpoint, const Endpoint& remote_endpoint)
  : self_endpoint_(local_endpoint), peer_endpoint_(remote_endpoint) {
  
  current_stats_.bandwidth_ = 100000.0;
  current_stats_.latency_ = 1.0;
  current_stats_.loss_rate_ = 0.0;
  current_stats_.max_frag_ = 1024 * 1024;
  current_stats_.best_frag_ = 64 * 1024;
  current_stats_.updated_ = 0;
}

RoCEAdapter::~RoCEAdapter() {
  disconnect();
  cleanup();
}

bool RoCEAdapter::initializeVerbs() {
  try {
    verbs_manager_ = std::make_unique<VerbsManager>();
    
    VerbsManager::ConnectionConfig conn_config;
    conn_config.port_num = 1;
    conn_config.gid_index = 0;
    conn_config.max_qp_per_connection = 1;
    conn_config.cq_size = 100;
    
    if (!verbs_manager_->initialize()) {
      return false;
    }
    
    connection_id_ = verbs_manager_->createConnection(conn_config);
    if (connection_id_ == 0) {
      return false;
    }
    
    VerbsManager::QPConfig qp_config;
    qp_id_ = verbs_manager_->createQP(connection_id_, qp_config);
    if (qp_id_ == 0) {
      return false;
    }
    
    return true;
  } catch (const std::exception& e) {
    return false;
  }
}

bool RoCEAdapter::setupConnection() {
  try {
    auto local_metadata = verbs_manager_->getLocalMetadata(connection_id_, qp_id_);
    (void)local_metadata; // 消除未使用变量警告
    
    VerbsRemotePeerInfo remote_info;
    remote_info.qp_num = std::stoul(peer_endpoint_.attributes_.at("qp_num"));
    remote_info.lid = std::stoul(peer_endpoint_.attributes_.at("lid"));
    
    std::string gid_str = peer_endpoint_.attributes_.at("gid");
    if (gid_str.length() == 32) {
      for (int i = 0; i < 16; ++i) {
        std::string byte_str = gid_str.substr(i * 2, 2);
        remote_info.gid.raw[i] = static_cast<uint8_t>(std::stoul(byte_str, nullptr, 16));
      }
    }
    
    if (!verbs_manager_->modifyQPToInit(connection_id_, qp_id_)) {
      return false;
    }
    
    if (!verbs_manager_->modifyQPToRTR(connection_id_, qp_id_, 
          remote_info.qp_num, remote_info.lid, 0, 1, 0, &remote_info.gid)) {
      return false;
    }
    
    if (!verbs_manager_->modifyQPToRTS(connection_id_, qp_id_)) {
      return false;
    }
    
    if (!verbs_manager_->connect(connection_id_, remote_info)) {
      return false;
    }
    
    connected_ = true;
    return true;
  } catch (const std::exception& e) {
    return false;
  }
}

bool RoCEAdapter::registerMemoryRegions() {
  std::lock_guard<std::mutex> lock(tx_mutex_);
  
  for (const auto& tx_pair : active_transactions_) {
    const auto& tx = tx_pair.second;
    
    if (registered_mrs_.find(tx.src_addr) == registered_mrs_.end()) {
      auto pd = verbs_manager_->getPD();
      if (!pd) return false;
      
      try {
        auto mr = std::make_shared<VerbsMemoryRegion>(*pd, tx.src_addr, tx.data_size, 
                                                       IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
        registered_mrs_[tx.src_addr] = mr;
      } catch (const std::exception& e) {
        return false;
      }
    }
  }
  
  return true;
}

void RoCEAdapter::cleanup() {
  if (verbs_manager_ && connection_id_ != 0) {
    verbs_manager_->destroyConnection(connection_id_);
    connection_id_ = 0;
    qp_id_ = 0;
  }
  verbs_manager_.reset();
  registered_mrs_.clear();
  active_transactions_.clear();
  std::queue<uint64_t> empty_queue;
  std::swap(completed_transactions_, empty_queue);
}

bool RoCEAdapter::prepSend(const MemRegion& dst, const MemRegion& src, uint64_t tx_id) {
  std::lock_guard<std::mutex> lock(tx_mutex_);
  
  if (!connected_) {
    return false;
  }
  
  RoCETransaction tx;
  tx.tx_id = tx_id;
  tx.completed = false;
  tx.src_addr = src.ptr_;
  tx.dst_addr = dst.ptr_;
  tx.data_size = src.size_;
  tx.start_time = std::chrono::duration_cast<std::chrono::microseconds>(
    std::chrono::steady_clock::now().time_since_epoch()).count();
  
  active_transactions_[tx_id] = tx;
  return true;
}

bool RoCEAdapter::postSend(uint64_t tx_id) {
  std::lock_guard<std::mutex> lock(tx_mutex_);
  
  auto it = active_transactions_.find(tx_id);
  if (it == active_transactions_.end()) {
    return false;
  }
  
  if (!connected_ || !verbs_manager_) {
    return false;
  }
  
  try {
    ibv_send_wr wr{};
    ibv_send_wr* bad_wr = nullptr;
    ibv_sge sge{};
    
    auto mr_it = registered_mrs_.find(it->second.src_addr);
    if (mr_it == registered_mrs_.end()) {
      if (!registerMemoryRegions()) {
        return false;
      }
      mr_it = registered_mrs_.find(it->second.src_addr);
      if (mr_it == registered_mrs_.end()) {
        return false;
      }
    }
    
    sge.addr = reinterpret_cast<uint64_t>(it->second.src_addr);
    sge.length = it->second.data_size;
    sge.lkey = mr_it->second->getLKey();
    
    wr.wr_id = tx_id;
    wr.next = nullptr;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_RDMA_WRITE;
    wr.send_flags = IBV_SEND_SIGNALED;
    
    wr.wr.rdma.remote_addr = reinterpret_cast<uint64_t>(it->second.dst_addr);
    wr.wr.rdma.rkey = 0;
    
    if (!verbs_manager_->postSend(connection_id_, qp_id_, &wr, &bad_wr)) {
      return false;
    }
    
    return true;
  } catch (const std::exception& e) {
    return false;
  }
}

void RoCEAdapter::signal(uint64_t tx_mask) {
  signal_mask_.fetch_or(tx_mask, std::memory_order_release);
}

uint64_t RoCEAdapter::checkSignals() {
  uint64_t current_mask = signal_mask_.load(std::memory_order_acquire);
  uint64_t completed_mask = 0;
  
  std::lock_guard<std::mutex> lock(tx_mutex_);
  
  if (!verbs_manager_) return 0;
  
  ibv_wc wc[10];
  int num_completions = verbs_manager_->pollCQ(connection_id_, 10, wc);
  
  for (int i = 0; i < num_completions; ++i) {
    if (wc[i].status == IBV_WC_SUCCESS) {
      uint64_t completed_tx_id = wc[i].wr_id;
      auto tx_it = active_transactions_.find(completed_tx_id);
      if (tx_it != active_transactions_.end()) {
        tx_it->second.completed = true;
        completed_transactions_.push(completed_tx_id);
        
        if (current_mask & (1ULL << completed_tx_id)) {
          completed_mask |= (1ULL << completed_tx_id);
        }
      }
    }
  }
  
  while (!completed_transactions_.empty()) {
    uint64_t completed_tx_id = completed_transactions_.front();
    completed_transactions_.pop();
    
    auto tx_it = active_transactions_.find(completed_tx_id);
    if (tx_it != active_transactions_.end()) {
      active_transactions_.erase(tx_it);
    }
  }
  
  signal_mask_.fetch_and(~completed_mask, std::memory_order_release);
  return completed_mask;
}

bool RoCEAdapter::waitTx(uint64_t tx_id, uint32_t timeout) {
  auto start_time = std::chrono::steady_clock::now();
  auto timeout_duration = std::chrono::milliseconds(timeout);
  
  while (std::chrono::steady_clock::now() - start_time < timeout_duration) {
    uint64_t completed = checkSignals();
    
    if (completed & (1ULL << tx_id)) {
      return true;
    }
    
    std::this_thread::sleep_for(std::chrono::microseconds(100));
  }
  
  return false;
}

bool RoCEAdapter::flush(uint64_t tx_mask) {
  auto start_time = std::chrono::steady_clock::now();
  constexpr uint32_t timeout_ms = 5000;
  auto timeout_duration = std::chrono::milliseconds(timeout_ms);
  
  while (std::chrono::steady_clock::now() - start_time < timeout_duration) {
    uint64_t completed = checkSignals();
    
    if ((completed & tx_mask) == tx_mask) {
      return true;
    }
    
    std::this_thread::sleep_for(std::chrono::microseconds(100));
  }
  
  return false;
}

bool RoCEAdapter::connect(const Endpoint& self, const Endpoint& peer) {
  if (connected_) {
    disconnect();
  }
  
  self_endpoint_ = self;
  peer_endpoint_ = peer;
  
  if (!initializeVerbs()) {
    return false;
  }
  
  if (!setupConnection()) {
    cleanup();
    return false;
  }
  
  if (!registerMemoryRegions()) {
    cleanup();
    return false;
  }
  
  connected_ = true;
  return true;
}

void RoCEAdapter::disconnect() {
  connected_ = false;
  cleanup();
}

bool RoCEAdapter::connected() const {
  return connected_.load();
}

NetMetrics RoCEAdapter::getStats() const {
  std::lock_guard<std::mutex> lock(stats_mutex_);
  return current_stats_;
}

void RoCEAdapter::updateStats(const NetMetrics& stats) {
  std::lock_guard<std::mutex> lock(stats_mutex_);
  current_stats_ = stats;
  current_stats_.updated_ = std::chrono::duration_cast<std::chrono::microseconds>(
    std::chrono::steady_clock::now().time_since_epoch()).count();
}

bool RoCEAdapter::supportsMemType(uint32_t mem_type) const {
  return mem_type == 0 || mem_type == 1;
}

uint32_t RoCEAdapter::maxConcurrentTx() const {
  return 1024;
}

ChannelType RoCEAdapter::getType() const {
  return ChannelType::RDMA;
}

const Endpoint& RoCEAdapter::getSelfEndpoint() const {
  return self_endpoint_;
}

const Endpoint& RoCEAdapter::getPeerEndpoint() const {
  return peer_endpoint_;
}

} // namespace pccl::communicator
