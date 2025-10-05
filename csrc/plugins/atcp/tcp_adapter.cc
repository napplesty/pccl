#include "plugins/atcp/tcp_adapter.h"
#include <thread>
#include <chrono>

namespace pccl::communicator {

TCPAdapter::TCPAdapter(const Endpoint& local_endpoint, const Endpoint& remote_endpoint)
  : self_endpoint_(local_endpoint), peer_endpoint_(remote_endpoint) {
  
  current_stats_.bandwidth_ = 10000.0;
  current_stats_.latency_ = 10.0;
  current_stats_.loss_rate_ = 0.001;
  current_stats_.max_frag_ = 64 * 1024;
  current_stats_.best_frag_ = 8 * 1024;
  current_stats_.updated_ = 0;
}

TCPAdapter::~TCPAdapter() {
  disconnect();
  cleanup();
}

bool TCPAdapter::initializeTCP() {
  try {
    tcp_manager_ = std::make_unique<pccl::TcpManager>();
    
    std::string local_ip = self_endpoint_.attributes_.at("ip");
    std::string token = "tcp_token_" + std::to_string(reinterpret_cast<uintptr_t>(this));
    
    if (!tcp_manager_->initialize(local_ip, token)) {
      return false;
    }
    
    connection_id_ = tcp_manager_->createConnection();
    if (connection_id_ == 0) {
      return false;
    }
    
    pccl::TcpManager::QPConfig qp_config;
    qp_id_ = tcp_manager_->createQP(connection_id_, qp_config);
    if (qp_id_ == 0) {
      return false;
    }
    
    std::string peer_ip = peer_endpoint_.attributes_.at("ip");
    uint16_t peer_port = std::stoul(peer_endpoint_.attributes_.at("port"));
    std::string peer_gid = peer_endpoint_.attributes_.at("gid");
    tcp_manager_->registerGID(peer_gid, peer_ip, peer_port);
    
    return true;
  } catch (const std::exception& e) {
    return false;
  }
}

bool TCPAdapter::setupConnection() {
  try {
    if (!tcp_manager_->modifyQPToInit(connection_id_, qp_id_)) {
      return false;
    }
    
    std::string peer_gid = peer_endpoint_.attributes_.at("gid");
    if (!tcp_manager_->modifyQPToRTR(connection_id_, qp_id_, peer_gid)) {
      return false;
    }
    
    if (!tcp_manager_->modifyQPToRTS(connection_id_, qp_id_)) {
      return false;
    }
    
    connected_ = true;
    return true;
  } catch (const std::exception& e) {
    return false;
  }
}

void TCPAdapter::processCompletions() {
  if (!tcp_manager_) return;
  
  pccl::TcpWC wc[10];
  int num_completions = tcp_manager_->pollCQ(connection_id_, 10, wc);
  
  std::lock_guard<std::mutex> lock(tx_mutex_);
  for (int i = 0; i < num_completions; ++i) {
    if (wc[i].status == pccl::TcpWCStatus::Success) {
      uint64_t completed_tx_id = wc[i].byte_len;
      auto tx_it = active_transactions_.find(completed_tx_id);
      if (tx_it != active_transactions_.end()) {
        tx_it->second.completed = true;
        completed_transactions_.push(completed_tx_id);
      }
    }
  }
}

void TCPAdapter::cleanup() {
  if (tcp_manager_ && connection_id_ != 0) {
    tcp_manager_.reset();
    connection_id_ = 0;
    qp_id_ = 0;
  }
  active_transactions_.clear();
  std::queue<uint64_t> empty_queue;
  std::swap(completed_transactions_, empty_queue);
}

bool TCPAdapter::prepSend(const MemRegion& dst, const MemRegion& src, uint64_t tx_id) {
  std::lock_guard<std::mutex> lock(tx_mutex_);
  
  if (!connected_) {
    return false;
  }
  
  TCPTransaction tx;
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

bool TCPAdapter::postSend(uint64_t tx_id) {
  std::lock_guard<std::mutex> lock(tx_mutex_);
  
  auto it = active_transactions_.find(tx_id);
  if (it == active_transactions_.end()) {
    return false;
  }
  
  if (!connected_ || !tcp_manager_) {
    return false;
  }
  
  try {
    pccl::TcpSendWR wr{};
    pccl::TcpSGE sge{};
    
    sge.addr = it->second.src_addr;
    sge.length = it->second.data_size;
    sge.lkey = 0;
    
    wr.next = nullptr;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    
    if (!tcp_manager_->postSend(connection_id_, qp_id_, &wr)) {
      return false;
    }
    
    return true;
  } catch (const std::exception& e) {
    return false;
  }
}

void TCPAdapter::signal(uint64_t tx_mask) {
  signal_mask_.fetch_or(tx_mask, std::memory_order_release);
}

uint64_t TCPAdapter::checkSignals() {
  uint64_t current_mask = signal_mask_.load(std::memory_order_acquire);
  uint64_t completed_mask = 0;
  
  processCompletions();
  
  std::lock_guard<std::mutex> lock(tx_mutex_);
  while (!completed_transactions_.empty()) {
    uint64_t completed_tx_id = completed_transactions_.front();
    completed_transactions_.pop();
    
    if (current_mask & (1ULL << completed_tx_id)) {
      completed_mask |= (1ULL << completed_tx_id);
      
      auto tx_it = active_transactions_.find(completed_tx_id);
      if (tx_it != active_transactions_.end()) {
        active_transactions_.erase(tx_it);
      }
    }
  }
  
  signal_mask_.fetch_and(~completed_mask, std::memory_order_release);
  return completed_mask;
}

bool TCPAdapter::waitTx(uint64_t tx_id, uint32_t timeout) {
  auto start_time = std::chrono::steady_clock::now();
  auto timeout_duration = std::chrono::milliseconds(timeout);
  
  while (std::chrono::steady_clock::now() - start_time < timeout_duration) {
    processCompletions();
    
    std::lock_guard<std::mutex> lock(tx_mutex_);
    auto it = active_transactions_.find(tx_id);
    if (it != active_transactions_.end() && it->second.completed) {
      active_transactions_.erase(it);
      return true;
    }
    
    std::this_thread::sleep_for(std::chrono::microseconds(100));
  }
  
  return false;
}

bool TCPAdapter::flush(uint64_t tx_mask) {
  auto start_time = std::chrono::steady_clock::now();
  constexpr uint32_t timeout_ms = 5000;
  auto timeout_duration = std::chrono::milliseconds(timeout_ms);
  
  while (std::chrono::steady_clock::now() - start_time < timeout_duration) {
    processCompletions();
    
    std::lock_guard<std::mutex> lock(tx_mutex_);
    bool all_completed = true;
    
    for (const auto& tx_pair : active_transactions_) {
      if (tx_mask & (1ULL << tx_pair.first) && !tx_pair.second.completed) {
        all_completed = false;
        break;
      }
    }
    
    if (all_completed) {
      for (auto it = active_transactions_.begin(); it != active_transactions_.end();) {
        if (tx_mask & (1ULL << it->first)) {
          it = active_transactions_.erase(it);
        } else {
          ++it;
        }
      }
      return true;
    }
    
    std::this_thread::sleep_for(std::chrono::microseconds(100));
  }
  
  return false;
}

bool TCPAdapter::connect(const Endpoint& self, const Endpoint& peer) {
  if (connected_) {
    disconnect();
  }
  
  self_endpoint_ = self;
  peer_endpoint_ = peer;
  
  if (!initializeTCP()) {
    return false;
  }
  
  if (!setupConnection()) {
    cleanup();
    return false;
  }
  
  connected_ = true;
  return true;
}

void TCPAdapter::disconnect() {
  connected_ = false;
  cleanup();
}

bool TCPAdapter::connected() const {
  return connected_.load();
}

NetMetrics TCPAdapter::getStats() const {
  std::lock_guard<std::mutex> lock(stats_mutex_);
  return current_stats_;
}

void TCPAdapter::updateStats(const NetMetrics& stats) {
  std::lock_guard<std::mutex> lock(stats_mutex_);
  current_stats_ = stats;
  current_stats_.updated_ = std::chrono::duration_cast<std::chrono::microseconds>(
    std::chrono::steady_clock::now().time_since_epoch()).count();
}

bool TCPAdapter::supportsMemType(uint32_t mem_type) const {
  return mem_type == 0;
}

uint32_t TCPAdapter::maxConcurrentTx() const {
  return 256;
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

} // namespace pccl::communicator
