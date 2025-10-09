#include "plugins/aroce/roce_adapter.h"
#include "plugins/aroce/roce_utils.h"
#include "utils/logging.h"
#include "utils/hex_utils.hpp"
#include <infiniband/verbs.h>

namespace pccl::communicator {



RoCEAdapter::RoCEAdapter(const Endpoint& self, const Endpoint& peer)
  : self_endpoint_(self), peer_endpoint_(peer) {
  utils::unmarshal_from_hex_str(&verbs_manager_, self.attributes_.at("pccl.roce.verbsManager.global"));
}

RoCEAdapter::~RoCEAdapter() {
  disconnect();
}

bool RoCEAdapter::connect(const Endpoint& self, const Endpoint& peer) {
  try {
    if (!self.attributes_.contains("pccl.roce.verbsManager.global")) {
      PCCL_LOG_ERROR("Failed to setup Verbs manager");
      return false;
    }
    
    pd_ = verbs_manager_->getPD();
    if (!pd_) {
      PCCL_LOG_ERROR("Failed to get protection domain");
      return false;
    }
    
    int self_rank = std::stoi(self.attributes_.at("pccl.runtime.rank"));
    int peer_rank = std::stoi(peer.attributes_.at("pccl.runtime.rank"));

    std::string qp_id_key = std::format("pccl.roce.{}.{}.qp_id", self_rank, peer_rank);
    std::string qp_key = std::format("pccl.roce.{}.{}.qp_num", peer_rank, self_rank);
    std::string lid_key = std::format("pccl.roce.{}.{}.lid", peer_rank, self_rank);
    std::string gid_key = std::format("pccl.roce.{}.{}.gid", peer_rank, self_rank);
    
    pccl::communicator::VerbsRemotePeerInfo remote_info;
    remote_info.qp_num = std::stoul(peer.attributes_.at(qp_key));
    remote_info.lid = std::stoul(peer.attributes_.at(lid_key));

    utils::unmarshal_from_hex_str(&conn_id_, self.attributes_.at("pccl.roce.conn_id.global"));
    utils::unmarshal_from_hex_str(&remote_info.gid, peer.attributes_.at(gid_key));
    utils::unmarshal_from_hex_str(&qp_id_, self.attributes_.at(qp_id_key));
    
    PCCL_LOG_DEBUG("RoCE adapter connecting");

    if (!verbs_manager_->connect(conn_id_, qp_id_, remote_info)) {
      PCCL_LOG_ERROR("Failed to connect RoCE QP");
      return false;
    }
    
    PCCL_LOG_INFO("RoCE adapter connected successfully");
    return true;
  } catch (const std::exception& e) {
    PCCL_LOG_ERROR("RoCE connection failed: {}", e.what());
    return false;
  }
}

void RoCEAdapter::disconnect() {
  if (verbs_manager_ && conn_id_ != 0) {
    verbs_manager_->disconnect(conn_id_, qp_id_);
    verbs_manager_->destroyConnection(conn_id_, qp_id_);
    conn_id_ = 0;
    qp_id_ = 0;
  }
  
  std::lock_guard<std::mutex> lock(regions_mutex_);
  registered_regions_.clear();
  pd_.reset();
}

uint64_t RoCEAdapter::prepSend(const MemRegion& dst, const MemRegion& src) {
  if (!connected()) {
    PCCL_LOG_ERROR("RoCE adapter not connected");
    return 0;
  }
  
  auto wr = buildWriteWR(dst, src);
  ibv_send_wr* bad_wr = nullptr;
  
  int result = verbs_manager_->postSend(conn_id_, qp_id_, &wr, &bad_wr);
  if (result != 0) {
    PCCL_LOG_ERROR("Failed to post RoCE send: {}", result);
    return 0;
  }
  
  uint64_t tx_id = next_tx_id_.fetch_add(1);
  return tx_id;
}

void RoCEAdapter::postSend() {
}

void RoCEAdapter::signal(uint64_t tx_mask) {
}

bool RoCEAdapter::checkSignal(uint64_t tx_mask) {
  return true;
}

bool RoCEAdapter::waitTx(uint64_t tx_id) {
  return waitForCompletion(tx_id);
}

bool RoCEAdapter::flush() {
  if (!connected()) {
    return false;
  }
  
  ibv_wc wc;
  int retries = 0;
  const int max_retries = 1000;
  
  while (retries < max_retries) {
    int count = verbs_manager_->pollCQ(conn_id_, 1, &wc);
    if (count > 0) {
      if (wc.status == IBV_WC_SUCCESS) {
        PCCL_LOG_DEBUG("RoCE flush completed successfully");
        return true;
      } else {
        PCCL_LOG_ERROR("RoCE completion error: {}", VerbsLib::getInstance().wcStatusStr(wc.status));
        return false;
      }
    }
    std::this_thread::sleep_for(std::chrono::microseconds(10));
    retries++;
  }
  
  PCCL_LOG_ERROR("RoCE flush timeout");
  return false;
}

bool RoCEAdapter::connected() const {
  return conn_id_ != 0 && qp_id_ != 0 && verbs_manager_->isConnected(conn_id_);
}

NetMetrics RoCEAdapter::getStats() const {
  return NetMetrics{};
}

void RoCEAdapter::updateStats(const NetMetrics& stats) {
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

bool RoCEAdapter::registerMemoryRegion(MemRegion& region) {
  if (!pd_) {
    PCCL_LOG_ERROR("Protection domain not initialized");
    return false;
  }
  
  try {
    int access = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | 
                 IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_ATOMIC;
    
    auto mr = std::make_shared<pccl::communicator::VerbsMemoryRegion>(
      *pd_, region.ptr_, region.size_, access);
    
    std::lock_guard<std::mutex> lock(regions_mutex_);
    registered_regions_[region.ptr_] = mr;

    region.lkey_ = mr->getLKey();
    region.rkey_ = mr->getRKey();
    
    PCCL_LOG_DEBUG("Registered RoCE memory region: addr={}, size={}, lkey={}, rkey={}", 
                  region.ptr_, region.size_, mr->getLKey(), mr->getRKey());
    
    return true;
  } catch (const std::exception& e) {
    PCCL_LOG_ERROR("Failed to register RoCE memory region: {}", e.what());
    return false;
  }
}

bool RoCEAdapter::deregisterMemoryRegion(MemRegion& region) {
  std::lock_guard<std::mutex> lock(regions_mutex_);
  
  auto it = registered_regions_.find(region.ptr_);
  if (it != registered_regions_.end()) {
    registered_regions_.erase(it);
    PCCL_LOG_DEBUG("Deregistered RoCE memory region: addr={}", region.ptr_);
    return true;
  }
  
  return false;
}

ibv_send_wr RoCEAdapter::buildSendWR(const MemRegion& dst, const MemRegion& src) {
  ibv_send_wr wr{};
  ibv_sge sge{};
  
  sge.addr = reinterpret_cast<uint64_t>(src.ptr_);
  sge.length = src.size_;
  sge.lkey = getLocalKey(src.ptr_);
  
  wr.wr_id = next_tx_id_.load();
  wr.next = nullptr;
  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_SEND;
  wr.send_flags = IBV_SEND_SIGNALED;
  
  return wr;
}

ibv_send_wr RoCEAdapter::buildWriteWR(const MemRegion& dst, const MemRegion& src) {
  ibv_send_wr wr{};
  ibv_sge sge{};
  
  sge.addr = reinterpret_cast<uint64_t>(src.ptr_);
  sge.length = src.size_;
  sge.lkey = getLocalKey(src.ptr_);
  
  wr.wr_id = next_tx_id_.load();
  wr.next = nullptr;
  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_RDMA_WRITE;
  wr.send_flags = IBV_SEND_SIGNALED;
  
  wr.wr.rdma.remote_addr = reinterpret_cast<uint64_t>(dst.ptr_);
  wr.wr.rdma.rkey = dst.rkey_;
  
  return wr;
}

ibv_send_wr RoCEAdapter::buildReadWR(const MemRegion& dst, const MemRegion& src) {
  ibv_send_wr wr{};
  ibv_sge sge{};
  
  sge.addr = reinterpret_cast<uint64_t>(dst.ptr_);
  sge.length = dst.size_;
  sge.lkey = getLocalKey(dst.ptr_);
  
  wr.wr_id = next_tx_id_.load();
  wr.next = nullptr;
  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_RDMA_READ;
  wr.send_flags = IBV_SEND_SIGNALED;
  
  wr.wr.rdma.remote_addr = reinterpret_cast<uint64_t>(src.ptr_);
  wr.wr.rdma.rkey = src.rkey_;
  
  return wr;
}

bool RoCEAdapter::waitForCompletion(uint64_t tx_id) {
  if (!connected()) {
    return false;
  }
  
  ibv_wc wc;
  int retries = 0;
  const int max_retries = 1000;
  
  while (retries < max_retries) {
    int count = verbs_manager_->pollCQ(conn_id_, 1, &wc);
    if (count > 0) {
      if (wc.wr_id == tx_id || wc.status == IBV_WC_SUCCESS) {
        PCCL_LOG_DEBUG("RoCE transaction completed: tx_id={}", tx_id);
        return true;
      } else {
        PCCL_LOG_ERROR("RoCE completion error for tx_id {}: {}", 
                      tx_id, VerbsLib::getInstance().wcStatusStr(wc.status));
        return false;
      }
    }
    
    std::this_thread::sleep_for(std::chrono::microseconds(1));
    retries++;
  }
  
  PCCL_LOG_ERROR("RoCE transaction timeout for TX ID: {}", tx_id);
  return false;
}

uint32_t RoCEAdapter::getLocalKey(void* addr) {
  std::lock_guard<std::mutex> lock(regions_mutex_);
  auto it = registered_regions_.find(addr);
  if (it != registered_regions_.end()) {
    return it->second->getLKey();
  }
  return 0;
}

} // namespace pccl::communicator
