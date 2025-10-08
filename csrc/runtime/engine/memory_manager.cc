#include "runtime/engine/memory_manager.h"
#include "runtime/api/repr.h"
#include "utils/allocator.h"
#include "utils/exception.hpp"
#include "utils/logging.h"
#include <cstdint>
#include <random>
#include <algorithm>

namespace pccl::engine {

nlohmann::json GlobalBufferID::toJson() const {
  nlohmann::json j;
  j["addr"] = reinterpret_cast<uint64_t>(addr);
  j["value"] = value;
  j["shareable_handles"] = shareable_handles;
  return j;
}

GlobalBufferID GlobalBufferID::fromJson(const nlohmann::json& json_data) {
  GlobalBufferID gbid;
  PCCL_HOST_ASSERT(json_data.contains("value") && json_data.contains("addr") && 
                   json_data.contains("shareable_handles"), "Invalid buffer JSON");
  
  uint64_t addr_value = json_data["addr"];
  gbid.addr = reinterpret_cast<void*>(addr_value);
  gbid.value = json_data["value"];
  gbid.shareable_handles = json_data["shareable_handles"].get<std::map<std::string, std::string>>();
  return gbid;
}

std::string GlobalBufferID::toString() const {
  return toJson().dump();
}

MemoryManager::MemoryManager() : self_rank_(-1), initialized_(false) {
}

MemoryManager::~MemoryManager() {
  shutdown();
}

bool MemoryManager::initialize(runtime::RuntimeConfig& runtime_config) {
  if (initialized_) {
    PCCL_LOG_WARN("MemoryManager already initialized");
    return true;
  }
  
  try {
    self_rank_ = runtime_config.rank;
    host_sign_ = runtime_config.endpoint_configs.at("pccl.runtime.host_sign");
    global_buffers_[self_rank_] = {};
    
    for (auto& executor_type : runtime_config.buffer_nums) {
      auto buffer_size = runtime_config.buffer_sizes.at(executor_type.first);
      
      for (int i = 0; i < executor_type.second; i++) {
        GlobalBufferID gid;
        void* ptr = utils::allocate(executor_type.first, buffer_size);
        
        gid.addr = ptr;
        gid.setBufferSize(buffer_size);
        gid.setBufferIdx(i);
        gid.setExecutorType(executor_type.first);
        gid.setRank(self_rank_);
        
        if (executor_type.first == runtime::ExecutorType::CUDA) {
          gid.shareable_handles["cuda_ipc_handle"] = 
            utils::get_shareable_handle(executor_type.first, ptr);
        }
        
        global_buffers_[self_rank_].push_back(gid);
        
        PCCL_LOG_DEBUG("Allocated buffer: rank={}, type={}, idx={}, size={}, addr={}", 
                      self_rank_, static_cast<int>(executor_type.first), i, 
                      buffer_size, reinterpret_cast<uint64_t>(ptr));
      }
    }
    
    signal_buffer.addr = utils::allocate(runtime::ExecutorType::CPU, 4096 * sizeof(uint64_t));
    signal_buffer.setExecutorType(runtime::ExecutorType::CPU);
    signal_buffer.setBufferIdx(0);
    signal_buffer.setBufferSize(4096 * sizeof(uint64_t));
    signal_buffer.setRank(self_rank_);
    signal_buffer.shareable_handles["signal_buffer"] = "true";
    
    remote_signal_buffer_cache.addr = utils::allocate(runtime::ExecutorType::CPU, 4096 * sizeof(uint64_t));
    remote_signal_buffer_cache.setExecutorType(runtime::ExecutorType::CPU);
    remote_signal_buffer_cache.setBufferIdx(1);
    remote_signal_buffer_cache.setBufferSize(4096 * sizeof(uint64_t));
    remote_signal_buffer_cache.setRank(self_rank_);
    
    global_buffers_[self_rank_].push_back(signal_buffer);
    global_buffers_[self_rank_].push_back(remote_signal_buffer_cache);
    
    initialized_ = true;
    PCCL_LOG_INFO("MemoryManager initialized for rank {}", self_rank_);
    return true;
    
  } catch (const std::exception& e) {
    PCCL_LOG_ERROR("MemoryManager initialization failed: {}", e.what());
    return false;
  }
}

bool MemoryManager::initialize_cluster(const std::map<int, runtime::RuntimeConfig>& cluster_configs) {
  if (!initialized_) {
    PCCL_LOG_ERROR("MemoryManager not initialized");
    return false;
  }
  
  try {
    for (const auto& [rank, config] : cluster_configs) {
      if (rank == self_rank_) continue;
      
      PCCL_LOG_DEBUG("Processing remote rank {} configuration", rank);
      
      for (const auto& [key, value] : config.endpoint_configs) {
        if (key.find("pccl.buffer.") == 0) {
          try {
            nlohmann::json buffer_json = nlohmann::json::parse(value);
            GlobalBufferID remote_buffer = GlobalBufferID::fromJson(buffer_json);
            if (host_sign_ == config.endpoint_configs.at("pccl.runtime.host_sign") &&
                remote_buffer.getExecutorType() == runtime::ExecutorType::CUDA) {
              remote_buffer.ipc_addr = utils::from_shareable(runtime::ExecutorType::CUDA, remote_buffer.shareable_handles.at("cuda_ipc_handle"));
              PCCL_LOG_DEBUG("Registered cuda ipc buffer: rank={}, idx={}, ipc_ptr={}, raddr={}", remote_buffer.getRank(), remote_buffer.getBufferIdx(), remote_buffer.ipc_addr, remote_buffer.addr);
            }
            
            if (registerRemoteBuffer(remote_buffer)) {
              PCCL_LOG_DEBUG("Registered remote buffer: rank={}, type={}, idx={}", 
                            rank, static_cast<int>(remote_buffer.getExecutorType()), 
                            remote_buffer.getBufferIdx());
            }
          } catch (const std::exception& e) {
            PCCL_LOG_WARN("Failed to parse buffer config for key {}: {}", key, e.what());
          }
        }
      }
    }
    
    setupSignalSynchronization();
    
    PCCL_LOG_INFO("MemoryManager cluster initialization completed");
    return true;
    
  } catch (const std::exception& e) {
    PCCL_LOG_ERROR("MemoryManager cluster initialization failed: {}", e.what());
    return false;
  }
}

bool MemoryManager::update_peer(const runtime::RuntimeConfig& peer_config) {
  std::lock_guard<std::mutex> lock(remote_mutex_);
  
  int peer_rank = peer_config.rank;
  PCCL_LOG_DEBUG("Updating peer configuration for rank {}", peer_rank);
  
  for (const auto& [key, value] : peer_config.endpoint_configs) {
    if (key.find("pccl.buffer.") == 0) {
      try {
        nlohmann::json buffer_json = nlohmann::json::parse(value);
        GlobalBufferID remote_buffer = GlobalBufferID::fromJson(buffer_json);
        
        bool found = false;
        for (auto& existing_buffer : remote_buffers_[peer_rank]) {
          if (existing_buffer.buffer_id.getBufferIdx() == remote_buffer.getBufferIdx() &&
              existing_buffer.buffer_id.getExecutorType() == remote_buffer.getExecutorType()) {
            existing_buffer.buffer_id = remote_buffer;
            found = true;
            break;
          }
        }
        
        if (!found) {
          RemoteBufferInfo info;
          info.buffer_id = remote_buffer;
          info.registered_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
          remote_buffers_[peer_rank].push_back(info);
        }
        
      } catch (const std::exception& e) {
        PCCL_LOG_WARN("Failed to update buffer config for key {}: {}", key, e.what());
      }
    }
  }
  
  return true;
}

void MemoryManager::shutdown() {
  std::lock_guard<std::mutex> lock1(workspace_mutex_);
  std::lock_guard<std::mutex> lock2(buffer_mutex_);
  
  for (auto& [rank, buffers] : global_buffers_) {
    for (auto& buffer : buffers) {
      if (buffer.ipc_addr) {
        utils::close_shareable_handle(buffer.getExecutorType(), buffer.ipc_addr);
      }
    }
  }
  
  global_buffers_.clear();
  active_workspaces_.clear();
  remote_buffers_.clear();
  initialized_ = false;
  
  PCCL_LOG_INFO("MemoryManager shutdown completed");
}

WorkspaceHandle MemoryManager::get_workspace_for_operator(uint64_t operator_id, 
                                                       const std::vector<int>& ranks,
                                                       const std::map<runtime::ExecutorType, size_t>& buffer_requirements) {
  if (!initialized_) {
    PCCL_THROW(pccl::RuntimeException, "MemoryManager not initialized");
  }
  
  WorkspaceHandle handle;
  handle.operator_id = operator_id;
  handle.participant_ranks = ranks;
  
  if (!synchronizeBufferAllocation(operator_id, ranks, buffer_requirements)) {
    PCCL_THROW(pccl::RuntimeException, "Failed to synchronize buffer allocation");
  }
  
  for (int rank : ranks) {
    std::vector<GlobalBufferID> rank_buffers;
    
    for (const auto& [executor_type, size] : buffer_requirements) {
      GlobalBufferID buffer = allocateBufferForOperator(rank, executor_type, size);
      if (buffer.addr == nullptr) {
        PCCL_THROW(pccl::RuntimeException, "Failed to allocate buffer for operator");
      }
      rank_buffers.push_back(buffer);
    }
    
    handle.buffers[rank] = rank_buffers;
  }
  
  {
    std::lock_guard<std::mutex> lock(workspace_mutex_);
    active_workspaces_[operator_id] = handle;
  }
  
  PCCL_LOG_DEBUG("Created workspace for operator {} with {} participants", 
                operator_id, ranks.size());
  
  return handle;
}

bool MemoryManager::post_sync_workspace(const WorkspaceHandle& handle) {
  return synchronizeWorkspace(handle);
}

void MemoryManager::deallocate_workspace(const WorkspaceHandle& handle) {
  releaseWorkspaceBuffers(handle);
  
  std::lock_guard<std::mutex> lock(workspace_mutex_);
  active_workspaces_.erase(handle.operator_id);
  
  PCCL_LOG_DEBUG("Deallocated workspace for operator {}", handle.operator_id);
}

WorkspaceHandle MemoryManager::get_workspace(uint64_t operator_id) {
  std::lock_guard<std::mutex> lock(workspace_mutex_);
  auto it = active_workspaces_.find(operator_id);
  if (it != active_workspaces_.end()) {
    return it->second;
  }
  return WorkspaceHandle{};
}

std::vector<GlobalBufferID> MemoryManager::getLocalBuffers() const {
  std::lock_guard<std::mutex> lock(buffer_mutex_);
  auto it = global_buffers_.find(self_rank_);
  if (it != global_buffers_.end()) {
    return it->second;
  }
  return {};
}

GlobalBufferID MemoryManager::getBuffer(int rank, runtime::ExecutorType executor_type, int buffer_idx) const {
  std::lock_guard<std::mutex> lock(buffer_mutex_);
  
  if (rank == self_rank_) {
    auto it = global_buffers_.find(rank);
    if (it != global_buffers_.end()) {
      for (const auto& buffer : it->second) {
        if (buffer.getExecutorType() == executor_type && buffer.getBufferIdx() == buffer_idx) {
          return buffer;
        }
      }
    }
  } else {
    auto it = remote_buffers_.find(rank);
    if (it != remote_buffers_.end()) {
      for (const auto& remote_info : it->second) {
        if (remote_info.buffer_id.getExecutorType() == executor_type && 
            remote_info.buffer_id.getBufferIdx() == buffer_idx) {
          return remote_info.buffer_id;
        }
      }
    }
  }
  
  return GlobalBufferID{};
}

void MemoryManager::setCommInterface(std::shared_ptr<MemoryManagerCommInterface> comm_interface) {
  comm_interface_ = comm_interface;
}

GlobalBufferID MemoryManager::allocateBufferForOperator(int rank, runtime::ExecutorType executor_type, size_t size) {
  if (rank == self_rank_) {
    for (const auto& buffer : global_buffers_[self_rank_]) {
      if (buffer.getExecutorType() == executor_type && buffer.getBufferSize() >= size) {
        return buffer;
      }
    }
    
    GlobalBufferID new_buffer;
    void* ptr = utils::allocate(executor_type, size);
    new_buffer.addr = ptr;
    new_buffer.setBufferSize(size);
    new_buffer.setBufferIdx(global_buffers_[self_rank_].size());
    new_buffer.setExecutorType(executor_type);
    new_buffer.setRank(self_rank_);
    
    global_buffers_[self_rank_].push_back(new_buffer);
    return new_buffer;
  }
  
  return getBuffer(rank, executor_type, 0);
}

bool MemoryManager::synchronizeBufferAllocation(uint64_t operator_id, 
                                              const std::vector<int>& ranks,
                                              const std::map<runtime::ExecutorType, size_t>& requirements) {
  if (!comm_interface_) {
    PCCL_LOG_WARN("No communication interface for buffer synchronization");
    return true;
  }
  
  int master_rank = getMasterRank(ranks);
  if (master_rank == self_rank_) {
    std::map<int, std::map<runtime::ExecutorType, std::vector<int>>> allocation;
    
    for (int rank : ranks) {
      std::map<runtime::ExecutorType, std::vector<int>> rank_allocation;
      int buffer_idx = 0;
      for (const auto& [exec_type, size] : requirements) {
        rank_allocation[exec_type].push_back(buffer_idx++);
      }
      allocation[rank] = rank_allocation;
    }
    
    WorkspaceHandle sync_handle;
    sync_handle.operator_id = operator_id;
    sync_handle.participant_ranks = ranks;
    sync_handle.metadata["allocation"] = "master_allocation";
    
    return comm_interface_->syncWorkspaceAllocation(sync_handle);
  } else {
    return comm_interface_->waitForSignal(operator_id);
  }
}

bool MemoryManager::registerRemoteBuffer(const GlobalBufferID& remote_buffer) {
  std::lock_guard<std::mutex> lock(remote_mutex_);
  
  int rank = remote_buffer.getRank();
  RemoteBufferInfo info;
  info.buffer_id = remote_buffer;
  info.registered_time = std::chrono::duration_cast<std::chrono::milliseconds>(
    std::chrono::system_clock::now().time_since_epoch()).count();
  
  bool exists = false;
  for (auto& existing_info : remote_buffers_[rank]) {
    if (existing_info.buffer_id.getBufferIdx() == remote_buffer.getBufferIdx() &&
        existing_info.buffer_id.getExecutorType() == remote_buffer.getExecutorType()) {
      existing_info = info;
      exists = true;
      break;
    }
  }
  
  if (!exists) {
    remote_buffers_[rank].push_back(info);
  }
  
  if (comm_interface_) {
    comm_interface_->registerMemoryRegion(remote_buffer);
  }
  
  return true;
}

void MemoryManager::setupSignalSynchronization() {
  PCCL_LOG_DEBUG("Setting up signal synchronization mechanism");
}

bool MemoryManager::synchronizeWorkspace(const WorkspaceHandle& handle) {
  if (!comm_interface_) {
    return true;
  }
  
  return comm_interface_->syncWorkspaceAllocation(handle);
}

void MemoryManager::releaseWorkspaceBuffers(const WorkspaceHandle& handle) {
  for (const auto& [rank, buffers] : handle.buffers) {
    if (rank == self_rank_) {
      for (const auto& buffer : buffers) {
        if (comm_interface_) {
          comm_interface_->deregisterMemoryRegion(buffer);
        }
      }
    }
  }
}

std::string MemoryManager::generateShmemKey(void* ptr) {
  std::random_device rd;
  std::mt19937_64 gen(rd());
  std::uniform_int_distribution<uint64_t> dis;
  return std::to_string(dis(gen)) + "_" + std::to_string(reinterpret_cast<uint64_t>(ptr));
}

int MemoryManager::getMasterRank(const std::vector<int>& ranks) {
  if (ranks.empty()) return -1;
  return *std::min_element(ranks.begin(), ranks.end());
}

} // namespace pccl::engine
