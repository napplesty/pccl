#include "runtime/engine/memory_manager.h"
#include "runtime/api/repr.h"
#include "utils/allocator.h"
#include "utils/exception.hpp"
#include "utils/logging.h"
#include <cstdint>

namespace pccl::engine {

static int get_buffer_idx_in_signal_buffer(const GlobalBufferID &buffer) {
  return (int)buffer.getExecutorType() * max_buffers_per_type + buffer.getBufferIdx();
}

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
    
    signal_buffer.addr = utils::allocate(runtime::ExecutorType::CPU, num_signal_slots * sizeof(int64_t));
    signal_buffer.setExecutorType(runtime::ExecutorType::CPU);
    signal_buffer.setBufferIdx(0);
    signal_buffer.setBufferSize(num_signal_slots * sizeof(int64_t));
    signal_buffer.setRank(self_rank_);
    signal_buffer.shareable_handles["signal_buffer"] = "true";
    
    remote_signal_buffer_cache.addr = utils::allocate(runtime::ExecutorType::CPU, num_signal_slots * sizeof(int64_t));
    remote_signal_buffer_cache.setExecutorType(runtime::ExecutorType::CPU);
    remote_signal_buffer_cache.setBufferIdx(1);
    remote_signal_buffer_cache.setBufferSize(num_signal_slots * sizeof(int64_t));
    remote_signal_buffer_cache.setRank(self_rank_);
    
    global_buffers_[self_rank_].push_back(signal_buffer);
    global_buffers_[self_rank_].push_back(remote_signal_buffer_cache);

    for (auto &buffer_id : global_buffers_[self_rank_]) {
      PCCL_HOST_ASSERT(comm_interface_->registerMemoryRegion(buffer_id), "Fail to register memory region");
    }
    
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

  if (!comm_interface_) {
    PCCL_LOG_ERROR("MemoryManagerCommInterface not set");
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

            // register for scaling up
            if (host_sign_ == config.endpoint_configs.at("pccl.runtime.host_sign") &&
                remote_buffer.getExecutorType() == runtime::ExecutorType::CUDA) {
              remote_buffer.ipc_addr = utils::from_shareable(runtime::ExecutorType::CUDA, remote_buffer.shareable_handles.at("cuda_ipc_handle"));
              PCCL_LOG_DEBUG("Registered cuda ipc buffer: rank={}, idx={}, ipc_ptr={}, raddr={}", remote_buffer.getRank(), remote_buffer.getBufferIdx(), remote_buffer.ipc_addr, remote_buffer.addr);
            }

            if (global_buffers_.count(rank) == 0) {
              global_buffers_[rank] = {};
            }
            global_buffers_[rank].push_back(remote_buffer);
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
  return false;
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
  initialized_ = false;
  
  PCCL_LOG_INFO("MemoryManager shutdown completed");
}

WorkspaceHandle MemoryManager::get_workspace_for_operator(uint64_t operator_id, 
                                                          const std::vector<int>& ranks,
                                                          const std::map<runtime::ExecutorType, size_t>& buffer_requirements) {
  if (!initialized_) {
    PCCL_THROW(pccl::RuntimeException, "MemoryManager not initialized");
  }
  WorkspaceHandle handle{};
  handle.operator_id = operator_id;
  handle.participant_ranks = ranks;
  handle.buffers[self_rank_] = {};

  int *signal_buffer_ptr = (int *)signal_buffer.addr;

  for (auto &it : buffer_requirements) {
    size_t remain_nbytes = it.second;
    runtime::ExecutorType type = it.first;
    for (auto &buffer : global_buffers_[self_rank_]) {
      if (buffer.getExecutorType() != type) {
        continue;
      }
      if (signal_buffer_ptr[get_buffer_idx_in_signal_buffer(buffer)] == -1) {
        remain_nbytes -= std::min((int)remain_nbytes, buffer.getBufferSize());
        signal_buffer_ptr[get_buffer_idx_in_signal_buffer(buffer)] = operator_id;
        handle.buffers[self_rank_].push_back(buffer);
        if (remain_nbytes == 0) {
          break;
        }
      }
    }
  }
  return handle;
}

bool MemoryManager::post_sync_workspace(WorkspaceHandle& handle) {
  if (!comm_interface_) {
    return false;
  }
  return comm_interface_->syncWorkspaceAllocation(handle);
}

void MemoryManager::deallocate_workspace(WorkspaceHandle& handle) {
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
  auto it = global_buffers_.find(rank);
  if (it != global_buffers_.end()) {
    for (const auto& buffer : it->second) {
      if (buffer.getExecutorType() == executor_type && buffer.getBufferIdx() == buffer_idx) {
        return buffer;
      }
    }
  }
  return GlobalBufferID{};
}

void MemoryManager::setCommInterface(std::shared_ptr<MemoryManagerCommInterface> comm_interface) {
  comm_interface_ = comm_interface;
}

void MemoryManager::setupSignalSynchronization() {
  PCCL_LOG_DEBUG("Setting up signal synchronization mechanism");
  int64_t *ptr = (int64_t *)signal_buffer.addr;
  #pragma unroll 8
  for (int i = 0; i < num_signal_slots; i++) {
    ptr[i] = -1;
  }
}

void MemoryManager::releaseWorkspaceBuffers(WorkspaceHandle& handle) {
  int64_t *ptr = (int64_t *)signal_buffer.addr;
  for(auto &buffer : handle.buffers[self_rank_]) {
    ptr[get_buffer_idx_in_signal_buffer(buffer)] = -1;
  }
}

} // namespace pccl::engine
