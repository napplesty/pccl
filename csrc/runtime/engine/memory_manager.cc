#include "runtime/engine/memory_manager.h"
#include "utils/allocator.hpp"
#include "utils/logging.h"

namespace pccl::engine {

MemoryManager::MemoryManager() : initialized_(false) {
  std::memset(next_buffer_num_, 0, sizeof(next_buffer_num_));
}

MemoryManager::~MemoryManager() {
  shutdown();
}

bool MemoryManager::initialize(const DistributedMemoryConfig& config,
                 std::shared_ptr<MemoryManagerCommInterface> comm,
                 void* semaphore_buffer) {
  if (initialized_) {
    PCCL_LOG_WARN("MemoryManager already initialized");
    return false;
  }
  
  config_ = config;
  comm_interface_ = comm;
  semaphore_buffer_ = semaphore_buffer;
  
  if (!allocate_local_buffers(config)) {
    PCCL_LOG_ERROR("Failed to allocate local buffers");
    return false;
  }
  
  if (!setup_signal_buffer()) {
    PCCL_LOG_ERROR("Failed to setup signal buffer");
    return false;
  }
  
  if (comm_interface_ && !perform_buffer_registration()) {
    PCCL_LOG_ERROR("Failed to perform buffer registration");
    return false;
  }
  
  initialized_ = true;
  PCCL_LOG_INFO("MemoryManager initialized successfully");
  return true;
}

void MemoryManager::shutdown() {
  if (!initialized_) return;
  
  std::lock_guard<std::mutex> lock1(allocation_mutex_);
  std::lock_guard<std::mutex> lock2(workspace_mutex_);
  
  for (auto& workspace_pair : active_workspaces_) {
    deallocate_workspace(workspace_pair.second);
  }
  active_workspaces_.clear();
  
  for (auto& pool_pair : buffer_pools_) {
    for (auto& allocation : pool_pair.second) {
      if (allocation.base_address) {
        utils::Allocator::free(allocation.base_address, allocation.executor_type);
      }
    }
  }
  buffer_pools_.clear();
  
  if (signal_buffer_.base_address) {
    utils::Allocator::free(signal_buffer_.base_address, ExecutorType::CPU);
  }
  
  initialized_ = false;
  PCCL_LOG_INFO("MemoryManager shutdown completed");
}

WorkspaceHandle MemoryManager::get_workspace_for_operator(uint64_t operator_id,
                                            const std::vector<int>& ranks,
                                            const std::map<ExecutorType, size_t>& buffer_requirements) {
  if (!initialized_) {
    throw std::runtime_error("MemoryManager not initialized");
  }
  
  std::lock_guard<std::mutex> lock(workspace_mutex_);
  
  if (active_workspaces_.find(operator_id) != active_workspaces_.end()) {
    PCCL_LOG_WARN("Workspace for operator {} already exists", operator_id);
    return active_workspaces_[operator_id];
  }
  
  WorkspaceHandle handle;
  handle.operator_id = operator_id;
  handle.participant_ranks = ranks;
  handle.signal_buffer = signal_buffer_.base_address;
  
  std::lock_guard<std::mutex> alloc_lock(allocation_mutex_);
  
  for (const auto& requirement : buffer_requirements) {
    ExecutorType executor_type = requirement.first;
    size_t buffer_size = requirement.second;
    
    std::vector<GlobalBufferID> buffer_ids;
    int num_buffers = config_.buffers_per_executor.at(executor_type);
    
    for (int i = 0; i < num_buffers; ++i) {
      GlobalBufferID buffer_id = allocate_buffer(executor_type, buffer_size, operator_id);
      if (buffer_id.value == 0) {
        PCCL_LOG_ERROR("Failed to allocate buffer for executor type {}", static_cast<int>(executor_type));
        deallocate_workspace(handle);
        throw std::runtime_error("Buffer allocation failed");
      }
      buffer_ids.push_back(buffer_id);
    }
    
    handle.allocated_buffers[executor_type] = buffer_ids;
  }
  
  if (comm_interface_ && !ranks.empty()) {
    if (!sync_workspace_allocation(operator_id, ranks)) {
      PCCL_LOG_ERROR("Failed to sync workspace allocation for operator {}", operator_id);
      deallocate_workspace(handle);
      throw std::runtime_error("Workspace synchronization failed");
    }
    
    auto remote_infos = exchange_buffer_info(operator_id, ranks);
    for (const auto& remote_info : remote_infos) {
      int rank = remote_info.buffer_id.getRank();
      handle.remote_buffers[rank].push_back(remote_info.buffer_id);
    }
    
    if (!signal_workspace_ready(operator_id, ranks)) {
      PCCL_LOG_ERROR("Failed to signal workspace ready for operator {}", operator_id);
      deallocate_workspace(handle);
      throw std::runtime_error("Workspace signaling failed");
    }
  }
  
  active_workspaces_[operator_id] = handle;
  PCCL_LOG_DEBUG("Workspace allocated for operator {} with {} participant ranks",
                operator_id, ranks.size());
  
  return handle;
}

bool MemoryManager::post_sync_workspace(const WorkspaceHandle& handle) {
  if (!comm_interface_) {
    return true;
  }
  
  std::vector<int> other_ranks;
  for (int rank : handle.participant_ranks) {
    if (rank != config_.local_rank) {
      other_ranks.push_back(rank);
    }
  }
  
  if (other_ranks.empty()) {
    return true;
  }
  
  uint64_t sync_signal = handle.operator_id;
  return comm_interface_->broadcast_signal(other_ranks, sync_signal);
}

void MemoryManager::deallocate_workspace(const WorkspaceHandle& handle) {
  std::lock_guard<std::mutex> lock(allocation_mutex_);
  
  for (const auto& buffer_pair : handle.allocated_buffers) {
    for (const GlobalBufferID& buffer_id : buffer_pair.second) {
      deallocate_buffer(buffer_id);
    }
  }
  
  active_workspaces_.erase(handle.operator_id);
  PCCL_LOG_DEBUG("Workspace deallocated for operator {}", handle.operator_id);
}

void* MemoryManager::get_buffer_address(const GlobalBufferID& buffer_id) const {
  if (!initialized_) return nullptr;
  
  ExecutorType executor_type = buffer_id.getExecutorType();
  uint8_t buffer_num = buffer_id.getBufferNum();
  
  auto pool_it = buffer_pools_.find(executor_type);
  if (pool_it == buffer_pools_.end()) {
    return nullptr;
  }
  
  if (buffer_num >= pool_it->second.size()) {
    return nullptr;
  }
  
  return pool_it->second[buffer_num].base_address;
}

WorkspaceHandle MemoryManager::get_workspace(uint64_t operator_id) const {
  std::lock_guard<std::mutex> lock(workspace_mutex_);
  auto it = active_workspaces_.find(operator_id);
  if (it != active_workspaces_.end()) {
    return it->second;
  }
  return WorkspaceHandle{};
}

std::vector<GlobalBufferID> MemoryManager::get_workspace_buffers(uint64_t operator_id) const {
  std::lock_guard<std::mutex> lock(workspace_mutex_);
  auto it = active_workspaces_.find(operator_id);
  if (it == active_workspaces_.end()) {
    return {};
  }
  
  std::vector<GlobalBufferID> all_buffers;
  for (const auto& buffer_pair : it->second.allocated_buffers) {
    all_buffers.insert(all_buffers.end(),
                      buffer_pair.second.begin(),
                      buffer_pair.second.end());
  }
  return all_buffers;
}

bool MemoryManager::allocate_local_buffers(const DistributedMemoryConfig& config) {
  for (const auto& executor_config : config.buffers_per_executor) {
    ExecutorType executor_type = executor_config.first;
    int num_buffers = executor_config.second;
    size_t buffer_size = config.default_buffer_sizes.at(executor_type);
    
    std::vector<BufferAllocation> allocations;
    
    for (int i = 0; i < num_buffers; ++i) {
      void* base_addr = nullptr;
      if (!utils::Allocator::alloc(&base_addr, buffer_size, executor_type)) {
        PCCL_LOG_ERROR("Failed to allocate buffer for executor type {}",
                      static_cast<int>(executor_type));
        return false;
      }
      
      BufferAllocation allocation;
      allocation.base_address = base_addr;
      allocation.size = buffer_size;
      allocation.executor_type = executor_type;
      allocation.buffer_index = i;
      allocation.allocated = true;
      allocation.current_operator = 0;
      allocation.rkey = 0;
      
      allocations.push_back(allocation);
      PCCL_LOG_DEBUG("Allocated buffer {} for executor type {} at {}",
                    i, static_cast<int>(executor_type), base_addr);
    }
    
    buffer_pools_[executor_type] = allocations;
    next_buffer_num_[static_cast<int>(executor_type)] = num_buffers;
  }
  
  return true;
}

bool MemoryManager::setup_signal_buffer() {
  size_t signal_buffer_size = 1024 * 1024;
  
  if (!utils::Allocator::alloc(&signal_buffer_.base_address,
                              signal_buffer_size, ExecutorType::CPU)) {
    PCCL_LOG_ERROR("Failed to allocate signal buffer");
    return false;
  }
  
  signal_buffer_.size = signal_buffer_size;
  signal_buffer_.rkey = 0;
  
  if (comm_interface_) {
    if (!comm_interface_->register_memory_region(signal_buffer_.base_address,
                                                signal_buffer_size,
                                                signal_buffer_.rkey)) {
      PCCL_LOG_ERROR("Failed to register signal buffer");
      utils::Allocator::free(signal_buffer_.base_address, ExecutorType::CPU);
      return false;
    }
  }
  
  utils::Allocator::memset(signal_buffer_.base_address, 0, signal_buffer_size, ExecutorType::CPU);
  PCCL_LOG_DEBUG("Signal buffer allocated at {} with size {}",
                signal_buffer_.base_address, signal_buffer_size);
  
  return true;
}

GlobalBufferID MemoryManager::allocate_buffer(ExecutorType executor_type, size_t size, uint64_t operator_id) {
  auto pool_it = buffer_pools_.find(executor_type);
  if (pool_it == buffer_pools_.end()) {
    return GlobalBufferID{};
  }
  
  for (auto& allocation : pool_it->second) {
    if (!allocation.allocated && allocation.size >= size) {
      allocation.allocated = true;
      allocation.current_operator = operator_id;
      
      GlobalBufferID buffer_id(config_.local_rank, executor_type,
                              allocation.buffer_index, allocation.size);
      
      PCCL_LOG_DEBUG("Allocated buffer {} for operator {}",
                    allocation.buffer_index, operator_id);
      
      return buffer_id;
    }
  }
  
  PCCL_LOG_ERROR("No available buffer for executor type {} with size {}",
                static_cast<int>(executor_type), size);
  return GlobalBufferID{};
}

void MemoryManager::deallocate_buffer(const GlobalBufferID& buffer_id) {
  ExecutorType executor_type = buffer_id.getExecutorType();
  uint8_t buffer_num = buffer_id.getBufferNum();
  
  auto pool_it = buffer_pools_.find(executor_type);
  if (pool_it == buffer_pools_.end()) {
    return;
  }
  
  if (buffer_num >= pool_it->second.size()) {
    return;
  }
  
  auto& allocation = pool_it->second[buffer_num];
  allocation.allocated = false;
  allocation.current_operator = 0;
  
  if (allocation.base_address) {
    utils::Allocator::memset(allocation.base_address, 0, allocation.size, executor_type);
  }
  
  PCCL_LOG_DEBUG("Deallocated buffer {} for executor type {}",
                buffer_num, static_cast<int>(executor_type));
}

bool MemoryManager::sync_workspace_allocation(uint64_t operator_id, const std::vector<int>& ranks) {
  if (!comm_interface_) {
    return true;
  }
  
  size_t buffer_info_size = sizeof(GlobalBufferID) * 10;
  std::vector<uint8_t> send_buffer(buffer_info_size, 0);
  std::vector<uint8_t> recv_buffer(buffer_info_size * ranks.size(), 0);
  
  auto local_buffers = get_workspace_buffers(operator_id);
  if (local_buffers.size() * sizeof(GlobalBufferID) > buffer_info_size) {
    PCCL_LOG_ERROR("Buffer info size too small");
    return false;
  }
  
  std::memcpy(send_buffer.data(), local_buffers.data(),
              local_buffers.size() * sizeof(GlobalBufferID));
  
  if (!comm_interface_->allgather(send_buffer.data(), recv_buffer.data(),
                                 buffer_info_size)) {
    PCCL_LOG_ERROR("Failed to allgather buffer info");
    return false;
  }
  
  return true;
}

std::vector<MemoryManager::RemoteBufferInfo> MemoryManager::exchange_buffer_info(uint64_t operator_id, const std::vector<int>& ranks) {
  std::vector<RemoteBufferInfo> remote_infos;
  
  if (!comm_interface_) {
    return remote_infos;
  }
  
  for (int rank : ranks) {
    if (rank == config_.local_rank) continue;
    
    auto local_buffers = get_workspace_buffers(operator_id);
    for (const GlobalBufferID& buffer_id : local_buffers) {
      RemoteBufferInfo info;
      info.buffer_id = buffer_id;
      info.remote_addr = nullptr;
      info.remote_rkey = 0;
      
      remote_infos.push_back(info);
    }
  }
  
  return remote_infos;
}

bool MemoryManager::perform_buffer_registration() {
  if (!comm_interface_) {
    return true;
  }
  
  for (auto& pool_pair : buffer_pools_) {
    for (auto& allocation : pool_pair.second) {
      if (allocation.base_address && allocation.rkey == 0) {
        if (!comm_interface_->register_memory_region(allocation.base_address,
                                                    allocation.size,
                                                    allocation.rkey)) {
          PCCL_LOG_ERROR("Failed to register memory region for buffer");
          return false;
        }
      }
    }
  }
  
  return true;
}

bool MemoryManager::signal_workspace_ready(uint64_t operator_id, const std::vector<int>& ranks) {
  if (!comm_interface_) {
    return true;
  }
  
  return comm_interface_->barrier(ranks);
}

} // namespace pccl::engine
