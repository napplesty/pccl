#pragma once

#include <cstdint>
#include <vector>
#include <map>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <runtime/api/repr.h>

namespace pccl::engine {

using runtime::ExecutorType;

struct GlobalBufferID {
  uint64_t value;

  GlobalBufferID() : value(0) {}
  GlobalBufferID(uint32_t rank, ExecutorType executor_type, uint8_t buffer_num, uint32_t buffer_size) {
    value = (static_cast<uint64_t>(rank) << 44) |
            (static_cast<uint64_t>(executor_type) << 40) |
            (static_cast<uint64_t>(buffer_num) << 32) |
            (static_cast<uint64_t>(buffer_size));
  }

  uint32_t getRank() const { return (value >> 44) & 0xFFFFF; }
  ExecutorType getExecutorType() const { return static_cast<ExecutorType>((value >> 40) & 0xF); }
  uint8_t getBufferNum() const { return (value >> 32) & 0xFF; }
  uint32_t getBufferSize() const { return value & 0xFFFFFFFF; }
};

struct SignalBufferEntry {
  uint64_t operator_id;
  int buffer_index;
};

struct WorkspaceHandle {
  uint64_t operator_id;
  std::vector<int> participant_ranks;
  std::map<ExecutorType, std::vector<GlobalBufferID>> allocated_buffers;
  std::map<int, std::vector<GlobalBufferID>> remote_buffers;
  void* signal_buffer;
};

struct DistributedMemoryConfig {
  int local_rank;
  int world_size;
  std::map<ExecutorType, int> buffers_per_executor;
  std::map<ExecutorType, unsigned long long> default_buffer_sizes;
  std::map<std::string, std::string> extra_config;
};

class MemoryManagerCommInterface {
public:
  virtual ~MemoryManagerCommInterface() = default;

  virtual bool register_memory_region(void* addr, size_t size, uint32_t& rkey) = 0;
  virtual bool deregister_memory_region(uint32_t rkey) = 0;
  
  virtual bool remote_write(int target_rank, void* local_addr, void* remote_addr, 
                           size_t size, uint32_t remote_rkey) = 0;
  virtual bool remote_read(int target_rank, void* local_addr, void* remote_addr, 
                          size_t size, uint32_t remote_rkey) = 0;
  
  virtual bool signal(int target_rank, uint64_t signal_value) = 0;
  virtual bool wait_signal(uint64_t signal_value, uint32_t timeout_ms = 5000) = 0;
  
  virtual bool broadcast_signal(const std::vector<int>& targets, uint64_t signal_value) = 0;
  virtual bool allgather(const void* send_data, void* recv_data, size_t data_size) = 0;
  
  virtual bool barrier(const std::vector<int>& participants) = 0;
};

class MemoryManager {
public:
  MemoryManager();
  ~MemoryManager();

  bool initialize(const DistributedMemoryConfig& config, 
                 std::shared_ptr<MemoryManagerCommInterface> comm = nullptr,
                 void* semaphore_buffer = nullptr);

  void shutdown();

  WorkspaceHandle get_workspace_for_operator(uint64_t operator_id, 
                                            const std::vector<int>& ranks,
                                            const std::map<ExecutorType, size_t>& buffer_requirements);

  bool post_sync_workspace(const WorkspaceHandle& handle);

  void deallocate_workspace(const WorkspaceHandle& handle);

  void* get_buffer_address(const GlobalBufferID& buffer_id) const;

  WorkspaceHandle get_workspace(uint64_t operator_id) const;

  std::vector<GlobalBufferID> get_workspace_buffers(uint64_t operator_id) const;

private:
  struct BufferAllocation {
    void* base_address;
    size_t size;
    ExecutorType executor_type;
    uint8_t buffer_index;
    bool allocated;
    uint64_t current_operator;
    uint32_t rkey;
  };

  struct SignalBuffer {
    void* base_address;
    size_t size;
    std::vector<SignalBufferEntry> entries;
    uint32_t rkey;
  };

  struct RemoteBufferInfo {
    GlobalBufferID buffer_id;
    void* remote_addr;
    uint32_t remote_rkey;
  };

  bool allocate_local_buffers(const DistributedMemoryConfig& config);
  bool setup_signal_buffer();
  GlobalBufferID allocate_buffer(ExecutorType executor_type, size_t size, uint64_t operator_id);
  void deallocate_buffer(const GlobalBufferID& buffer_id);
  bool sync_workspace_allocation(uint64_t operator_id, const std::vector<int>& ranks);
  std::vector<RemoteBufferInfo> exchange_buffer_info(uint64_t operator_id, const std::vector<int>& ranks);
  bool perform_buffer_registration();
  bool signal_workspace_ready(uint64_t operator_id, const std::vector<int>& ranks);

  DistributedMemoryConfig config_;
  std::shared_ptr<MemoryManagerCommInterface> comm_interface_;
  void* semaphore_buffer_;

  std::map<ExecutorType, std::vector<BufferAllocation>> buffer_pools_;
  SignalBuffer signal_buffer_;
  std::unordered_map<uint64_t, WorkspaceHandle> active_workspaces_;
  std::unordered_map<int, std::vector<RemoteBufferInfo>> remote_buffer_cache_;
  
  mutable std::mutex allocation_mutex_;
  mutable std::mutex workspace_mutex_;
  mutable std::mutex comm_mutex_;

  uint32_t next_buffer_num_[static_cast<int>(ExecutorType::CUDA) + 1];
  bool initialized_;
};

} // namespace pccl::engine
