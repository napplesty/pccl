#pragma once

#include <cstdint>
#include <vector>
#include <map>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <runtime/api/repr.h>
#include <runtime/api/runtime.h>

namespace pccl::engine {

class GlobalBufferID {
public:
  void *ipc_addr = nullptr;
  void* addr;
  uint64_t value;
  std::map<std::string, std::string> shareable_handles;

  GlobalBufferID() : addr(nullptr), value(0) {}
  GlobalBufferID(uint32_t rank, runtime::ExecutorType executor_type, uint8_t buffer_idx, uint32_t buffer_size) {
    value = (static_cast<uint64_t>(rank) << 44) |
            (static_cast<uint64_t>(executor_type) << 40) |
            (static_cast<uint64_t>(buffer_idx) << 32) |
            (static_cast<uint64_t>(buffer_size));
  }

  int getRank() const { return (value >> 44) & 0xFFFFF; }
  runtime::ExecutorType getExecutorType() const { 
    return static_cast<runtime::ExecutorType>((value >> 40) & 0xF); 
  }
  int getBufferIdx() const { return (value >> 32) & 0xFF; }
  int getBufferSize() const { return value & 0xFFFFFFFF; }
  
  void setRank(int rank) {
    value = (value & ~(0xFFFFFULL << 44)) | (static_cast<uint64_t>(rank) << 44);
  }
  void setExecutorType(runtime::ExecutorType executor_type) {
    value = (value & ~(0xFULL << 40)) | (static_cast<uint64_t>(executor_type) << 40);
  }
  void setBufferIdx(uint8_t buffer_idx) {
    value = (value & ~(0xFFULL << 32)) | (static_cast<uint64_t>(buffer_idx) << 32);
  }
  void setBufferSize(uint32_t buffer_size) {
    value = (value & ~0xFFFFFFFFULL) | buffer_size;
  }

  nlohmann::json toJson() const;
  static GlobalBufferID fromJson(const nlohmann::json& json_data);
  
  std::string toString() const;
};

struct WorkspaceHandle {
  uint64_t operator_id;
  std::vector<int> participant_ranks;
  std::map<int, std::vector<GlobalBufferID>> buffers;
  std::map<std::string, std::string> metadata;
  
  WorkspaceHandle() : operator_id(0) {}
};

class MemoryManagerCommInterface {
public:
  virtual ~MemoryManagerCommInterface() = default;
  virtual bool registerMemoryRegion(const GlobalBufferID& buffer_id) = 0;
  virtual bool deregisterMemoryRegion(const GlobalBufferID& buffer_id) = 0;
  virtual bool syncWorkspaceAllocation(const WorkspaceHandle& handle) = 0;
  virtual bool waitForSignal(uint64_t signal_id, int timeout_ms = 5000) = 0;
  virtual bool sendSignal(uint64_t signal_id, int target_rank) = 0;
};

class MemoryManager {
public:
  MemoryManager();
  ~MemoryManager();

  bool initialize(runtime::RuntimeConfig& runtime_config);
  bool initialize_cluster(const std::map<int, runtime::RuntimeConfig>& cluster_configs);
  bool update_peer(const runtime::RuntimeConfig& peer_config);
  void shutdown();
  
  WorkspaceHandle get_workspace_for_operator(uint64_t operator_id, 
                                           const std::vector<int>& ranks,
                                           const std::map<runtime::ExecutorType, size_t>& buffer_requirements);
  bool post_sync_workspace(const WorkspaceHandle& handle);
  void deallocate_workspace(const WorkspaceHandle& handle);
  WorkspaceHandle get_workspace(uint64_t operator_id);
  
  std::vector<GlobalBufferID> getLocalBuffers() const;
  GlobalBufferID getBuffer(int rank, runtime::ExecutorType executor_type, int buffer_idx) const;
  
  void setCommInterface(std::shared_ptr<MemoryManagerCommInterface> comm_interface);

private:
  struct RemoteBufferInfo {
    GlobalBufferID buffer_id;
    std::string connection_info;
    uint64_t registered_time;
  };

  GlobalBufferID allocateBufferForOperator(int rank, runtime::ExecutorType executor_type, size_t size);
  bool synchronizeBufferAllocation(uint64_t operator_id, 
                                 const std::vector<int>& ranks,
                                 const std::map<runtime::ExecutorType, size_t>& requirements);
  bool registerRemoteBuffer(const GlobalBufferID& remote_buffer);
  void setupSignalSynchronization();
  bool synchronizeWorkspace(const WorkspaceHandle& handle);
  void releaseWorkspaceBuffers(const WorkspaceHandle& handle);
  std::string generateShmemKey(void* ptr);
  int getMasterRank(const std::vector<int>& ranks);
  
  std::shared_ptr<MemoryManagerCommInterface> comm_interface_;
  
  GlobalBufferID signal_buffer;
  GlobalBufferID remote_signal_buffer_cache;
  
  std::unordered_map<uint64_t, WorkspaceHandle> active_workspaces_;
  std::map<int, std::vector<GlobalBufferID>> global_buffers_;
  std::map<int, std::vector<RemoteBufferInfo>> remote_buffers_;
  
  std::set<int> ranks_in_a_host_;
  
  mutable std::mutex workspace_mutex_;
  mutable std::mutex buffer_mutex_;
  mutable std::mutex remote_mutex_;

  std::string host_sign_;
  int self_rank_;
  bool initialized_;
  std::atomic<uint64_t> next_operator_id_{1};
};

} // namespace pccl::engine
