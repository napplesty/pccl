#pragma once
#include <runtime/engine/graph_executor.h>
#include <runtime/communicator/channel.h>
#include <runtime/engine/memory_manager.h>
#include <thread>
#include <vector>
#include <atomic>

namespace pccl::engine::host {

class HostExecutorManager {
public:
  HostExecutorManager(GraphBufferLayout* graph_layout, 
                      int num_threads, 
                      ReadyQueueLayout* ready_queues,
                      int num_queues,
                      std::shared_ptr<communicator::ChannelManager> channel_manager,
                      std::shared_ptr<MemoryManager> memory_manager);
  ~HostExecutorManager();

  bool initialize();
  void launch();
  void wait();
  void stop();

private:
  void worker_thread(int thread_id);
  bool try_pop_primitive(ReadyQueueLayout* queue, OperatorLayout** primitive, int* execute_index);
  void execute_primitive(OperatorLayout* primitive, int execute_index);
  void execute_write_primitive(OperatorLayout* primitive, int execute_index, const PrimitiveConfig& config);
  void execute_notify_primitive(OperatorLayout* primitive, int execute_index, const PrimitiveConfig& config);
  void execute_get_notify_primitive(OperatorLayout* primitive, int execute_index, const PrimitiveConfig& config);
  void update_dependencies(OperatorLayout* primitive);
  void enqueue_primitive(OperatorLayout* primitive);
  bool try_push_primitive(ReadyQueueLayout* queue, OperatorLayout* primitive);
  void* get_signal_buffer_address(int buffer_idx, int rank);
  communicator::Endpoint create_endpoint_for_rank(int rank);

  GraphBufferLayout* graph_layout_;
  int num_threads_;
  ReadyQueueLayout* ready_queues_;
  int num_queues_;
  std::shared_ptr<communicator::ChannelManager> channel_manager_;
  std::shared_ptr<MemoryManager> memory_manager_;
  std::vector<std::thread> executors_;
  std::atomic<bool> running_;
};

} // namespace pccl::engine::host
