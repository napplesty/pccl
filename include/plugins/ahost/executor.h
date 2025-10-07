#pragma once
#include <runtime/engine/graph_executor.h>
#include <thread>

namespace pccl::engine::host {

class HostExecutorManager {
public:
  HostExecutorManager(GraphBufferLayout* graph_layout, 
                      int num_threadss, 
                      ReadyQueueLayout *ready_queues,
                      int num_queues);
  ~HostExecutorManager();
  bool initialize();
  void launch();
private:
  GraphBufferLayout* graph_layout_;
  int num_threads_;
  ReadyQueueLayout *ready_queues_;
  int num_queues_;
  std::vector<std::thread> executors;
};

bool initialize_host_executor(HostExecutorManager* manager);
void start_host_executor(HostExecutorManager* manager);
void stop_host_executor(HostExecutorManager* manager);
void wait_host_executor(HostExecutorManager* manager);

} // namespace pccl::engine::host
