#include "plugins/ahost/executor.h"

namespace pccl::engine::host {

HostExecutorManager::HostExecutorManager(GraphBufferLayout* graph_layout, 
                                        int num_threads, 
                                        ReadyQueueLayout *ready_queues,
                                        int num_queues)
  : graph_layout_(graph_layout),
    num_threads_(num_threads),
    ready_queues_(ready_queues),
    num_queues_(num_queues) {
}

HostExecutorManager::~HostExecutorManager() {
}

bool HostExecutorManager::initialize() {
  return true;
}

void HostExecutorManager::launch() {
}

void HostExecutorManager::wait() {
}

bool initialize_host_executor(HostExecutorManager* manager) {
  return manager->initialize();
}

void start_host_executor(HostExecutorManager* manager) {
  manager->launch();
}

void stop_host_executor(HostExecutorManager* manager) {
}

void wait_host_executor(HostExecutorManager* manager) {
}

} // namespace pccl::engine::host