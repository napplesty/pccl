#pragma once
#include <runtime/engine/graph_executor.h>

namespace pccl::engine::host {

class HostExecutorManager;

HostExecutorManager* create_host_executor_manager(GraphBufferLayout* graph_layout, int num_threads);
void destroy_host_executor_manager(HostExecutorManager* manager);
bool initialize_host_executor(HostExecutorManager* manager);
void start_host_executor(HostExecutorManager* manager);
void stop_host_executor(HostExecutorManager* manager);
void wait_host_executor(HostExecutorManager* manager);

} // namespace pccl::engine::host
