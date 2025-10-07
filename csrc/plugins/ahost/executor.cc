#include "runtime/engine/graph_executor.h"

namespace pccl::engine {

GraphExecutor::GraphExecutor() : graph_layout_(nullptr) {
}

GraphExecutor::~GraphExecutor() {
}

bool GraphExecutor::initialize(GraphBufferLayout* graph_layout, 
                              const std::map<ExecutorType, int> &executor_config,
                              std::map<std::string, std::string> &extra_params) {
  graph_layout_ = graph_layout;
  return true;
}

void GraphExecutor::issue() {
}

void GraphExecutor::wait() {
}

void GraphExecutor::initialize_ready_queues() {
}

} // namespace pccl::engine
