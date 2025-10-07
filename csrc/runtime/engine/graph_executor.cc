#include "runtime/engine/graph_executor.h"

namespace pccl::engine {

class GraphExecutor::Impl {
public:
  bool initialize(GraphBufferLayout* graph_layout, 
                 const std::map<ExecutorType, int> &executor_config,
                 std::map<std::string, std::string> &extra_params) {
    return false;
  }

  void issue() {
  }

  void wait() {
  }

  void initialize_ready_queues() {
  }
};

GraphExecutor::GraphExecutor() : impl_(new Impl()) {
}

GraphExecutor::~GraphExecutor() {
}

bool GraphExecutor::initialize(GraphBufferLayout* graph_layout, 
                              const std::map<ExecutorType, int> &executor_config,
                              std::map<std::string, std::string> &extra_params) {
  return impl_->initialize(graph_layout, executor_config, extra_params);
}

void GraphExecutor::issue() {
  impl_->issue();
}

void GraphExecutor::wait() {
  impl_->wait();
}

void GraphExecutor::initialize_ready_queues() {
  impl_->initialize_ready_queues();
}

} // namespace pccl::engine
