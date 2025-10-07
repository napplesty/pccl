#include "runtime/api/runtime.h"
#include "c10/cuda/CUDAStream.h"
#include "runtime/api/configs.h"
#include "runtime/communicator/channel.h"
#include "runtime/communicator/oob_comm.h"
#include "runtime/engine/memory_manager.h"
#include "runtime/engine/graph_executor.h"
#include "utils/logging.h"
#include <torch/torch.h>
#include <ATen/cuda/CUDAContext.h>
#include <memory>

namespace pccl::runtime {

static std::shared_ptr<PcclConfig> global_config = nullptr;
static std::unique_ptr<engine::MemoryManager> memory_manager = nullptr;
static std::unique_ptr<engine::GraphExecutor> graph_executor = nullptr;

std::shared_ptr<PcclConfig> get_global_config() {
  return global_config;
}

bool initializeRuntime(const RuntimeConfig& config) {
  try {
    global_config = std::make_shared<PcclConfig>();
    global_config->executor_stream = c10::cuda::getCurrentCUDAStream();
    
    engine::DistributedMemoryConfig mem_config;

    mem_config.local_rank = config.local_rank;
    mem_config.world_size = config.world_size;
    mem_config.buffers_per_executor = config.buffers_per_executor;
    mem_config.default_buffer_sizes = config.default_buffer_sizes;
    mem_config.extra_config = config.extra_config;
    
    memory_manager = std::make_unique<engine::MemoryManager>();

    if (!memory_manager->initialize(mem_config)) {
      PCCL_LOG_ERROR("Failed to initialize memory manager");
      return false;
    }
    
    graph_executor = std::make_unique<engine::GraphExecutor>();
    
    PCCL_LOG_INFO("Runtime initialized successfully");
    return true;
  } catch (const std::exception& e) {
    PCCL_LOG_ERROR("Runtime initialization failed: {}", e.what());
    return false;
  }

}

void shutdownRuntime() {

  if (graph_executor) {
    graph_executor->stop();
    graph_executor->wait();
    graph_executor.reset();
  }
  
  if (memory_manager) {
    memory_manager->shutdown();
    memory_manager.reset();
  }
  
  global_config.reset();
  
  PCCL_LOG_INFO("Runtime shutdown completed");
}

bool executeGraph(const PrimitiveGrpah& graph, std::vector<int> &participants, torch::Tensor &input, torch::Tensor &output) {
  if (!memory_manager || !graph_executor) {
    PCCL_LOG_ERROR("Runtime not initialized");
    return false;
  }
  
  try {
    auto buffers = graph.getBuffers();
    auto operators = graph.getOperators();
    auto executors = graph.getExecutors();
    
    std::map<engine::ExecutorType, int> executor_config;
    for (auto exec_type : executors) {
      executor_config[exec_type] = 1;
    }
    
    engine::GraphBufferLayout layout;
    layout.num_operators = operators.size();
    layout.num_queues = executors.size() * (executors.size() - 1);
    
    if (!graph_executor->initialize(&layout, executor_config)) {
      PCCL_LOG_ERROR("Failed to initialize graph executor");
      return false;
    }
    
    graph_executor->start();
    graph_executor->wait();
    
    PCCL_LOG_INFO("Graph execution completed successfully");
    return true;
  } catch (const std::exception& e) {
    PCCL_LOG_ERROR("Graph execution failed: {}", e.what());
    return false;
  }
}

} // namespace pccl::runtime
