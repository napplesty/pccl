#pragma once

#include <runtime/api/repr.h>
#include <runtime/engine/memory_manager.h>
#include <runtime/engine/graph_executor.h>
#include <map>

namespace pccl::runtime {

struct RuntimeConfig {
  int local_rank;
  int world_size;
  std::map<engine::ExecutorType, int> buffers_per_executor;
  std::map<engine::ExecutorType, unsigned long long> default_buffer_sizes;
  std::map<std::string, std::string> extra_config;
};

bool initializeRuntime(const RuntimeConfig& config);

void shutdownRuntime();

bool executeGraph(const PrimitiveGrpah& graph, std::vector<int> &participants);

} // namespace pccl::runtime
