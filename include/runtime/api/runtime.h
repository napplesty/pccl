#pragma once

#include <nlohmann/json.hpp>
#include <string>
#include <torch/extension.h>
#include <runtime/api/repr.h>
#include <map>
#include <vector>

namespace pccl::runtime {

struct RuntimeConfig {
  int rank;
  int world_size;
  std::map<ExecutorType, int> buffer_nums;
  std::map<ExecutorType, unsigned long long> buffer_sizes;
  std::map<std::string, std::string> endpoint_configs;

  std::string toJson();
  static RuntimeConfig fromJson(std::string json);
};

bool initializeRuntime(std::vector<RuntimeConfig>& runtime_configs, int rank, int world_size);

bool updatePeer(RuntimeConfig& peer_config);

void shutdownRuntime();

bool executeGraph(PrimitiveGrpah& graph, 
                  std::vector<int>& participants, 
                  torch::Tensor& input, 
                  torch::Tensor& output);

uint64_t generateOperatorId();

void registerCommunicationResources(RuntimeConfig& config);

std::map<ExecutorType, int> getExecutorConfig(const PrimitiveGrpah& graph);

} // namespace pccl::runtime
