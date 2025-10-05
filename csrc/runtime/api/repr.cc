#include "runtime/api/repr.h"
#include <fstream>

namespace pccl::runtime {

nlohmann::json BufferConfig::toJson() const {
  nlohmann::json j;
  j["buffer_idx"] = buffer_idx;
  j["dtype"] = static_cast<int>(dtype);
  j["size"] = size;
  j["executor_type"] = static_cast<int>(executor_type);
  return j;
}

BufferConfig BufferConfig::fromJson(const nlohmann::json& j) {
  BufferConfig config;
  config.buffer_idx = j.value("buffer_idx", 0);
  config.dtype = static_cast<DataType>(j.value("dtype", 0));
  config.size = j.value("size", 0ULL);
  config.executor_type = static_cast<ExecutorType>(j.value("executor_type", 0));
  return config;
}

nlohmann::json ExecutorConfig::toJson() const {
  nlohmann::json j;
  j["executor_type"] = static_cast<int>(executor_type);
  j["num_total_executors"] = num_total_executors;
  return j;
}

ExecutorConfig ExecutorConfig::fromJson(const nlohmann::json& j) {
  ExecutorConfig config;
  config.executor_type = static_cast<ExecutorType>(j.value("executor_type", 0));
  config.num_total_executors = j.value("num_total_executors", 0);
  return config;
}

nlohmann::json PrimitiveConfig::toJson() const {
  nlohmann::json j;
  j["type"] = static_cast<int>(type);
  j["dtype"] = static_cast<int>(dtype);
  j["target_rank"] = target_rank;
  j["src_buffer_idx"] = src_buffer_idx;
  j["dst_buffer_idx"] = dst_buffer_idx;
  j["compute_op"] = static_cast<int>(compute_op);
  j["executor_type"] = static_cast<int>(executor_type);
  j["num_executors"] = num_executors;
  j["data_size"] = data_size;
  j["signal_value"] = signal_value;
  j["num_dependencies"] = num_dependencies;
  
  nlohmann::json followers_json = nlohmann::json::array();
  for (int i = 0; i < num_followers; ++i) {
    followers_json.push_back(followers[i]);
  }
  j["followers"] = followers_json;
  j["num_followers"] = num_followers;
  
  return j;
}

PrimitiveConfig PrimitiveConfig::fromJson(const nlohmann::json& j) {
  PrimitiveConfig config;
  config.type = static_cast<PrimitiveType>(j.value("type", 0));
  config.dtype = static_cast<DataType>(j.value("dtype", 0));
  config.target_rank = j.value("target_rank", 0);
  config.src_buffer_idx = j.value("src_buffer_idx", 0);
  config.dst_buffer_idx = j.value("dst_buffer_idx", 0);
  config.compute_op = static_cast<ComputeType>(j.value("compute_op", 0));
  config.executor_type = static_cast<ExecutorType>(j.value("executor_type", 0));
  config.num_executors = j.value("num_executors", 0);
  config.data_size = j.value("data_size", 0ULL);
  config.signal_value = j.value("signal_value", 0ULL);
  config.num_dependencies = j.value("num_dependencies", 0);
  
  auto followers_json = j.find("followers");
  if (followers_json != j.end() && followers_json->is_array()) {
    config.num_followers = 0;
    for (const auto& follower : *followers_json) {
      if (config.num_followers < 8) {
        config.followers[config.num_followers++] = follower;
      }
    }
  } else {
    config.num_followers = 0;
  }
  
  return config;
}

PrimitiveGrpah::PrimitiveGrpah(int rank) : rank_(rank) {}

PrimitiveGrpah::PrimitiveGrpah(const nlohmann::json& j) {
  rank_ = j.value("rank", 0);
  
  auto buffers_json = j.find("buffers");
  if (buffers_json != j.end() && buffers_json->is_array()) {
    for (const auto& buffer_json : *buffers_json) {
      buffers_.push_back(BufferConfig::fromJson(buffer_json));
    }
  }
  
  auto executors_json = j.find("executors");
  if (executors_json != j.end() && executors_json->is_array()) {
    for (const auto& executor_json : *executors_json) {
      executors_.push_back(ExecutorConfig::fromJson(executor_json));
    }
  }
  
  auto operators_json = j.find("operators");
  if (operators_json != j.end() && operators_json->is_array()) {
    for (const auto& op_json : *operators_json) {
      operators_.push_back(PrimitiveConfig::fromJson(op_json));
    }
  }
}

void PrimitiveGrpah::addBuffer(int idx, DataType dtype, unsigned long long size) {
  BufferConfig config;
  config.buffer_idx = idx;
  config.dtype = dtype;
  config.size = size;
  config.executor_type = ExecutorType::CPU;
  buffers_.push_back(config);
}

int PrimitiveGrpah::addOperator(const PrimitiveConfig& op) {
  operators_.push_back(op);
  return operators_.size() - 1;
}

void PrimitiveGrpah::addDependency(int from_op_id, int to_op_id) {
  if ((int)from_op_id >= 0 && from_op_id < (int)operators_.size() &&
      to_op_id >= 0 && to_op_id < (int)operators_.size()) {
    operators_[to_op_id].num_dependencies++;
    if (operators_[from_op_id].num_followers < 8) {
      operators_[from_op_id].followers[operators_[from_op_id].num_followers++] = to_op_id;
    }
  }
}

PrimitiveGrpah PrimitiveGrpah::loadFromFile(const std::filesystem::path& filename, int rank) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file: " + filename.string());
  }
  
  nlohmann::json j;
  file >> j;
  file.close();
  
  PrimitiveGrpah graph(rank);
  graph = PrimitiveGrpah(j);
  return graph;
}

PrimitiveGrpah PrimitiveGrpah::loadFromJson(const nlohmann::json& j) {
  return PrimitiveGrpah(j);
}

int PrimitiveGrpah::getRank() const {
  return rank_;
}

const std::vector<BufferConfig>& PrimitiveGrpah::getBuffers() const {
  return buffers_;
}

const std::vector<PrimitiveConfig>& PrimitiveGrpah::getOperators() const {
  return operators_;
}

std::vector<ExecutorType> PrimitiveGrpah::getExecutors() const {
  std::vector<ExecutorType> executor_types;
  for (const auto& executor : executors_) {
    executor_types.push_back(executor.executor_type);
  }
  return executor_types;
}

} // namespace pccl::runtime
