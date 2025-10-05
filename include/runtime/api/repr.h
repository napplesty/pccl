#pragma once

#include <vector>
#include <filesystem>
#include <nlohmann/json.hpp>

namespace pccl::runtime {

enum class PrimitiveType {
  WRITE,
  COMPUTE,
  COPY,
  SIGNAL,
  WAITSIGNAL
};

enum class DataType {
  F32,
  F16,
  BF16,
};

enum class ComputeType {
  SUM,
  MAX,
  MIN,
  PROD
};

enum class ExecutorType {
  CPU,
  CUDA,
  LAST,
};

struct BufferConfig {
  int buffer_idx;
  DataType dtype;
  unsigned long long size;
  ExecutorType executor_type;
  
  nlohmann::json toJson() const;
  static BufferConfig fromJson(const nlohmann::json& j);
};

struct ExecutorConfig {
  ExecutorType executor_type;
  int num_total_executors;

  nlohmann::json toJson() const;
  static ExecutorConfig fromJson(const nlohmann::json& j);
};

struct PrimitiveConfig {
  PrimitiveType type;
  DataType dtype;
  int target_rank;
  int src_buffer_idx;
  int dst_buffer_idx;
  ComputeType compute_op;
  ExecutorType executor_type;
  int num_executors;
  unsigned long long data_size;
  unsigned long long signal_value;
  int num_dependencies;
  int followers[8];
  int num_followers;

  nlohmann::json toJson() const;
  static PrimitiveConfig fromJson(const nlohmann::json& j);
};

class PrimitiveGrpah {
private:
  int rank_;
  std::vector<BufferConfig> buffers_;
  std::vector<ExecutorConfig> executors_;
  std::vector<PrimitiveConfig> operators_;

public:
  PrimitiveGrpah(int rank);
  PrimitiveGrpah(const nlohmann::json& j);
  
  void addBuffer(int idx, DataType dtype, unsigned long long size);
  int addOperator(const PrimitiveConfig& op);
  void addDependency(int from_op_id, int to_op_id);
  
  static PrimitiveGrpah loadFromFile(const std::filesystem::path& filename, int rank);
  static PrimitiveGrpah loadFromJson(const nlohmann::json& j);
  
  int getRank() const;
  const std::vector<BufferConfig>& getBuffers() const;
  const std::vector<PrimitiveConfig>& getOperators() const;
  std::vector<ExecutorType> getExecutors() const;
};

} // namespace pccl::runtime
