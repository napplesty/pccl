#pragma once

#include <string>
#include <string_view>
#include <unordered_map>
#include <set>
#include <vector>
#include <atomic>
#include <mutex>

namespace engine_c {

using GeneralType = short;
using PrimitiveType = GeneralType;
using DataType = GeneralType;
using ComputeType = GeneralType;
using DeviceType = GeneralType;
using ExecutorType = GeneralType;

struct TypeCompatibility {
  ExecutorType executor_type;
  GeneralType target_type;
  bool is_compatible;
};

class TypeRegistry {
public:
  static PrimitiveType registerPrimitive(std::string_view name);
  static DataType registerDataType(std::string_view name);
  static ComputeType registerComputeType(std::string_view name);
  static DeviceType registerDeviceType(std::string_view name);
  static ExecutorType registerExecutorType(std::string_view name);

  static void registerCompatibility(std::string_view executor_name, std::string_view type_name);

  static std::string_view getTypeName(GeneralType type);
  static GeneralType getTypeId(std::string_view name);

  static const std::set<GeneralType> &getCompatibleTypes(ExecutorType executor_type);
  static const std::set<ExecutorType> &getCompatibleExecutors(GeneralType type);

  static void clear();

private:
  TypeRegistry() = default;
  ~TypeRegistry() = default;
  TypeRegistry(const TypeRegistry &) = delete;
  TypeRegistry &operator=(const TypeRegistry &) = delete;

  static TypeRegistry &getInstance() {
    static TypeRegistry instance;
    return instance;
  }

  PrimitiveType _registerPrimitiveType(std::string_view name);
  DataType _registerDataType(std::string_view name);
  ComputeType _registerComputeType(std::string_view name);
  DeviceType _registerDeviceType(std::string_view name);
  ExecutorType _registerExecutorType(std::string_view name);
  
  void _registerCompatibilityInternal(ExecutorType executor_type, GeneralType target_type);
  
  std::string_view _getTypeName(GeneralType type) const;
  GeneralType _getTypeId(std::string_view name) const;

  mutable std::mutex mutex_;
  
  std::unordered_map<std::string, GeneralType> name_cache_;

  std::unordered_map<PrimitiveType, std::string> primitive_cache_;
  std::unordered_map<DataType, std::string> data_cache_;
  std::unordered_map<ComputeType, std::string> compute_cache_;
  std::unordered_map<DeviceType, std::string> device_cache_;
  std::unordered_map<ExecutorType, std::string> executor_cache_;
  
  std::unordered_map<ExecutorType, std::set<GeneralType>> compatibility_map_;
  std::unordered_map<GeneralType, std::set<ExecutorType>> reverse_compatibility_map_;
  
  std::atomic<GeneralType> next_type_id_{0};
};

}
