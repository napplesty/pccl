#include "base/registry.h"
#include <stdexcept>

namespace engine_c {

PrimitiveType TypeRegistry::registerPrimitive(std::string_view name) {
  return getInstance()._registerPrimitiveType(name);
}

DataType TypeRegistry::registerDataType(std::string_view name) {
  return getInstance()._registerDataType(name);
}

ComputeType TypeRegistry::registerComputeType(std::string_view name) {
  return getInstance()._registerComputeType(name);
}

DeviceType TypeRegistry::registerDeviceType(std::string_view name) {
  return getInstance()._registerDeviceType(name);
}

ExecutorType TypeRegistry::registerExecutorType(std::string_view name) {
  return getInstance()._registerExecutorType(name);
}

void TypeRegistry::registerCompatibility(std::string_view executor_name, std::string_view type_name) {
  auto& instance = getInstance();
  std::lock_guard<std::mutex> lock(instance.mutex_);
  
  auto executor_it = instance.name_cache_.find(std::string(executor_name));
  if (executor_it == instance.name_cache_.end()) {
    throw std::runtime_error("Executor type not found: " + std::string(executor_name));
  }
  
  auto type_it = instance.name_cache_.find(std::string(type_name));
  if (type_it == instance.name_cache_.end()) {
    throw std::runtime_error("Target type not found: " + std::string(type_name));
  }
  
  instance._registerCompatibilityInternal(static_cast<ExecutorType>(executor_it->second), type_it->second);
}

std::string_view TypeRegistry::getTypeName(GeneralType type) {
  return getInstance()._getTypeName(type);
}

GeneralType TypeRegistry::getTypeId(std::string_view name) {
  return getInstance()._getTypeId(name);
}

const std::set<GeneralType> &TypeRegistry::getCompatibleTypes(ExecutorType executor_type) {
  auto& instance = getInstance();
  std::lock_guard<std::mutex> lock(instance.mutex_);
  
  static const std::set<GeneralType> empty_set;
  auto it = instance.compatibility_map_.find(executor_type);
  return (it != instance.compatibility_map_.end()) ? it->second : empty_set;
}

const std::set<ExecutorType> &TypeRegistry::getCompatibleExecutors(GeneralType type) {
  auto& instance = getInstance();
  std::lock_guard<std::mutex> lock(instance.mutex_);
  
  static const std::set<ExecutorType> empty_set;
  auto it = instance.reverse_compatibility_map_.find(type);
  return (it != instance.reverse_compatibility_map_.end()) ? it->second : empty_set;
}

void TypeRegistry::clear() {
  auto& instance = getInstance();
  std::lock_guard<std::mutex> lock(instance.mutex_);
  
  instance.name_cache_.clear();
  instance.primitive_cache_.clear();
  instance.data_cache_.clear();
  instance.compute_cache_.clear();
  instance.device_cache_.clear();
  instance.executor_cache_.clear();
  instance.compatibility_map_.clear();
  instance.reverse_compatibility_map_.clear();
  instance.next_type_id_.store(0);
}

PrimitiveType TypeRegistry::_registerPrimitiveType(std::string_view name) {
  std::lock_guard<std::mutex> lock(mutex_);
  
  std::string name_str(name);
  auto it = name_cache_.find(name_str);
  if (it != name_cache_.end()) {
    return static_cast<PrimitiveType>(it->second);
  }
  
  GeneralType new_id = next_type_id_.fetch_add(1) + 1;
  name_cache_[name_str] = new_id;
  primitive_cache_[static_cast<PrimitiveType>(new_id)] = name_str;
  
  return static_cast<PrimitiveType>(new_id);
}

DataType TypeRegistry::_registerDataType(std::string_view name) {
  std::lock_guard<std::mutex> lock(mutex_);
  
  std::string name_str(name);
  auto it = name_cache_.find(name_str);
  if (it != name_cache_.end()) {
    return static_cast<DataType>(it->second);
  }
  
  GeneralType new_id = next_type_id_.fetch_add(1) + 1;
  name_cache_[name_str] = new_id;
  data_cache_[static_cast<DataType>(new_id)] = name_str;
  
  return static_cast<DataType>(new_id);
}

ComputeType TypeRegistry::_registerComputeType(std::string_view name) {
  std::lock_guard<std::mutex> lock(mutex_);
  
  std::string name_str(name);
  auto it = name_cache_.find(name_str);
  if (it != name_cache_.end()) {
    return static_cast<ComputeType>(it->second);
  }
  
  GeneralType new_id = next_type_id_.fetch_add(1) + 1;
  name_cache_[name_str] = new_id;
  compute_cache_[static_cast<ComputeType>(new_id)] = name_str;
  
  return static_cast<ComputeType>(new_id);
}

DeviceType TypeRegistry::_registerDeviceType(std::string_view name) {
  std::lock_guard<std::mutex> lock(mutex_);
  
  std::string name_str(name);
  auto it = name_cache_.find(name_str);
  if (it != name_cache_.end()) {
    return static_cast<DeviceType>(it->second);
  }
  
  GeneralType new_id = next_type_id_.fetch_add(1) + 1;
  name_cache_[name_str] = new_id;
  device_cache_[static_cast<DeviceType>(new_id)] = name_str;
  
  return static_cast<DeviceType>(new_id);
}

ExecutorType TypeRegistry::_registerExecutorType(std::string_view name) {
  std::lock_guard<std::mutex> lock(mutex_);
  
  std::string name_str(name);
  auto it = name_cache_.find(name_str);
  if (it != name_cache_.end()) {
    return static_cast<ExecutorType>(it->second);
  }
  
  GeneralType new_id = next_type_id_.fetch_add(1) + 1;
  name_cache_[name_str] = new_id;
  executor_cache_[static_cast<ExecutorType>(new_id)] = name_str;
  
  return static_cast<ExecutorType>(new_id);
}

void TypeRegistry::_registerCompatibilityInternal(ExecutorType executor_type, GeneralType target_type) {
  compatibility_map_[executor_type].insert(target_type);
  reverse_compatibility_map_[target_type].insert(executor_type);
}

std::string_view TypeRegistry::_getTypeName(GeneralType type) const {
  std::lock_guard<std::mutex> lock(mutex_);
  
  auto it = primitive_cache_.find(static_cast<PrimitiveType>(type));
  if (it != primitive_cache_.end()) {
    return it->second;
  }
  
  auto data_it = data_cache_.find(static_cast<DataType>(type));
  if (data_it != data_cache_.end()) {
    return data_it->second;
  }
  
  auto compute_it = compute_cache_.find(static_cast<ComputeType>(type));
  if (compute_it != compute_cache_.end()) {
    return compute_it->second;
  }
  
  auto device_it = device_cache_.find(static_cast<DeviceType>(type));
  if (device_it != device_cache_.end()) {
    return device_it->second;
  }
  
  auto executor_it = executor_cache_.find(static_cast<ExecutorType>(type));
  if (executor_it != executor_cache_.end()) {
    return executor_it->second;
  }
  
  return "";
}

GeneralType TypeRegistry::_getTypeId(std::string_view name) const {
  std::lock_guard<std::mutex> lock(mutex_);
  
  std::string name_str(name);
  auto it = name_cache_.find(name_str);
  return (it != name_cache_.end()) ? it->second : 0;
}

}
