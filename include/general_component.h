#pragma once

#include "registered_memory.h"
#include "types.h"
#include <concepts>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace pccl {

struct FuncLib {
  virtual ComponentTypeFlags get_component_flags() const = 0;
  virtual FunctionTypeFlags get_function_flags() const = 0;
  virtual ~FuncLib() = default;
};

struct FuncContext {
  virtual void context_guard(int local_index) = 0;
  virtual ~FuncContext() = default;
};
  
struct FuncStream {
  virtual void stream_guard(void* stream) = 0;
  virtual void* stream(int index) = 0;
  virtual void* stream(int index, int priority) = 0;
  virtual ~FuncStream() = default;
};

struct FuncMemory {
  virtual RegisteredMemory alloc(RegisteredMemory mem) = 0;
  virtual void free(RegisteredMemory mem) = 0;
  virtual HandleType export_handle(RegisteredMemory mem) = 0;
  virtual RegisteredMemory import_handle(HandleType handle) = 0;
  virtual void unimport_handle(RegisteredMemory mem) = 0;
  virtual ~FuncMemory() = default;
};

struct FuncConnection {
  virtual PluginTypeFlags get_plugin_type() const = 0;
  virtual HandleType export_handle() = 0;
  virtual bool connect(HandleType handle) = 0;
  virtual bool set_rts(HandleType handle) = 0;
  virtual bool set_rtr(HandleType handle) = 0;
  virtual bool disconnect(HandleType handle) = 0;
  virtual int get_num_connected() const = 0;
  virtual uint64_t get_bandwidth() const = 0;
  virtual uint64_t get_latency() const = 0;
  virtual ~FuncConnection() = default;
};

struct FuncRegMem {
  virtual RegisteredMemory reg(RegisteredMemory mem) = 0;
  virtual RegisteredMemory unreg(RegisteredMemory mem) = 0;
  virtual ~FuncRegMem() = default;
};

struct FuncPhaseSwitch {
  virtual int get_phase() const = 0;
  virtual void phase_reg(const std::string& path, int phase) = 0;
  virtual void launch_phase_change(int phase) = 0;
  virtual void commit(int phase) = 0;
  virtual ~FuncPhaseSwitch() = default;
};

class GeneralComponent : public FuncLib {};
class DeviceComponent : public FuncLib, public FuncContext, public FuncMemory, public FuncStream {};
class NetComponent : public FuncLib, public FuncConnection, public FuncRegMem {};
class SwitchComponent : public FuncLib, public FuncPhaseSwitch {};

class ComponentRegistry {
public:
  static ComponentRegistry &get_instance() {
    static ComponentRegistry instance;
    return instance;
  }

  template <typename T>
  void register_component(T *component_ptr) {
    static_assert(std::is_convertible_v<T, GeneralComponent>, "Component must inherit from GeneralComponent");
    
    ComponentTypeFlags component_flags = component_ptr->get_component_flags();
    FunctionTypeFlags function_flags = component_ptr->get_function_flags();
    components_.emplace(std::make_tuple(component_flags, function_flags), 
                     static_cast<GeneralComponent*>(component_ptr));
  }

  ComponentTypeFlags available_components();
  PluginTypeFlags available_plugins();
  std::vector<NetComponent *> net_components();

private:
  ComponentRegistry() = default;
  ~ComponentRegistry() = default;
  ComponentRegistry(const ComponentRegistry &) = delete;
  ComponentRegistry &operator=(const ComponentRegistry &) = delete;

private:
  std::map<ComponentTypeFlags, std::vector<FunctionTypeFlags>> component_types;
  std::map<ComponentTypeFlags, std::vector<PluginTypeFlags>> component_plugins;
  std::map<std::tuple<ComponentTypeFlags, FunctionTypeFlags>, GeneralComponent*> components_;
};

} // namespace pccl