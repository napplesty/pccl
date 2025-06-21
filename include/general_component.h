#pragma once

#include "registered_memory.h"
#include "types.h"
#include <concepts>
#include <cstdint>
#include <vector>

namespace pccl {

template <typename T>
concept FuncLib = requires(T &component) {
  { component.get_component_flags() } -> std::same_as<ComponentTypeFlags>;
  { component.get_function_flags() } -> std::same_as<FunctionTypeFlags>;
};

template <typename T>
concept FuncContext = requires(T &component, int local_index) {
  { component.context_guard(local_index) } -> std::same_as<void>;
};

template <typename T>
concept FuncStream = requires(T &component, int index, int priority, void *stream) {
  { component.stream_guard(stream) } -> std::same_as<void>;
  { component.stream(index) } -> std::convertible_to<void *>;
  { component.stream(index, priority) } -> std::convertible_to<void *>;
};

template <typename T>
concept FuncMemory = requires(T &component, RegisteredMemory mem, HandleType handle) {
  { component.alloc(mem) } -> std::same_as<RegisteredMemory>;
  { component.free(mem) } -> std::same_as<void>;
  { component.export_handle(mem) } -> std::same_as<HandleType>;
  { component.import_handle(handle) } -> std::same_as<RegisteredMemory>;
  { component.unimport_handle(mem) } -> std::same_as<void>;
};

template <typename T>
concept FuncConnection = requires(T &component, HandleType handle) {
  { component.export_handle() } -> std::same_as<HandleType>;
  { component.connect(handle) } -> std::same_as<bool>;
  { component.set_rts(handle) } -> std::same_as<bool>;
  { component.set_rtr(handle) } -> std::same_as<bool>;
  { component.disconnect(handle) } -> std::same_as<bool>;
  { component.get_num_connected() } -> std::convertible_to<int>;
  { component.get_bandwidth() } -> std::convertible_to<uint64_t>;
  { component.get_latency() } -> std::convertible_to<uint64_t>;
};

template <typename T>
concept FuncRegMem = requires(T &component, RegisteredMemory mem) {
  { component.reg(mem) } -> std::same_as<RegisteredMemory>;
  { component.unreg(mem) } -> std::same_as<RegisteredMemory>;
};

template <typename T>
concept FuncPhaseSwitch = requires(T &component, int phase, std::string path) {
  { component.get_phase() } -> std::convertible_to<int>;
  { component.phase_reg(path, phase) } -> std::same_as<void>;
  { component.launch_phase_change(phase) } -> std::same_as<void>;
  { component.commit(phase) } -> std::same_as<void>;
};

template <typename T>
concept GeneralComponent = FuncLib<T>;

template <typename T>
concept DeviceComponent = FuncLib<T> && FuncContext<T> && FuncMemory<T> && FuncStream<T>;

template <typename T>
concept NetComponent = FuncLib<T> && FuncConnection<T> && FuncRegMem<T>;

template <typename T>
concept SwitchComponent = FuncLib<T> && FuncPhaseSwitch<T>;

class ComponentRegistry {
public:
  static ComponentRegistry &get_instance() {
    static ComponentRegistry instance;
    return instance;
  }

  template <GeneralComponent T> 
  void register_component(T *component_ptr) {
    ComponentTypeFlags component_flags = component_ptr->get_component_flags();
    FunctionTypeFlags function_flags = component_ptr->get_function_flags();
    uintptr_t ptr = component_ptr;
    components_.emplace(std::make_tuple(component_flags, function_flags), ptr);
  }

private:
  ComponentRegistry() = default;
  ~ComponentRegistry() = default;
  ComponentRegistry(const ComponentRegistry &) = delete;
  ComponentRegistry &operator=(const ComponentRegistry &) = delete;

private:
  std::map<ComponentTypeFlags, std::vector<FunctionTypeFlags>> self_component_types;
  std::map<ComponentTypeFlags, std::vector<PluginTypeFlags>> component_plugins;
  std::map<std::tuple<ComponentTypeFlags, FunctionTypeFlags>, uintptr_t> components_;
};

} // namespace pccl