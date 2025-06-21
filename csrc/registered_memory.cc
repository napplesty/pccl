#include "registered_memory.h"
#include "general_component.h"
#include "types.h"
#include <stdexcept>
#include <format>

namespace pccl {

RegisteredMemory::RegisteredMemory(ComponentTypeFlags component_flags, size_t size, TagId tag)
  : tag(tag), component_flags(component_flags), size(size) {
  ptrs.resize(component_flags.count());
}

RegisteredMemory::~RegisteredMemory() {}

HandleType RegisteredMemory::export_handle() {
  HandleType handle;
  handle["tag"] = tag;
  handle["component_flags"] = component_flags.to_string();
  for (size_t i = 0; i < component_flags.count(); i++) {
    handle["ptr" + std::to_string(i)] = (uintptr_t)ptrs[i];;
  }
  for (HandlePtr ptr : handles) {
    std::string plugin_flag = std::get<0>(*ptr).to_string();
    handle[plugin_flag] = std::get<1>(*ptr);
  }
  handle["size"] = size;
  return handle;
};

void *RegisteredMemory::get_ptr(ComponentTypeFlags flag) {
  if ((flag&component_flags).count() == 0) {
    throw std::runtime_error(std::format("flag not matched {} and {}", flag.to_string(), component_flags.to_string()));
  }
  int pos = flag._Find_first(), self_pos = component_flags._Find_first();
  int index = 0;
  while (self_pos != pos) {
    self_pos = component_flags._Find_next(self_pos);
    index ++;
  }
  return ptrs[index];
}

RegisteredMemory RegisteredMemory::import_handle(HandleType handle) {
  RegisteredMemory mem;
  
  mem.tag = handle["tag"].get<TagId>();
  mem.component_flags = ComponentTypeFlags(handle["component_flags"].get<std::string>());
  mem.size = handle["size"].get<size_t>();

  const size_t ptr_count = mem.component_flags.count();
  mem.ptrs.resize(ptr_count);
  for (size_t i = 0; i < ptr_count; ++i) {
    const std::string key = "ptr" + std::to_string(i);
    if (!handle.contains(key)) {
      throw std::runtime_error("Invalid handle format: missing pointer entry");
    }
    mem.ptrs[i] = reinterpret_cast<void*>(handle[key].get<uintptr_t>());
  }

  for (auto& [k, v] : handle.items()) {
    if (k.starts_with("plugin_")) { // 假设插件句柄以 plugin_ 前缀标识
      auto plugin_flag = PluginTypeFlags(k.substr(7)); // 去除前缀
      auto handle_ptr = std::make_shared<std::tuple<PluginTypeFlags, HandleType>>(
          plugin_flag, v.get<HandleType>());
      mem.handles.push_back(handle_ptr);
    }
  }

  return mem;
}

} // namespace pccl