#include "general_component.h"
#include "types.h"

namespace pccl {

ComponentTypeFlags ComponentRegistry::available_components() {
  ComponentTypeFlags flag;
  for (auto &item : component_types) {
    flag |= item.first;
  }
  return flag;
}

PluginTypeFlags ComponentRegistry::available_plugins() {
  PluginTypeFlags flag;
  for (auto &item : component_plugins) {
    for (auto &plugin_flag : item.second) {
      flag |= plugin_flag;
    }
  }
  return flag;
}


} // namespace pccl