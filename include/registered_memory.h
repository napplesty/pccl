#pragma once

#include "types.h"
#include <cstddef>
#include <memory>
#include <tuple>
#include <vector>

namespace pccl {

class RegisteredMemory {
public:
  RegisteredMemory(ComponentTypeFlags component_flags, size_t size, TagId tag);
  ~RegisteredMemory();

  HandleType export_handle();
  static RegisteredMemory import_handle(HandleType handle);
  void *get_ptr(ComponentTypeFlags flag);

public:
  using HandlePtr = std::shared_ptr<std::tuple<PluginTypeFlags, HandleType>>;
  TagId tag;
  ComponentTypeFlags component_flags;
  std::vector<HandlePtr> handles;
  std::vector<void *> ptrs;
  size_t size;
};

}; // namespace pccl
