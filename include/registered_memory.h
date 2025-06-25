#pragma once

#include "types.h"
#include <cstddef>
#include <memory>
#include <tuple>
#include <vector>

namespace pccl {

class RegisteredMemory {
public:
  RegisteredMemory() = default;
  RegisteredMemory(ComponentTypeFlags component_flags, size_t size, TagId tag);
  ~RegisteredMemory();

  HandleType export_handle();
  static RegisteredMemory import_handle(HandleType handle);
  void *get_ptr(ComponentTypeFlags flag);
  inline TagId get_tag() const { return tag; }
  inline ComponentTypeFlags get_component_flags() const { return component_flags; }
  inline size_t get_size() const { return size; }

public:
  using HandlePtr = std::shared_ptr<std::tuple<PluginTypeFlags, HandleType>>;
  TagId tag;
  ComponentTypeFlags component_flags;
  std::vector<HandlePtr> handles;
  std::vector<void *> ptrs;
  size_t size;
};

}; // namespace pccl
