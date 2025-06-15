#pragma once

#include "types.h"
#include <concepts>
#include <cstddef>

namespace pccl {

template <typename T>
concept FuncContext = requires(T &component, int local_index) {
  { component.context_guard(local_index) } -> std::same_as<void>;
};

template <typename T>
concept FuncStream =
    requires(T &component, int index, int priority, void *stream) {
      { component.stream_guard(stream) } -> std::same_as<void>;
      { component.stream(index) } -> std::convertible_to<void *>;
      { component.stream(index, priority) } -> std::convertible_to<void *>;
    };

template <typename T>
concept FuncMemory =
    requires(T &component, size_t size, ComponentTypeFlags flags, void *ptr) {
      { component.alloc(size, flags) } -> std::same_as(PcclPtr);
      { component.free(ptr, flags) } -> std::same_as<void>;
      { component.export_handle(ptr, size, flags) } -> std::same_as<PcclPtr>;
    };

template <typename T>
concept FuncRegMem =
    requires(T &component, void *ptr, size_t size, ComponentTypeFlags flags) {
      { component.reg(ptr, size, flags) } -> std::same_as<void>;
      { component.unreg(ptr, size, flags) } -> std::same_as<void>;
    };

} // namespace pccl