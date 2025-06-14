#pragma once

#include "types.h"
#include <optional>
#include <cstdint>

namespace pccl {

class RegisteredMemory {
public:
    RegisteredMemory(size_t sizes[(uint64_t)BufferType::BufferTypeEnd],
                     ComponentTypeFlags component_types,
                     PluginTypeFlags plugin_types);
    ~RegisteredMemory();

    template<BufferType buffer_type>
    std::optional<void *> get_buffer_ptr() {
        if constexpr (buffer_type >= BufferType::BufferTypeEnd) {
            return std::nullopt;
        }
        if (ptrs_[(uint64_t)buffer_type] == nullptr) {
            return std::nullopt;
        }
        return ptrs_[(uint64_t)buffer_type];
    }

    template<BufferType buffer_type>
    std::optional<size_t> get_buffer_size() {
        if constexpr (buffer_type >= BufferType::BufferTypeEnd) {
            return std::nullopt;
        }
        if (size_[(uint64_t)buffer_type] == 0) {
            return std::nullopt;
        }
        return size_[(uint64_t)buffer_type];
    }
    
private:
    void * ptrs_[(uint64_t)BufferType::BufferTypeEnd];
    size_t size_[(uint64_t)BufferType::BufferTypeEnd];
    const ComponentTypeFlags component_types_;
    const PluginTypeFlags plugin_types_;
};

};