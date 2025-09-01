#pragma once

#include <cstdint>
#include <string_view>

namespace pccl {

inline uint64_t fnv1a_64(const std::string_view text) {
  constexpr uint64_t prime = 0x00000100000001B3;
  constexpr uint64_t offset = 0xCBF29CE484222325;
  
  uint64_t hash = offset;
  for (char c : text) {
    hash ^= c;
    hash *= prime;
  }
  return hash;
}

} // namespace pccl



