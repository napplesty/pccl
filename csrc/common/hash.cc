#include <common/hash.h>

constexpr static unsigned long FNV_OFFSET_BASIS = 14695981039346656037ull;
constexpr static unsigned long FNV_PRIME = 1099511628211ull;

long engine_c::utils::hash(std::string_view str) {
  long hash = FNV_OFFSET_BASIS;
  
  for (size_t i = 0; i < str.size(); ++i) {
    hash ^= static_cast<unsigned char>(str[i]);
    hash *= FNV_PRIME;
  }
  
  return static_cast<long>(hash);
}

