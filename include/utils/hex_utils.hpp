#pragma once

#include <string>

namespace pccl::utils {

inline std::string marshal_to_hex_str(void *ptr, size_t nbyte) {
  static const char hex_chars[] = "0123456789ABCDEF";
  std::string result;
  result.reserve(nbyte * 2);
  
  unsigned char *byte_ptr = static_cast<unsigned char *>(ptr);
  for (size_t i = 0; i < nbyte; ++i) {
    unsigned char byte = byte_ptr[i];
    result.push_back(hex_chars[byte >> 4]);
    result.push_back(hex_chars[byte & 0x0F]);
  }
    
  return result;
}

inline void unmarshal_from_hex_str(void *ptr, const std::string &str) {
  unsigned char *byte_ptr = static_cast<unsigned char *>(ptr);
  size_t nbyte = str.length() / 2;

  auto hex_char_to_value = [](char c) -> unsigned char {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return c - 'a' + 10;
  };
    
  for (size_t i = 0; i < nbyte; ++i) {
    char high_nibble = str[2 * i];
    char low_nibble = str[2 * i + 1];
    byte_ptr[i] = (hex_char_to_value(high_nibble) << 4) | hex_char_to_value(low_nibble);
  }
}

}