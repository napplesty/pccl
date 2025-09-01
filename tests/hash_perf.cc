#pragma once

#include <chrono>
#include <iostream>
#include <cstdlib>
#include <cstring>
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

inline uint64_t fnv1a_64_optimized(const std::string_view text) {
    constexpr uint64_t prime = 0x00000100000001B3;
    constexpr uint64_t offset = 0xCBF29CE484222325;
    
    uint64_t hash = offset;
    const char* data = text.data();
    const size_t len = text.size();
    
    // 处理8字节块
    size_t i = 0;
    const size_t aligned_len = len & ~0x7; // 8字节对齐的长度
    
    for (; i < aligned_len; i += 8) {
        uint64_t chunk;
        // 一次处理8个字节
        memcpy(&chunk, data + i, 8);
        
        // 对每个字节进行处理（手动展开）
        for (int j = 0; j < 8; j++) {
            uint8_t byte = (chunk >> (j * 8)) & 0xFF;
            hash ^= byte;
            hash *= prime;
        }
    }
    
    // 处理剩余字节
    for (; i < len; i++) {
        hash ^= static_cast<uint8_t>(data[i]);
        hash *= prime;
    }
    
    return hash;
}

inline uint64_t fnv1a_64_fast(const std::string_view text) {
    constexpr uint64_t prime = 0x00000100000001B3;
    constexpr uint64_t offset = 0xCBF29CE484222325;
    
    uint64_t hash = offset;
    const char* data = text.data();
    const size_t len = text.size();
    
    // 处理主要块（16字节对齐）
    size_t i = 0;
    const size_t block_size = 16;
    const size_t main_blocks = len / block_size;
    
    for (size_t block = 0; block < main_blocks; block++) {
        // 一次处理16个字节（减少循环开销）
        for (int j = 0; j < block_size; j++) {
            hash ^= static_cast<uint8_t>(data[i++]);
            hash *= prime;
        }
    }
    
    // 处理剩余字节（使用Duff's device风格优化）
    size_t remaining = len % block_size;
    if (remaining > 0) {
        // 使用循环展开处理剩余字节
        while (remaining >= 8) {
            hash ^= static_cast<uint8_t>(data[i++]); hash *= prime;
            hash ^= static_cast<uint8_t>(data[i++]); hash *= prime;
            hash ^= static_cast<uint8_t>(data[i++]); hash *= prime;
            hash ^= static_cast<uint8_t>(data[i++]); hash *= prime;
            hash ^= static_cast<uint8_t>(data[i++]); hash *= prime;
            hash ^= static_cast<uint8_t>(data[i++]); hash *= prime;
            hash ^= static_cast<uint8_t>(data[i++]); hash *= prime;
            hash ^= static_cast<uint8_t>(data[i++]); hash *= prime;
            remaining -= 8;
        }
        
        // 处理最后几个字节
        while (remaining-- > 0) {
            hash ^= static_cast<uint8_t>(data[i++]);
            hash *= prime;
        }
    }
    
    return hash;
}

void benchmark_hashers() {
    std::string test_data(2000, 'a'); // 1MB测试数据
    
    auto start = std::chrono::high_resolution_clock::now();
    auto result1 = fnv1a_64(test_data);
    auto end1 = std::chrono::high_resolution_clock::now();
    
    auto result2 = fnv1a_64_optimized(test_data);
    auto end2 = std::chrono::high_resolution_clock::now();
    
    auto result3 = fnv1a_64_fast(test_data);
    auto end3 = std::chrono::high_resolution_clock::now();

    std::cout << result1 << result2 << result3 << std::endl;
    
    std::cout << "Original: " << std::chrono::duration_cast<std::chrono::microseconds>(end1 - start).count() << "μs\n";
    std::cout << "Optimized: " << std::chrono::duration_cast<std::chrono::microseconds>(end2 - end1).count() << "μs\n";
    std::cout << "Fast: " << std::chrono::duration_cast<std::chrono::microseconds>(end3 - end2).count() << "μs\n";
}

} // namespace pccl


int main() {
  pccl::benchmark_hashers();
}
