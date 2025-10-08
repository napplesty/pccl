#include <iostream>
#include <map>
#include <string>

namespace pccl::debug {

inline void printMap(const std::map<std::string, std::string>& map) {
  for (const auto& pair : map) {
    std::cout << "\"" << pair.first << "\": \"" << pair.second << "\"" << std::endl;
  }
}

}