#pragma once

#include <string>
#include <map>
#include <base/registry.h>

namespace engine_c {
  
struct BufferMeta {
  DataType dtype_;
  DeviceType device_;
  int buffer_idx_;
  long size_;
};

struct GlobalBufferIdentifier {
  void *addr_;
  int num_max_slots_;
  DataType dtype_;
  DeviceType device_;
  std::map<std::string, std::string> shareable_handles;

  std::string serialize();
  static GlobalBufferIdentifier deserialize();
};

struct Workspace {
  std::vector<int> participants;
  std::map<int, std::vector<GlobalBufferIdentifier>> buffers;
};

}


