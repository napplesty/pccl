#pragma once

#include <map>
#include <string>
#include <vector>
#include <base/registry.h>

namespace engine_c {

struct ChunkMeta {
  DataType dtype_;
  void *addr_;
  long num_elems_;
};
  
struct BufferMeta {
  DeviceType device_;
  int buffer_idx_;
  long size_;
};

struct GlobalBufferIdentifier {
  void *addr_;
  int num_max_slots_;
  DeviceType device_;
  long size_;
  std::map<std::string, std::string> shareable_handles_;

  std::string serialize();
  static GlobalBufferIdentifier deserialize();
};

struct DataWorkspaceMeta {
  std::vector<int> participants_;
  std::map<int, std::vector<GlobalBufferIdentifier>> buffers_;
};

class BufferManager {
public:
  BufferManager();

  void regBuffer(void *buf, long size, 
                 int *signals, long slots,
                 int rank, GeneralType type);

  std::tuple<void *, void *> getDevBuffer(GeneralType type);
        
  void regCommResource(void *buf, GeneralType comm_type);
            
  DataWorkspaceMeta &assignBuffer(long op_id, 
                                  long size, 
                                  GeneralType type,
                                  std::vector<int> &participants);
  
};

}


