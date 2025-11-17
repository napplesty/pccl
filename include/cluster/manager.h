#pragma once

#include <map>
#include <base/registry.h>

namespace engine_c {

struct DeviceBufferMeta {
  int num_buffers_;
  long total_size_;
};

struct NodeMeta {
  int rank_;
  std::map<DeviceType, DeviceBufferMeta> buffer_infos_;
  std::map<std::string, std::string> endpoint_configs_;

  std::string serialize();
  static NodeMeta deserialize();
};

class ClusterManager {
public:
  ClusterManager(std::map<std::string, std::string>);

  bool initialize();
  
  const NodeMeta &getLocalMeta();

private:
  std::map<int, NodeMeta> node_infos_;
};

}





