#pragma once

#include <map>
#include <base/registry.h>
#include <cluster/daemon.h>

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
  ClusterManager(std::map<std::string, std::string> &config);

  bool initializeSelf();
  bool initializeDaemonServer(NodeMeta &master_node);
  bool killDaemonServer();
  const NodeMeta &getLocalMeta();
  bool addOrUpdateNode(NodeMeta &node_meta);

private:
  std::map<int, NodeMeta> node_infos_;
  Daemon daemon_;
};

}





