#pragma once

#include <string>

namespace pccl {

class VerbsConfig {
  VerbsConfig();
public:
  static VerbsConfig& getInstance() {
    static VerbsConfig config;
    return config;
  }

public:
  std::string lib_path;
  std::string device_name;
  int port_num;
  int gid_index;
  int qkey;

  int max_send_wr{1024};
  int max_recv_wr{1024};
  int max_send_sge{1};
  int max_recv_sge{1};
  int max_inline_data{0};
};

} // namespace pccl