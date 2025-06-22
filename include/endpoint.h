#pragma once

#include "registered_memory.h"
#include "types.h"
#include <nlohmann/json.hpp>

namespace pccl {

class Communicator;

class Endpoint {
public:
  Endpoint() = default;
  Endpoint(Communicator &communicator);
  HandleType export_handle();
  static Endpoint import_handle(const HandleType &handle);

  inline uint64_t get_host_hash() const { return endpoint_host_hash; }
  inline uint64_t get_pid_hash() const { return endpoint_pid_hash; }

private:
  uint64_t endpoint_host_hash;
  uint64_t endpoint_pid_hash;
  std::vector<std::tuple<PluginTypeFlags, HandleType>> endpoint_plugin_handles;
  RegisteredMemory endpoint_lib_mem;
  RegisteredMemory endpoint_data_mem;
};

} // namespace pccl