#pragma once

#include "registered_memory.h"
#include "types.h"
#include <nlohmann/json.hpp>

namespace pccl {

class Communicator;

class Endpoint {
public:
  Endpoint(Communicator &communicator);

  HandleType expoet_handle();
  static Endpoint import_handle(const HandleType &handle);

private:
  uint64_t endpoint_host_hash;
  uint64_t endpoint_pid_hash;
  std::map<PluginTypeFlags, HandleType> endpoint_plugin_handles;
  RegisteredMemory endpoint_lib_mem;
  RegisteredMemory endpoint_data_mem;
};

} // namespace pccl