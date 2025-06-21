#pragma once

#include "endpoint.h"
#include "registered_memory.h"
#include "types.h"
namespace pccl {

class Communicator {
public:
  Communicator();
  ~Communicator();

  void init();
  Endpoint export_handle();
  void import_handle(int rank, Endpoint &handle);
  ComponentTypeFlags get_enabled_components();
  PluginTypeFlags get_enabled_plugins();
  RegisteredMemory get_lib_mem();
  RegisteredMemory get_buffer_mem();
};

} // namespace pccl