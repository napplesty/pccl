#pragma once

#include "endpoint.h"
#include "registered_memory.h"
#include "types.h"

namespace pccl {

class Communicator {
public:
  Communicator(int rank);
  ~Communicator();

  inline int get_rank() const { return rank; }

  Endpoint export_endpoint();
  void import_endpoint(int rank, Endpoint &endpoint);
  ComponentTypeFlags get_enabled_components();
  PluginTypeFlags get_enabled_plugins();
  RegisteredMemory get_lib_mem(int rank, TagId tag_id);
  RegisteredMemory get_buffer_mem(int rank, TagId tag_id);

private:
  int rank;
};

} // namespace pccl