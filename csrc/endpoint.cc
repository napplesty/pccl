#include "endpoint.h"
#include "communicator.h"
#include "general_component.h"
#include "types.h"

namespace pccl {

HandleType Endpoint::export_handle() {
  HandleType handle;
  handle["host_hash"] = endpoint_host_hash;
  handle["pid_hash"] = endpoint_pid_hash;
  handle["lib_mem"] = endpoint_lib_mem.export_handle();
  handle["data_mem"] = endpoint_data_mem.export_handle();
  for (auto &[plugin_type, handle]: endpoint_plugin_handles) {
    handle["plugin_handles"][plugin_type.to_string()] = handle;
  }
  return handle;
} 

Endpoint::Endpoint(Communicator &communicator)
  : endpoint_host_hash(get_host_hash()),
    endpoint_pid_hash(get_pid_hash()),
    endpoint_lib_mem(communicator.get_lib_mem(communicator.get_rank(), 0)),
    endpoint_data_mem(communicator.get_buffer_mem(communicator.get_rank(), 0)) {
  for (auto *net_component: ComponentRegistry::get_instance().net_components()) {
    endpoint_plugin_handles.emplace_back(
      net_component->get_plugin_type(),
      net_component->export_handle()
    );
  }
}

Endpoint Endpoint::import_handle(const HandleType &handle) {
  Endpoint endpoint;
  endpoint.endpoint_host_hash = handle["host_hash"].get<uint64_t>();
  endpoint.endpoint_pid_hash = handle["pid_hash"].get<uint64_t>();
  endpoint.endpoint_lib_mem = RegisteredMemory::import_handle(handle["lib_mem"]);
  endpoint.endpoint_data_mem = RegisteredMemory::import_handle(handle["data_mem"]);
  for (auto &[plugin_type, handle]: handle["plugin_handles"].items()) {
    endpoint.endpoint_plugin_handles.emplace_back(PluginTypeFlags(plugin_type.data()), handle);
  }
  return endpoint;
}


} // namespace pccl