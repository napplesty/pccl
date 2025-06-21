#include "general_component.h"
#include "communicator.h"
#include "endpoint.h"
#include "registered_memory.h"
#include "types.h"

namespace pccl {

Communicator::Communicator() {}

Communicator::~Communicator() {}

Endpoint Communicator::export_endpoint() {
  return Endpoint(*this);
}

void Communicator::import_endpoint(int rank, Endpoint &endpoint) {

}

ComponentTypeFlags Communicator::get_enabled_components() {
  return ComponentRegistry::get_instance().available_components();
}

PluginTypeFlags Communicator::get_enabled_plugins() {
  return ComponentRegistry::get_instance().available_plugins();
}

RegisteredMemory Communicator::get_lib_mem(int rank, TagId tag_id) {
  return RegisteredMemory(ComponentTypeFlags(), 0, tag_id);
}

RegisteredMemory Communicator::get_buffer_mem(int rank, TagId tag_id) {
  return RegisteredMemory(ComponentTypeFlags(), 0, tag_id);
}



} // namespace pccl