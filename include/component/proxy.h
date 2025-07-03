#pragma once

#include "types.h"
#include "registered_memory.h"
#include <cstddef>

namespace pccl {

struct alignas(16) ProxyTrigger {
  ProxyId proxy_id;
  OperatorId operator_id;
  bool has_value;
  OperationId operation_id;
};

struct ProxyComponentEnd {
  ProxyTrigger *head;
  ProxyTrigger *tail;
  ProxyTrigger *buffer;
  size_t size;
};

class ProxyFifo {
public:
  ProxyFifo(RegisteredMemory buffer);
  ProxyComponentEnd get_end(ComponentTypeFlags component_flag);
};

};