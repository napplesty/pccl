#pragma once

#include <map>

#include "runtime.h"

namespace pccl {

struct Device::Impl {
  std::map<TransportFlags, NetworkAddress> address_;
  
};

struct Cluster::Impl {};

}  // namespace pccl