#pragma once

#include "config.h"
#include "runtime.h"

namespace pccl {

enum class SlotStatus : uint8_t {
  ALLOCATED = 0,
  FREE = 1,
};

struct LibPredefinedBufferSlot {
  uint64_t tag;
  SlotStatus status;
};

struct LibBufferMetaSlot {
  uint64_t tag;
  uint8_t handle[128];
  TransportFlags transport;
  SlotStatus status;
};

struct LibBufferSlot {
  uint64_t latest_version;
  LibPredefinedBufferSlot predefined_slot[Config::NUM_SLOT * 2];
  LibBufferMetaSlot meta_slot[Config::NUM_SLOT];
};

} // namespace pccl