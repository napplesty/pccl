#pragma once

#include "config.h"
#include "runtime.h"

namespace pccl {

struct Operation {
  OperationType type;
  ChannelType channelType;
  uint8_t nInputs;
  uint8_t nOutputs;

  union {
    uint8_t *inputBufferIds;
    uint8_t *outputBufferIds;
    TransportFlags *transportFlags;
  };

  union {
    struct {
      uint32_t deviceSyncerIndex;
      uint32_t nThreadBlocks;
    };
    struct {
      uint32_t *inputOffsets;
      uint32_t *outputOffsets;
      uint32_t srcOffset;
      uint32_t dstOffset;
      uint32_t size;
    };
    int netconf_phase;
  };
};

struct __attribute__((aligned(16))) DeviceExecutionPlan {
  uint8_t nMemoryChannels;                              // 1 bytes
  uint8_t nPortChannels;                                // 1 bytes
  uint32_t nOperations;                                 // 4 bytes
  Operation operations[MAX_OPERATION_PER_THREADBLOCK];  // 64 * 100 = 6400 bytes
};

}  // namespace pccl
