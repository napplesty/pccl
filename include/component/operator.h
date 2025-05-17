#pragma once

#include "config.h"
#include "runtime.h"

namespace pccl {

struct Channels {
  DeviceHandle<MemoryChannel> memory_channels[Config::MAX_CHANNEL];
  DeviceHandle<PortChannel> port_channels[Config::MAX_CHANNEL];
};

struct Operation {
  OperationType type;
  ChannelType channelType;
  uint8_t nInputs;
  uint8_t nOutputs;

  union {
    uint8_t inputChannelIndexes[Config::MAX_CHANNEL_PER_OPERATION];
  };

  union {
    uint8_t outputChannelIndexes[Config::MAX_CHANNEL_PER_OPERATION];
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
  uint8_t nMemoryChannels;                                      // 1 bytes
  uint8_t nPortChannels;                                        // 1 bytes
  uint16_t nOperations;                                         // 2 bytes
  Operation operations[Config::MAX_OPERATION_PER_THREADBLOCK];  // 64 * 100 = 6400 bytes
};

}  // namespace pccl
