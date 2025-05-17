#pragma once

#include "component/operator.h"
#include "config.h"
#include "cuda/memory_channel.h"
#include "cuda/packet.h"
#include "cuda/port_channel.h"
#include "cuda/reduce_kernel.h"
#include "cuda/sync.h"
#include "device.h"
#include "runtime.h"

namespace pccl {

template <typename T, typename PacketType = LL16Packet>

__global__ __launch_bounds__(512, 1) void executionKernel(int rank, T* input, T* output,
                                                          DeviceExecutionPlan* plan, int num_plan,
                                                          uint32_t flag) {
  extern __shared__ int4 sharedMem[];
  int sm_id = blockIdx.x;
  int tid = threadIdx.x;
  for (int plan_idx = sm_id; plan_idx < num_plan; plan_idx += gridDim.x) {
    DeviceExecutionPlan* localPlan = plan + plan_idx;
    for (size_t i = tid; i < sizeof(DeviceExecutionPlan) / sizeof(int4); i += blockDim.x) {
      sharedMem[i] = ((int4*)localPlan)[i];
    }
    __syncshm();
    localPlan = (DeviceExecutionPlan*)sharedMem;
    int nOperations = localPlan->nOperations;
    Operation* operations = localPlan->operations;
    DeviceHandle<MemoryChannel>* memoryChannels = localPlan->channels.memoryChannels;
    DeviceHandle<PortChannel>* portChannels = localPlan->channels.portChannels;
    for (int i = 0; i < nOperations; i++) {
      Operation& op = operations[i];
      if (op.type == OperationType::NOP) {
        __syncthreads();
      } else if (op.type == OperationType::BARRIER) {
        int nThreadBlocks = op.nThreadBlocks;
        int syncStateIndex = op.deviceSyncerIndex;
        deviceSyncers[syncStateIndex].sync(nThreadBlocks);
      } else if (op.type == OperationType::SIGNAL) {
        handleSignal(memoryChannels, portChannels, op.outputChannelIndexes, op.nOutputs,
                     op.channelType);
      } else if (op.type == OperationType::WAIT) {
        handleWait(memoryChannels, portChannels, op.inputChannelIndexes, op.nInputs,
                   op.channelType);
      } else if (op.type == OperationType::FLUSH) {
        handleFlush(portChannels, op.outputChannelIndexes, op.nOutputs);
      } else if (op.type == OperationType::PUT) {
        handlePut(memoryChannels, portChannels, op.outputChannelIndexes, op.outputOffsets,
                  op.inputOffsets, op.nOutputs, op.size, op.channelType);
      } else if (op.type == OperationType::GET) {
        handleGet(memoryChannels, op.inputChannelIndexes, op.outputOffsets, op.inputOffsets,
                  op.nInputs, op.size);
      } else if (op.type == OperationType::COPY) {
        handleCopy(dst, src, op.dstOffset, op.srcOffset, op.size);
      }
    }
  }
}
}  // namespace pccl