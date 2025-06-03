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

// CUDA specific headers, useful for __half, __nv_bfloat16, vector types etc.
#include <cuda_bf16.h> // If bfloat16 is used
#include <cuda_fp16.h>

// PCCL_CUDA_DEVICE_INLINE should be defined in a common PCCL header (e.g.,
// device.h or config.h) If not, define it here for now.
#ifndef PCCL_CUDA_DEVICE_INLINE
#define PCCL_CUDA_DEVICE_INLINE __forceinline__ __device__
#endif

namespace pccl {

// MSCCLPP style getBuffer adapted for PCCL
template <typename T>
PCCL_CUDA_DEVICE_INLINE T *get_pccl_buffer(
    T *input_param, T *output_param,
    [[maybe_unused]] T *scratch_param, // Scratch not typically selected via
                                       // BufferType in PCCL Operation
    BufferType bufferType) {
  if (bufferType == BufferType::INPUT) {
    return input_param;
  }
  if (bufferType == BufferType::OUTPUT) {
    return output_param;
  }
  return nullptr;
}

// --- Handler function implementations (adapted from MSCCLPP logic) ---

PCCL_CUDA_DEVICE_INLINE void handle_pccl_signal(
    DeviceHandle<MemoryChannel> *memoryChannels,
    DeviceHandle<PortChannel> *portChannels,
    uint8_t *channelIndicesArr, // Corresponds to op.outputChannelIndexes
    int numChannelsToSignal,    // Corresponds to op.nOutputs
    ChannelType typeOfChannel   // Corresponds to op.channelType
) {
  int tid = threadIdx.x;
  if (tid < numChannelsToSignal && channelIndicesArr) {
    uint8_t ch_idx = channelIndicesArr[tid];
    if (typeOfChannel == ChannelType::MEMORY) {
      if (memoryChannels)
        memoryChannels[ch_idx].signal();
    } else if (typeOfChannel == ChannelType::PORT) {
      if (portChannels)
        portChannels[ch_idx].signal();
    }
  }
}

PCCL_CUDA_DEVICE_INLINE void handle_pccl_wait(
    DeviceHandle<MemoryChannel> *memoryChannels,
    DeviceHandle<PortChannel> *portChannels,
    uint8_t *channelIndicesArr, // Corresponds to op.inputChannelIndexes
    int numChannelsToWaitOn,    // Corresponds to op.nInputs
    ChannelType typeOfChannel   // Corresponds to op.channelType
) {
  int tid = threadIdx.x;
  if (tid < numChannelsToWaitOn && channelIndicesArr) {
    uint8_t ch_idx = channelIndicesArr[tid];
    if (typeOfChannel == ChannelType::MEMORY) {
      if (memoryChannels)
        memoryChannels[ch_idx].wait();
    } else if (typeOfChannel == ChannelType::PORT) {
      if (portChannels)
        portChannels[ch_idx].wait();
    }
  }
}

PCCL_CUDA_DEVICE_INLINE void handle_pccl_flush(
    DeviceHandle<PortChannel> *portChannels,
    uint8_t *channelIndicesArr, // Corresponds to op.outputChannelIndexes
    int numChannelsToFlush      // Corresponds to op.nOutputs
) {
  int tid = threadIdx.x;
  if (tid < numChannelsToFlush && portChannels && channelIndicesArr) {
    portChannels[channelIndicesArr[tid]].flush();
  }
}

template <bool PutWithSignal = false,
          bool PutWithSignalAndFlush = false> // Mimicking MSCCLPP options
PCCL_CUDA_DEVICE_INLINE void
handle_pccl_put(DeviceHandle<MemoryChannel> *memoryChannels,
                DeviceHandle<PortChannel> *portChannels,
                uint8_t *dstChannelIndicesArr, // op.outputChannelIndexes
                uint32_t *destinationOffsets,  // op.outputOffsets
                uint32_t *sourceOffsets,       // op.inputOffsets
                int numDestChannels,           // op.nOutputs
                uint32_t dataSizeInBytes,      // op.size
                ChannelType typeOfChannel      // op.channelType
) {
  if (typeOfChannel == ChannelType::MEMORY) {
    int tid = threadIdx.x;
    if (numDestChannels == 1 && memoryChannels && dstChannelIndicesArr &&
        destinationOffsets && sourceOffsets) {
      memoryChannels[dstChannelIndicesArr[0]].put(
          destinationOffsets[0], sourceOffsets[0], dataSizeInBytes, threadIdx.x,
          blockDim.x);
    } else if (tid < numDestChannels && memoryChannels &&
               dstChannelIndicesArr && destinationOffsets && sourceOffsets) {
      memoryChannels[dstChannelIndicesArr[tid]].put(
          destinationOffsets[tid], sourceOffsets[tid], dataSizeInBytes, 0,
          1); // Assumes put handles threading or it's 1-to-1
    }
  } else if (typeOfChannel == ChannelType::PORT) {
    int tid = threadIdx.x;
    if (tid < numDestChannels && portChannels && dstChannelIndicesArr &&
        destinationOffsets && sourceOffsets) {
      portChannels[dstChannelIndicesArr[tid]].put(
          destinationOffsets[tid], sourceOffsets[tid], dataSizeInBytes);
      if constexpr (PutWithSignal) { /* PCCL: Use separate SIGNAL op */
      }
      if constexpr (PutWithSignalAndFlush) { /* PCCL: Use separate SIGNAL/FLUSH
                                                ops */
      }
    }
  }
}

PCCL_CUDA_DEVICE_INLINE void handle_pccl_get(
    DeviceHandle<MemoryChannel> *memoryChannels,
    uint8_t *srcChannelIndicesArr, // op.inputChannelIndexes
    uint32_t *localDestOffsets,    // op.outputOffsets (local destination)
    uint32_t *remoteSrcOffsets,    // op.inputOffsets (remote source)
    int numSrcChannels,            // op.nInputs
    uint32_t dataSizeInBytes       // op.size
) {
  int tid = threadIdx.x;
  if (numSrcChannels == 1 && memoryChannels && srcChannelIndicesArr &&
      localDestOffsets && remoteSrcOffsets) {
    memoryChannels[srcChannelIndicesArr[0]].get(
        remoteSrcOffsets[0], localDestOffsets[0], dataSizeInBytes, threadIdx.x,
        blockDim.x);
  } else if (tid < numSrcChannels && memoryChannels && srcChannelIndicesArr &&
             localDestOffsets && remoteSrcOffsets) {
    memoryChannels[srcChannelIndicesArr[tid]].get(
        remoteSrcOffsets[tid], localDestOffsets[tid], dataSizeInBytes, 0,
        1); // Assumes get handles threading or it's 1-to-1
  }
}

PCCL_CUDA_DEVICE_INLINE void handle_pccl_copy(void *destinationBuffer,
                                              void *sourceBuffer,
                                              uint32_t destOffsetInBytes,
                                              uint32_t srcOffsetInBytes,
                                              size_t dataSizeInBytes) {
  if (!destinationBuffer || !sourceBuffer)
    return;
  char *srcDataPtr = static_cast<char *>(sourceBuffer) + srcOffsetInBytes;
  char *dstDataPtr = static_cast<char *>(destinationBuffer) + destOffsetInBytes;
  for (size_t i = threadIdx.x; i < dataSizeInBytes; i += blockDim.x) {
    dstDataPtr[i] = srcDataPtr[i];
  }
}

template <typename T, bool SendToRemote = true>
PCCL_CUDA_DEVICE_INLINE void handle_pccl_reduce_op(
    T *local_dst_buffer, uint32_t localDstOffsetBytes, T *local_src_buffer,
    uint32_t localSrcOffsetBytes,
    T *incoming_data_buffer, // Buffer for data from other ranks (e.g., scratch
                             // or part of input_buffer)
    uint32_t *remoteSrcOffsetsBytes, int num_remote_sources,
    DeviceHandle<MemoryChannel> *memChannels, uint8_t *remoteSrcChannelIndices,
    DeviceHandle<MemoryChannel> *dstMemChannels, uint8_t *destChannelIndices,
    uint32_t *remoteDstOffsetsBytes, int num_dest_channels,
    uint32_t dataSizeInBytes) {
  size_t num_elements = dataSizeInBytes / sizeof(T);
  T *effective_local_src_ptr = reinterpret_cast<T *>(
      reinterpret_cast<char *>(local_src_buffer) + localSrcOffsetBytes);
  T *effective_local_dst_ptr = reinterpret_cast<T *>(
      reinterpret_cast<char *>(local_dst_buffer) + localDstOffsetBytes);

  for (size_t elem_idx = threadIdx.x; elem_idx < num_elements;
       elem_idx += blockDim.x) {
    T accumulator = effective_local_src_ptr[elem_idx];

    for (int r_src_idx = 0; r_src_idx < num_remote_sources; ++r_src_idx) {
      T remote_val;
      if (memChannels && remoteSrcChannelIndices && remoteSrcOffsetsBytes) {
        uint32_t channel_data_base_offset_bytes =
            remoteSrcOffsetsBytes[r_src_idx];
        remote_val = memChannels[remoteSrcChannelIndices[r_src_idx]].read<T>(
            (channel_data_base_offset_bytes / sizeof(T)) + elem_idx);
      } else if (incoming_data_buffer && remoteSrcOffsetsBytes) {
        T *remote_data_src_ptr = reinterpret_cast<T *>(
            reinterpret_cast<char *>(incoming_data_buffer) +
            remoteSrcOffsetsBytes[r_src_idx]);
        remote_val = remote_data_src_ptr[elem_idx];
      } else {
        continue;
      }
      // Assumes pccl::custom_add_elements is defined in cuda/reduce_kernel.h
      // for various types (T, __half2, etc.)
      accumulator = pccl::custom_add_elements(accumulator, remote_val);
    }

    effective_local_dst_ptr[elem_idx] = accumulator;

    if (SendToRemote) {
      for (int dest_ch_idx = 0; dest_ch_idx < num_dest_channels;
           ++dest_ch_idx) {
        if (dstMemChannels && destChannelIndices && remoteDstOffsetsBytes) {
          uint32_t remote_ch_dst_base_offset_bytes =
              remoteDstOffsetsBytes[dest_ch_idx];
          dstMemChannels[destChannelIndices[dest_ch_idx]].write<T>(
              (remote_ch_dst_base_offset_bytes / sizeof(T)) + elem_idx,
              accumulator);
        }
      }
    }
  }
}

// Main Kernel
template <typename T, typename PacketType = LL16Packet>
__global__ __launch_bounds__(
    Config::PCCL_KERNEL_THREADS_PER_BLOCK,
    Config::
        PCCL_KERNEL_MIN_BLOCKS_PER_SM) void executionKernel([[maybe_unused]] int
                                                                rank_arg, // Useful
                                                                          // for
                                                                          // debugging
                                                            T *input_main_buffer,
                                                            T *output_main_buffer,
                                                            DeviceExecutionPlan *
                                                                arrayOfPlans_gpu, // Pointer to an array of plans on GPU
                                                            int total_num_plans,
                                                            [[maybe_unused]] uint32_t
                                                                flag_arg // MSCCLPP-style
                                                                         // flag,
                                                                         // potentially
                                                                         // for
                                                                         // future
                                                                         // packet
                                                                         // ops
) {
  extern __shared__ char shared_memory_buffer[];
  int4 *shared_plan_location_int4 =
      reinterpret_cast<int4 *>(shared_memory_buffer);

  int sm_block_id = blockIdx.x;
  int tid_in_block = threadIdx.x;

  for (int plan_batch_offset = 0; plan_batch_offset < total_num_plans;
       plan_batch_offset += gridDim.x) {
    int current_plan_global_idx = sm_block_id + plan_batch_offset;
    if (current_plan_global_idx >= total_num_plans) {
      break;
    }

    DeviceExecutionPlan *current_plan_gpu_ptr =
        arrayOfPlans_gpu + current_plan_global_idx;

    for (size_t i = tid_in_block;
         i < sizeof(DeviceExecutionPlan) / sizeof(int4); i += blockDim.x) {
      shared_plan_location_int4[i] =
          (reinterpret_cast<int4 *>(current_plan_gpu_ptr))[i];
    }
    __syncthreads();

    DeviceExecutionPlan *plan_in_shmem =
        reinterpret_cast<DeviceExecutionPlan *>(shared_plan_location_int4);

    int num_ops_in_plan = plan_in_shmem->nOperations;
    Operation *ops_array_in_shmem = plan_in_shmem->operations;

    DeviceHandle<MemoryChannel> *mem_ch_handles =
        plan_in_shmem->channels.memoryChannels;
    DeviceHandle<PortChannel> *port_ch_handles =
        plan_in_shmem->channels.portChannels;

    for (int op_loop_idx = 0; op_loop_idx < num_ops_in_plan; op_loop_idx++) {
      Operation &current_op = ops_array_in_shmem[op_loop_idx];

      if (current_op.type == OperationType::NOP) {
        __syncthreads();
      } else if (current_op.type == OperationType::BARRIER) {
        if (current_op.nThreadBlocks > 0 &&
            current_op.deviceSyncerIndex < MAX_DEVICE_SYNCERS) {
          deviceSyncers[current_op.deviceSyncerIndex].sync(
              current_op.nThreadBlocks);
        }
      } else if (current_op.type == OperationType::SIGNAL) {
        handle_pccl_signal(mem_ch_handles, port_ch_handles,
                           current_op.outputChannelIndexes, current_op.nOutputs,
                           current_op.channelType);
      } else if (current_op.type == OperationType::WAIT) {
        handle_pccl_wait(mem_ch_handles, port_ch_handles,
                         current_op.inputChannelIndexes, current_op.nInputs,
                         current_op.channelType);
      } else if (current_op.type == OperationType::FLUSH) {
        handle_pccl_flush(port_ch_handles, current_op.outputChannelIndexes,
                          current_op.nOutputs);
      } else if (current_op.type == OperationType::PUT) {
        handle_pccl_put(
            mem_ch_handles, port_ch_handles, current_op.outputChannelIndexes,
            current_op.outputOffsets, current_op.inputOffsets,
            current_op.nOutputs, current_op.size, current_op.channelType);
      } else if (current_op.type == OperationType::GET) {
        handle_pccl_get(mem_ch_handles, current_op.inputChannelIndexes,
                        current_op.outputOffsets, current_op.inputOffsets,
                        current_op.nInputs, current_op.size);
      } else if (current_op.type == OperationType::COPY) {
        T *dst_buf = get_pccl_buffer<T>(input_main_buffer, output_main_buffer,
                                        nullptr, current_op.dstBufferType);
        T *src_buf = get_pccl_buffer<T>(input_main_buffer, output_main_buffer,
                                        nullptr, current_op.srcBufferType);
        handle_pccl_copy(dst_buf, src_buf, current_op.dstOffset,
                         current_op.srcOffset, current_op.size);
      } else if (current_op.type == OperationType::REDUCE) {
        T *dst_reduce_buf =
            get_pccl_buffer<T>(input_main_buffer, output_main_buffer, nullptr,
                               current_op.dstBufferType);
        T *src_reduce_buf =
            get_pccl_buffer<T>(input_main_buffer, output_main_buffer, nullptr,
                               current_op.srcBufferType);
        bool send_reduced_result_remotely = (current_op.nOutputs > 0);
        T *reduce_incoming_buf =
            input_main_buffer; // Simplified assumption for incoming remote data
                               // buffer

        if (send_reduced_result_remotely) {
          handle_pccl_reduce_op<T, true>(
              dst_reduce_buf, current_op.dstOffset, src_reduce_buf,
              current_op.srcOffset, reduce_incoming_buf,
              current_op.inputOffsets, current_op.nInputs, mem_ch_handles,
              current_op.inputChannelIndexes, mem_ch_handles,
              current_op.outputChannelIndexes, current_op.outputOffsets,
              current_op.nOutputs, current_op.size);
        } else { // Local reduce only
          handle_pccl_reduce_op<T, false>(
              dst_reduce_buf, current_op.dstOffset, src_reduce_buf,
              current_op.srcOffset, reduce_incoming_buf,
              current_op.inputOffsets, current_op.nInputs, mem_ch_handles,
              current_op.inputChannelIndexes, nullptr, nullptr, nullptr,
              0, // No remote output
              current_op.size);
        }
      }
      // Other MSCCLPP operations like specific Packet ops or NVLS ops are
      // omitted as pccl::OperationType does not seem to have direct
      // equivalents.
    } // end for operations
    __syncthreads();
  } // end for plans
}

} // namespace pccl