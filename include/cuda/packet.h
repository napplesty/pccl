#pragma once

#include <cstdint>

#include "device.h"

namespace pccl {

union alignas(16) SimplePacket {
  struct {
    uint32_t flag;
    uint32_t base_data;
  };
};

union alignas(16) LL16Packet {
  struct {
    uint32_t flag0;
    uint32_t flag1;
    uint32_t data0;
    uint32_t data1;
  };
  uint32_t data[4];
#if defined(PCCL_CUDA_DEVICE_COMPILE)
  ulonglong2 raw_;
  PCCL_CUDA_DEVICE_INLINE LL16Packet() : raw_{0, 0} {}
  PCCL_CUDA_DEVICE_INLINE LL16Packet(uint32_t flag, uint32_t data0,
                                     uint32_t data1) {
    raw_.x = (uint64_t(flag) << 32) | flag;
    raw_.y = (uint64_t(data0) << 32) | data1;
  }
  PCCL_CUDA_DEVICE_INLINE void write(uint32_t flag, uint32_t data0,
                                     uint32_t data1) {
#if defined(USE_CUDA)
    asm volatile("st.volatile.global.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"(&raw_),
                 "r"(flag), "r"(flag), "r"(data0), "r"(data1));
#elif defined(USE_HIP)
    uint4 data_to_store = make_uint4(flag, flag, data0, data1);
    asm volatile("global_store_dwordx4 %0, %1, off glc"
                 : /* no outputs */
                 : "s"(&raw_), "v"(data_to_store)
                 : "memory");
#else
    uint64_t val_x =
        (static_cast<uint64_t>(flag) << 32) | static_cast<uint64_t>(flag);
    uint64_t val_y =
        (static_cast<uint64_t>(data0) << 32) | static_cast<uint64_t>(data1);
    atomicStore(&raw_.x, val_x, memoryOrderRelaxed);
    atomicStore(&raw_.y, val_y, memoryOrderRelaxed);
#endif
  }

  PCCL_CUDA_DEVICE_INLINE void write(uint64_t val, uint32_t flag) {
    write((uint32_t)val, (uint32_t)(val >> 32), flag);
  }
  PCCL_CUDA_DEVICE_INLINE void write(uint2 val, uint32_t flag) {
    write(val.x, val.y, flag);
  }
  PCCL_CUDA_DEVICE_INLINE uint32_t readOnce(uint32_t flag, uint2& data) const {
#if defined(USE_CUDA)
    uint32_t flag1, flag2;
    asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(flag1), "=r"(flag2), "=r"(data.x), "=r"(data.y)
                 : "l"(&raw_));
    return (flag1 != flag) || (flag2 != flag);
#elif defined(USE_HIP)
    uint4 loaded_data;
    asm volatile("global_load_dwordx4 %0, %1, off glc"
                 : "=v"(loaded_data)
                 : "s"(&raw_)
                 : "memory");
    uint32_t flag0_loaded = loaded_data.x;
    uint32_t flag1_loaded = loaded_data.y;
    data.x = loaded_data.z;  // data0
    data.y = loaded_data.w;  // data1
    return (flag0_loaded != flag) || (flag1_loaded != flag);
#else
    ulonglong2 reg;
    reg.x = atomicLoad(&(raw_.x), memoryOrderRelaxed);
    reg.y = atomicLoad(&(raw_.y), memoryOrderRelaxed);
    uint32_t flag0 = (reg.x >> 32) & 0xffffffff;
    uint32_t flag1 = reg.x & 0xffffffff;
    uint32_t data0 = (reg.y >> 32) & 0xffffffff;
    uint32_t data1 = reg.y & 0xffffffff;
    data.x = data0;
    data.y = data1;
    return (flag0 != flag) || (flag1 != flag);
#endif
  }

  PCCL_CUDA_DEVICE_INLINE uint2 read(uint32_t flag,
                                     int64_t maxSpinCount = 100000000) const {
    uint2 data;
    POLL_MAYBE_JAILBREAK(readOnce(flag, data), maxSpinCount);
    return data;
  }
#endif
};

}  // namespace pccl
