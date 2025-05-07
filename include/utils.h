#pragma once

#include <pthread.h>
#include <sched.h>
#include <unistd.h>

#include <cstdint>

#define PCCL_API __attribute__((visibility("default")))
#define PCCL_OFFSET_OF(type, member) ((size_t) & ((type *)0)->member)

namespace pccl {

PCCL_API int bindToCpu();
PCCL_API uint64_t getHostHash();
PCCL_API uint64_t getPidHash();

}  // namespace pccl