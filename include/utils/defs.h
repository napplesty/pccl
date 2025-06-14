#pragma once

#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#include <cstdint>

#define PCCL_API __attribute__((visibility("default")))
#define PCCL_OFFSET_OF(type, member) ((size_t) & ((type *)0)->member)

namespace pccl {

void set_affinity(int cpu_id);
uint64_t host_hash();
uint64_t pid_hash();

} // namespace pccl