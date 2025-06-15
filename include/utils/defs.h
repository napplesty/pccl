#pragma once

#include <cstdint>
#include <pthread.h>
#include <sched.h>
#include <string>
#include <unistd.h>

#define PCCL_API __attribute__((visibility("default")))
#define PCCL_OFFSET_OF(type, member) ((size_t) & ((type *)0)->member)

namespace pccl {

std::string get_start_timestamp();
void create_dir(const std::string &path);
void set_affinity(int cpu_id);
uint64_t host_hash();
uint64_t pid_hash();

} // namespace pccl