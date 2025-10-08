#pragma once
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <runtime/api/repr.h>

namespace pccl::utils {

void *allocate(runtime::ExecutorType executor_type, size_t size);
std::string get_shareable_handle(runtime::ExecutorType executor_type, void *addr);
void *from_shareable(runtime::ExecutorType executor_type, const std::string &shareable_handle);
void close_shareable_handle(runtime::ExecutorType executor_type, void* ptr);
bool generic_memcpy(runtime::ExecutorType src_type, runtime::ExecutorType dst_type, void* src, void* dst, size_t nbytes);

}
