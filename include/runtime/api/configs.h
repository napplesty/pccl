#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <memory>

namespace pccl::runtime {

struct PcclConfig {
  cudaStream_t executor_stream;
};

std::shared_ptr<PcclConfig> get_global_config();

}

