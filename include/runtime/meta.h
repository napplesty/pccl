#pragma once 

namespace pccl {

enum class OperatorType { 
  SYNC_COMPUTE,
  ASYNC_COMPUTE,
  SYNC_IO, 
  ASYNC_IO 
};

} // namespace pccl

