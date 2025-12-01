#pragma once

#include <base/device.h>

namespace engine_c {
  
class CudaDevice : public DeviceBase {
public:
  bool allocatorAvailable() override;
  void *allocate(long nbytes) override;
  void deallocate(void *ptr) override;
  
  bool IPCAvailable() override;
  std::string allocateIpcBuffer(void **addr, long size) override;
  long mapBuffer(std::string &shareable_ShareableHandle, void **addr) override;
};

}
