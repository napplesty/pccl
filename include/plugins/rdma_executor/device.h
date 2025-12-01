#pragma once

#include <base/device.h>

namespace engine_c {
  
class RdmaDevice : public DeviceBase {
public:
  bool remoteCommAvailable() override;
  std::string activate() override;
  std::string registerBuffer(void *addr, long size) override;
  void connect(std::string handle) override;
  void disconnect(std::string handle) override;
};

}
