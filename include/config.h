#pragma once

namespace pccl {

struct Config {
  static unsigned long DEVICE_BUFFER_SIZE;
  static unsigned long HOST_BUFFER_SIZE;
  static unsigned long SLOT_GRANULARITY;
  static int PROXY_FLUSH_PERIOD;
  static int PROXY_MAX_FLUSH_SIZE;
  static int PROXY_CHECK_STOP_PERIOD;
};

} // namespace pccl