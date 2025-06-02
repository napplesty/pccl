#pragma once

#include <memory>
#include <netdb.h>
#include <string>

namespace pccl {

struct Config {
  // kernel
  constexpr static int WARP_SIZE = 32;
  constexpr static int WARP_PER_SM = 10;
  constexpr static int WARP_FOR_SCHEDULE = 1;
  constexpr static int WARP_FOR_PROXY = 1;
  constexpr static int WARP_FOR_MEMORY = 8;
  constexpr static int MAX_SM_COUNT = 24;
  constexpr static int DEVICE_SYNCER_SIZE = MAX_SM_COUNT * 2 * 8;
  constexpr static int MAX_OPERATIONS_PER_CAPSULE = 64;
  constexpr static int FIFO_BUFFER_SIZE =
      MAX_OPERATIONS_PER_CAPSULE * WARP_SIZE;
  constexpr static int INTER_SM_FIFO_SIZE =
      MAX_OPERATIONS_PER_CAPSULE * WARP_SIZE / WARP_PER_SM;

  // network connections
  constexpr static int MAX_CHANNEL_PER_OPERATION = WARP_SIZE;
  constexpr static int MAX_ACTIVE_CONNECTIONS =
      MAX_CHANNEL_PER_OPERATION * MAX_OPERATIONS_PER_CAPSULE;

  // buffer
  constexpr static size_t WORKSPACE_SIZE = 256 * 1024 * 1024;
  constexpr static size_t DEVICE_BUFFER_SIZE = 256 * 1024 * 1024;
  constexpr static size_t HOST_BUFFER_SIZE = 256 * 1024 * 1024;
  constexpr static size_t SLOT_SIZE = 1024 * 1024;

  // ib
  constexpr static int MAX_CQ_SIZE = MAX_ACTIVE_CONNECTIONS;
  constexpr static int MAX_CQ_POLL_NUM = WARP_SIZE;
  constexpr static int MAX_SEND_WR = 2 * MAX_CQ_SIZE;
  constexpr static int MAX_WR_PER_SEND = WARP_SIZE;

  // proxy
  constexpr static int ProxyFlushPeriod = 4;
  constexpr static int ProxyStopCheckPeriod = 4000;
};

struct env {
  const int rank;
  const int localRank;
  const int worldSize;
  const std::string socketFamily;
  const std::string socketAddr0;
  const std::string socketPort0;
  const std::string socketAddr1;
  const std::string socketPort1;
  const std::string ibSocketFamily;
  const std::string ibDevice0;
  const std::string ibDevice1;
  const std::string ibPort0;
  const std::string ibPort1;
  const std::string netConfFile;
  const std::string netConfAddr;
  const std::string netConfPort;
  const std::string netConfModel;
  const std::string profileDir;
  const std::string enableTransportList;
};

std::shared_ptr<env> getEnv();

} // namespace pccl
