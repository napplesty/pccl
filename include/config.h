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
  constexpr static size_t LIB_BUFFER_SIZE = 256 * 1024 * 1024;
  constexpr static size_t DEVICE_BUFFER_SIZE = 256 * 1024 * 1024;
  constexpr static size_t HOST_BUFFER_SIZE = 256 * 1024 * 1024;
  constexpr static size_t NUM_SLOT = 256;

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
  int rank;
  int localRank;
  int worldSize;
  std::string socketFamily;
  std::string socketAddr0;
  std::string socketPort0;
  std::string socketAddr1;
  std::string socketPort1;
  std::string ibSocketFamily;
  std::string ibDevice0;
  std::string ibDevice1;
  std::string ibPort0;
  std::string ibPort1;
  std::string netConfFile;
  std::string netConfAddr;
  std::string netConfPort;
  std::string netConfModel;
  std::string profileDir;
  std::string enableTransportList;
};

std::shared_ptr<env> getEnv();

} // namespace pccl
