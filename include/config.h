#pragma once

#include <netdb.h>

#include <cstdint>
#include <memory>
#include <string>

namespace pccl {

struct Config {
  // kernel
  constexpr static int WARP_SIZE = 32;
  constexpr static int WARP_PER_SM = 8;
  constexpr static int WARP_FOR_SCHEDULE = 1;
  constexpr static int WARP_FOR_PROXY = 1;
  constexpr static int WARP_FOR_MEMORY = 6;
  constexpr static int MAX_SM_LIB = 18;
  constexpr static int DEVICE_SYNCER_SIZE = MAX_SM_LIB * 2 * 8;
  constexpr static int MAX_OPERATIONS_PER_CAPSULE = 64;
  constexpr static int FIFO_BUFFER_SIZE = MAX_OPERATIONS_PER_CAPSULE * 8;

  // network connections
  constexpr static int MAX_ACTIVE_CONNECTIONS = 256;
  constexpr static int MAX_CHANNEL_PER_OPERATION = 32;

  // buffer
  constexpr static size_t WORKSPACE_SIZE = 4 * 1024 * 1024;
  constexpr static size_t DEVICE_BUFFER_SIZE = 256 * 1024 * 1024;
  constexpr static size_t HOST_BUFFER_SIZE = 256 * 1024 * 1024;
  constexpr static size_t SLOT_SIZE = 1024 * 1024;

  // ib
  constexpr static int MAX_CQ_SIZE = 64 * 32;
  constexpr static int MAX_CQ_POLL_NUM = 1;
  constexpr static int MAX_SEND_WR = 2 * MAX_CQ_SIZE;
  constexpr static int MAX_WR_PER_SEND = 32;

  // ether
  constexpr static int MAX_DATA_PACKET_SIZE = 5000;
  constexpr static int MAX_CONTROL_PACKET_SIZE = 128;
  constexpr static uint64_t MSCCLPP_SOCKET_MAGIC = 0x564ab9f2fc4b9d6cULL;
};

struct env {
  const int rank;
  const int localRank;
  const int worldSize;
  const ::std::string socketFamily;
  const ::std::string socketAddr0;
  const ::std::string socketPort0;
  const ::std::string socketAddr1;
  const ::std::string socketPort1;
  const ::std::string ibSocketFamily;
  const ::std::string ibDevice0;
  const ::std::string ibDevice1;
  const ::std::string ibPort0;
  const ::std::string ibPort1;
  const ::std::string netConfAddr;
  const ::std::string netConfPort;
  const ::std::string netConfModel;
  const ::std::string profileDir;
};

::std::shared_ptr<env> getEnv();

}  // namespace pccl
