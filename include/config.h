#pragma once

#include <netdb.h>

#include <cstdint>
#include <memory>
#include <string>

namespace pccl {
constexpr uint64_t DEFAULT_KERNEL_THREAD_NUM = 128;
constexpr uint64_t DEFAULT_FIFO_SIZE = DEFAULT_KERNEL_THREAD_NUM;

// ib
constexpr int DefaultMaxCqSize = 1024;
constexpr int DefaultMaxCqPollNum = 1;
constexpr int DefaultMaxSendWr = 8192;
constexpr int DefaultMaxWrPerSend = 64;

// ether
constexpr int MaxNumIfaces = 16;
constexpr int MaxIfaceNameLen = 16;
constexpr int MaxLenSocketName = (NI_MAXHOST + NI_MAXSERV + 1);
constexpr uint64_t MSCCLPP_SOCKET_MAGIC = 0x564ab9f2fc4b9d6cULL;

struct env {
  const int rank;
  const int localRank;
  const int worldSize;
  const ::std::string hostId;
  const ::std::string socketFamily;
  const ::std::string socketIface;
  const ::std::string ibDevice0;
  const ::std::string ibDevice1;
  const ::std::string ibPort0;
  const ::std::string ibPort1;
  const ::std::string executionPlanDir;
  const ::std::string netConfAddr;
};

::std::shared_ptr<env> getEnv();

#define PCCL_API __attribute__((visibility("default")))

}  // namespace pccl
