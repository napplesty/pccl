#pragma once

#include <netdb.h>

#include <cstdint>
#include <memory>
#include <string>

namespace pccl {

struct Config {
  // kernel
  static uint64_t DEFAULT_KERNEL_THREAD_NUM;
  static uint64_t DEFAULT_PROXY_THREAD_NUM;
  static uint64_t DEFAULT_MEMORY_THREAD_NUM;
  static uint64_t DEFAULT_FIFO_SIZE;
  static uint64_t DEFAULT_MAX_THREAD_BLOCK_NUM;
  static uint64_t DEFAULT_NUM_SYNCER;
  static int ProxyStopCheckPeriod;
  static int ProxyFlushPeriod;

  // channel
  static int MAX_CHANNEL;
  static int MAX_CHANNEL_PER_OPERATION;
  constexpr static int MAX_OPERATION_PER_THREADBLOCK = 64;
  static int MAX_LIB_BUFFER;
  static int MAX_LIB_BUFFER_SIZE;
  static int MAX_DEVICE_BUFFER;
  static int MAX_DEVICE_BUFFER_SIZE;
  static int MAX_HOST_BUFFER;
  static int MAX_HOST_BUFFER_SIZE;

  // ib
  static int DefaultMaxCqSize;
  static int DefaultMaxCqPollNum;
  static int DefaultMaxSendWr;
  static int DefaultMaxWrPerSend;

  // ether
  static int MaxDataPacketSize;
  static int MaxControlPacketSize;
  static uint64_t MSCCLPP_SOCKET_MAGIC;
};

struct env {
  const int rank;
  const int localRank;
  const int worldSize;
  const ::std::string socketFamily;
  const ::std::string socketAddr;
  const ::std::string socketPort;
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
