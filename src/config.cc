#include "config.h"

namespace pccl {

// kernel
// ThreadBlock execution model [0:32 prologue from input to buffer,
// 32:64 epilogue from buffer to output,
// 64:96 proxy handler,
// 96:DEFAULT_KERNEL_THREAD_NUM, memory handler]
uint64_t Config::DEFAULT_KERNEL_THREAD_NUM = 256;
uint64_t Config::DEFAULT_PROXY_THREAD_NUM = 32;
uint64_t Config::DEFAULT_MEMORY_THREAD_NUM =
    DEFAULT_KERNEL_THREAD_NUM - DEFAULT_PROXY_THREAD_NUM;
uint64_t Config::DEFAULT_FIFO_SIZE = Config::DEFAULT_KERNEL_THREAD_NUM;
uint64_t Config::DEFAULT_MAX_THREAD_BLOCK_NUM = 8;
uint64_t Config::DEFAULT_NUM_SYNCER = DEFAULT_MAX_THREAD_BLOCK_NUM;
int Config::ProxyStopCheckPeriod = 8192;
int Config::ProxyFlushPeriod = 4;

// channel
int Config::MAX_CHANNEL = 1024;
int Config::MAX_CHANNEL_PER_OPERATION = DEFAULT_PROXY_THREAD_NUM;

// buffer
int Config::MAX_LIB_BUFFER = 1;
int Config::MAX_LIB_BUFFER_SIZE = 16 * 1024;
int Config::MAX_DEVICE_BUFFER = 128;
int Config::MAX_DEVICE_BUFFER_SIZE = 1 * 1024 * 1024;
int Config::MAX_HOST_BUFFER = 32;
int Config::MAX_HOST_BUFFER_SIZE = 1 * 1024 * 1024;

// ib
int Config::DefaultMaxCqSize =
    MAX_OPERATION_PER_THREADBLOCK * DEFAULT_PROXY_THREAD_NUM;
int Config::DefaultMaxCqPollNum = 1;
int Config::DefaultMaxSendWr = 2 * DefaultMaxCqSize;
int Config::DefaultMaxWrPerSend = DEFAULT_PROXY_THREAD_NUM;

// ether
int Config::MaxDataPacketSize = 5000;
int Config::MaxControlPacketSize = 128;
uint64_t Config::MSCCLPP_SOCKET_MAGIC = 0x564ab9f2fc4b9d6cULL;

}  // namespace pccl