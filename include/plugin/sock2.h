#pragma once

#include <netinet/in.h>
#include <sys/socket.h>

#include <atomic>
#include <deque>
#include <map>
#include <memory>
#include <thread>

#include "runtime.h"

namespace pccl {

struct SockAddr {
  union {
    struct {
      uint32_t ip;
      uint16_t port;
    } v4;
    struct {
      struct in6_addr ip;
      uint16_t port;
    } v6;
  };
  sa_family_t family;
};

enum class MessageType : int8_t {
  REQUEST,      // Operation request
  RESPONSE,     // Operation response
  COMPLETION,   // Work completion notification
  CONNECT,      // Connection information exchange
  CONNECT_ACK,  // Connection acknowledgment
  ERROR         // Error response
};

enum class SockOpType : int8_t { READ, WRITE, ATOMIC_ADD, WRITE_WITH_IMM };

enum class SockStatus : int8_t {
  SUCCESS,
  ERROR_GENERAL,
  ERROR_MR_NOT_FOUND,
  ERROR_CONN_FAILED,
  ERROR_TIMEOUT,
  ERROR_INVALID_PARAM,
  ERROR_OUT_OF_MEMORY,
  ERROR_QP_STATE,
  ERROR_SEND_FAILED,
  ERROR_RECV_FAILED,
  ERROR_WR_OVERFLOW
};

struct MessageHeader {
  SockAddr addr;
  MessageType type;
  SockOpType op_type;
  SockStatus status;
  uint32_t imm_data;
  uint64_t wr_id;
  int qpn;
};

struct SockMrInfo {
  uintptr_t addr;
  size_t size;
  uint32_t mr_id;
  DeviceType dev_type;
};

struct MessagePayloadHeader {
  SockMrInfo mr_info;
  uint32_t offset;
  uint32_t size;
};

class SockCtx;

struct SockMr {
 public:
  ~SockMr() = default;

  uintptr_t addr;
  size_t size;
  uint32_t mr_id;
  bool is_host_memory;
};

struct SockWc {
  uint64_t wr_id;
  SockStatus status;
  uint32_t imm_data;
};

struct SockWr {
  uint64_t wr_id;
  SockOpType op_type;
  SockMrInfo local_mr_info;
  SockMrInfo remote_mr_info;
  uint32_t size;
  uint32_t local_offset;
  uint32_t remote_offset;
  uint32_t imm_data;
  bool signaled;
  uint64_t atomic_value;
};

struct SockQpInfo {
  SockAddr addr;
  int qpn;
};

class SockQp : public std::enable_shared_from_this<SockQp> {
 public:
  ~SockQp();
  SockStatus connect(const SockQpInfo &remote_info, std::shared_ptr<SockCtx> ctx);
  void stageLoad(const SockMr *mr, const SockMrInfo &info, size_t size, uint64_t wrId,
                 uint64_t srcOffset, uint64_t dstOffset, bool signaled);
  void stageSend(const SockMr *mr, const SockMrInfo &info, uint32_t size, uint64_t wrId,
                 uint64_t srcOffset, uint64_t dstOffset, bool signaled);
  void stageAtomicAdd(const SockMr *mr, const SockMrInfo &info, uint64_t wrId, uint64_t dstOffset,
                      uint64_t addVal, bool signaled);
  void stageSendWithImm(const SockMr *mr, const SockMrInfo &info, uint32_t size, uint64_t wrId,
                        uint64_t srcOffset, uint64_t dstOffset, bool signaled,
                        unsigned int immData);
  SockStatus postSend();
  int pollCq();

  bool active() const;
  int sendSockFd() const;
  int recvSockFd() const;

 private:
  friend class SockCtx;
  SockStatus sendReadRequest(MessageHeader &header, MessagePayloadHeader &payload_header,
                             const SockWr &wr);
  SockStatus sendWriteRequest(MessageHeader &header, MessagePayloadHeader &payload_header,
                              const SockWr &wr);
  SockStatus sendAtomicRequest(MessageHeader &header, MessagePayloadHeader &payload_header,
                               const SockWr &wr);

  int maxCqSize;
  int maxWrqSize;
  std::shared_ptr<SockCtx> ctx;
  std::shared_ptr<std::deque<SockWr>> wr_queue;
  std::shared_ptr<std::deque<SockWc>> wc_queue;
  int qpn;
  SockQpInfo remote_info;
  bool is_connected;
  int send_fd;
  int recv_fd;
};

struct SockQpStatistics {
  uint64_t last_called;
  uint64_t called_times;
};

struct SocketTask {
  std::shared_ptr<MessageHeader> header;
  std::shared_ptr<MessagePayloadHeader> payload_header;
  size_t sended_size;
  size_t recved_size;
  int send_fd;
  int recv_fd;
  std::shared_ptr<SockMr> mr;
};

class SockCtx : public std::enable_shared_from_this<SockCtx> {
 public:
  SockCtx(SockAddr addr);
  ~SockCtx();

  std::shared_ptr<SockQp> createQp(int max_cq_size, int max_wr);
  std::shared_ptr<SockMr> registerMr(void *buff, size_t size, bool isHostMemory);
  SockStatus sendData(int socket_fd, MessageHeader &header, MessagePayloadHeader &payload,
                      std::shared_ptr<SockMr> mr, uint32_t offset, uint32_t size);
  static std::atomic<uint32_t> next_mr_id;
  static std::atomic<int> next_qp_id;

  SockAddr addr;

 private:
  std::shared_ptr<SockQp> findQpByQpn(int qpn);

  void manageConnections();

  void sendThreadCycle();
  void recvThreadCycle();

 private:
  int listen_socket;
  int recv_epoll_fd;
  int send_epoll_fd;

  std::map<int, std::shared_ptr<SockQp>> qps;
  std::map<uint32_t, std::shared_ptr<SockMr>> mrs;

  std::map<int, int> qpn_to_send_fd;
  std::map<int, int> qpn_to_recv_fd;

  bool running;
  std::thread recv_thread;
  std::thread send_thread;

  std::deque<std::shared_ptr<std::deque<SockWr>>> wr_queue;
  std::deque<std::shared_ptr<std::deque<SockWc>>> wc_queue;
};

};  // namespace pccl