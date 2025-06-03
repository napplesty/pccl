#pragma once

#include <cstdint>
#include <netinet/in.h>
#include <sys/socket.h>

#include <atomic>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <thread>

namespace pccl {

struct SockAddr {
  union {
    struct {
      struct in_addr ip;
      uint16_t port;
    } v4;
    struct {
      struct in6_addr ip;
      uint16_t port;
    } v6;
  };
  sa_family_t family;
};

enum class SockOpType : int8_t {
  READ,
  WRITE,
  ATOMIC_ADD,
  WRITE_WITH_IMM,
  HAND_SHAKE,
  HAND_SHAKE_ACK,
};

enum class SockStatus : int8_t {
  SUCCESS,
  ERROR_GENERAL,
  ERROR_MR_NOT_FOUND,
  ERROR_QP_STATE,
};

struct SockMrInfo {
  uintptr_t addr;
  int64_t mr_id;
  bool is_host_memory;
};

class SockCtx;

struct SockMr {
  ~SockMr();
  uintptr_t addr;
  int64_t mr_id;
  bool is_host_memory;
  std::weak_ptr<SockCtx> ctx;
};

struct SockWc {
  uint64_t wr_id;
  SockStatus status;
  uint32_t imm_data;
};

struct SockWr {
  uint64_t wr_id;
  int64_t local_qpn;
  int64_t remote_qpn;
  SockOpType op_type;
  SockMrInfo local_mr_info;
  SockMrInfo remote_mr_info;
  uint32_t size;
  uint32_t local_offset;
  uint32_t remote_offset;
  uint32_t imm;
  uint64_t atomic_value;
};

struct SockQpInfo {
  SockAddr addr;
  int64_t qpn;
};

class SockQp : public std::enable_shared_from_this<SockQp> {
public:
  ~SockQp();
  SockStatus connect(const SockQpInfo &remote_info);
  void stageLoad(const SockMr *mr, const SockMrInfo &info, size_t size,
                 uint64_t wrId, uint64_t srcOffset, uint64_t dstOffset);
  void stageSend(const SockMr *mr, const SockMrInfo &info, uint32_t size,
                 uint64_t wrId, uint64_t srcOffset, uint64_t dstOffset);
  void stageAtomicAdd(const SockMr *mr, const SockMrInfo &info, uint64_t wrId,
                      uint64_t dstOffset, uint64_t addVal);
  void stageSendWithImm(const SockMr *mr, const SockMrInfo &info, uint32_t size,
                        uint64_t wrId, uint64_t srcOffset, uint64_t dstOffset,
                        unsigned int immData);
  SockStatus postSend();
  int pollCq();
  SockWc &getWcStatus(int idx);
  SockQpInfo getInfo() const;

private:
  SockWr &stageOp();
  friend class SockCtx;
  SockQp() = default;
  int maxCqSize;
  int maxWrqSize;
  std::weak_ptr<SockCtx> ctx;
  std::mutex qp_mutex;
  std::shared_ptr<std::deque<SockWr>> staged_operations;
  std::shared_ptr<std::deque<SockWc>> wc_queue;
  std::shared_ptr<std::deque<SockWc>> pulled_wcs;
  int64_t qpn;
  SockQpInfo remote_info;
  bool is_connected;
  SockAddr local_addr;
};

struct MessageHeader {
  union {
    uint32_t imm_data;
    uint64_t atomic_value;
  };
  uint64_t src_wr_id;
  int64_t src_qpn;
  int64_t dst_qpn;
  union {
    struct {
      SockMrInfo dst_mr_info;
      SockMrInfo src_mr_info;
      uint32_t dst_offset;
      uint32_t src_offset;
      uint32_t size;
    };
    SockAddr sock_addr;
  };
  SockOpType op_type;
};

class SockCtx : public std::enable_shared_from_this<SockCtx> {
public:
  SockCtx(SockAddr addr);
  ~SockCtx();

  std::shared_ptr<SockQp> createQp(int max_cq_size, int max_wr);
  std::shared_ptr<SockMr> registerMr(void *buff, size_t size,
                                     bool is_host_memory);
  void unregisterMr(int64_t mr_id);

private:
  friend class SockQp;

  struct SendRecvContext {
    ssize_t remain_size;
    std::shared_ptr<MessageHeader> header;
  };

  std::shared_ptr<SockQp> findQpByQpn(int64_t qpn);
  void sendThreadCycle();
  void recvThreadCycle();
  bool qp_connect(const SockQpInfo &remote_info, int64_t local_qpn);

  SockStatus sendHeader(int socket_fd, const MessageHeader &header);
  SockStatus sendData(int socket_fd, const MessageHeader &header);

  std::shared_ptr<std::deque<SockWr>> pop_from_global_send_queue();
  void push_to_global_send_queue(std::shared_ptr<std::deque<SockWr>> wr_queue);

  void register_qpn(int64_t qpn, std::shared_ptr<SockQp> qp);
  void unregister_qpn(int64_t qpn);

  bool handleHandshake(int fd, const MessageHeader &header);
  bool sendHandshakeAck(int fd, int64_t local_qpn, int64_t remote_qpn);
  bool processPartialSend(int fd, SendRecvContext &context);
  bool processPartialRecv(int fd, SendRecvContext &context);
  bool sendMessageHeader(int fd, const MessageHeader &header,
                         size_t &bytes_sent);
  bool sendMessageData(int fd, const SendRecvContext &context,
                       size_t &bytes_sent);
  bool recvMessageHeader(int fd, MessageHeader &header, size_t &bytes_recv);
  bool recvMessageData(int fd, SendRecvContext &context, size_t &bytes_recv);

private:
  std::mutex ctx_mutex;
  static std::atomic<int64_t> next_mr_id;
  static std::atomic<int64_t> next_qp_id;

  SockAddr addr;

  int listen_socket;
  int recv_epoll_fd;
  int send_epoll_fd;

  std::map<int64_t, std::shared_ptr<SockQp>> qps;
  std::map<int64_t, std::shared_ptr<SockMr>> mrs;
  std::map<int64_t, int> qpn_to_send_fd;
  std::map<int64_t, int> qpn_to_recv_fd;
  std::map<int, int64_t> fd_to_qpn;
  std::map<int, SendRecvContext> send_recv_contexts;

  std::atomic<bool> running;
  std::thread recv_thread;
  std::thread send_thread;

  std::mutex send_queue_mutex;
  std::deque<std::shared_ptr<std::deque<SockWr>>> global_send_queue;
};

}; // namespace pccl