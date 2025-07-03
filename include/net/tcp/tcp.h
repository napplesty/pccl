#pragma once

#include <cstddef>
#include <cstdint>
#include <netinet/in.h>
#include <sys/socket.h>

#include <atomic>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <thread>

#include "registered_memory.h"
#include "types.h"

namespace pccl {

class TcpCtx;

struct TcpAddress {
  union {
    struct in_addr v4_ip;
    struct in6_addr v6_ip;
  };
  uint16_t port;
  sa_family_t family;
};

struct TcpQpHandle {
  TcpAddress addr;
  int qpn;
};

enum class TcpOpType : int8_t {
  READ,
  WRITE,
  ATOMIC_ADD,
  HAND_SHAKE,
  HAND_SHAKE_ACK,
};

enum class TcpStatus : int8_t {
  SUCCESS,
  ERROR,
  ERROR_MR_NOT_FOUND,
  ERROR_QP_STATE,
};

struct TcpMr {
  uintptr_t addr;
  ComponentTypeFlags component_flag;
  int mr_id;
};

struct TcpWr {
  OperatorId operator_id;
  OperationId operation_id;
  int local_qpn;
  int remote_qpn;
  TcpMr local_mr_info;
  TcpMr remote_mr_info;
  uint32_t size;
  uint32_t local_offset;
  uint32_t remote_offset;
  TcpOpType op_type;
  int atomic_value;
};

struct TcpMessageHeader {
  TcpOpType op_type;
  int src_qpn;
  int dst_qpn;
  union {
    struct {
      OperatorId op_id;
      OperationId operation_id;
      TcpMr dst_mr_info;
      TcpMr src_mr_info;
      size_t dst_offset;
      size_t src_offset;
      size_t size;
    } read_load;
    struct {
      OperatorId op_id;
      OperationId operation_id;
      TcpMr dst_mr_info;
      int atomic_value;
    } atomic_add;
    TcpQpHandle src_handle;
  };
};

struct TcpWc {
  OperatorId operator_id;
  OperationId operation_id;
  TcpStatus status;
  int imm_data;
};

class TcpQp : public std::enable_shared_from_this<TcpQp> {
public:
  ~TcpQp();
  void rtr();
  void rts();
  void stageLoad(const TcpMr &mr, const TcpMr &remote_mr, size_t size,
                 OperatorId op_id, OperationId operation_id, 
                 size_t src_offset, size_t dst_offset);
  void stageSend(const TcpMr &mr, const TcpMr &remote_mr, size_t size,
                 OperatorId op_id, OperationId operation_id, 
                 size_t srcOffset, size_t dstOffset);
  void stageAtomicAdd(const TcpMr &mr, const TcpMr &remote_mr,
                      OperatorId op_id, OperationId operation_id, 
                      size_t dst_offset, int add_val);
  TcpStatus postSend();
  int pollCq();
  TcpWc &getWcStatus(int idx);

private:
  TcpWr &stageOp();
  friend class SockCtx;
  TcpQp() = default;
  int maxCqSize;
  int maxWrqSize;
  std::mutex qp_mutex;
  std::shared_ptr<std::deque<TcpWr>> staged_operations;
  std::shared_ptr<std::deque<TcpWc>> pulled_wcs;
  bool rtr_;
  bool rts_;
  TcpQpHandle remote_handle;
  TcpQpHandle local_handle;

  std::weak_ptr<TcpCtx> ctx_;
};

class TcpCtx : public std::enable_shared_from_this<TcpCtx> {
public:
  TcpCtx(TcpAddress addr);
  ~TcpCtx();

  std::shared_ptr<TcpQp> createQp(int max_cq_size, int max_wr);
  std::shared_ptr<TcpMr> registerMr(RegisteredMemory memory);
private:
  friend class SockQp;
  struct SendRecvContext {
    ssize_t remain_size;
    std::shared_ptr<TcpMessageHeader> header;
  };

  std::shared_ptr<TcpQp> findQpByQpn(int qpn);
  void sendThreadCycle();
  void recvThreadCycle();

  std::shared_ptr<std::deque<TcpWr>> pop_from_global_send_queue();
  void push_to_global_send_queue(std::shared_ptr<std::deque<TcpWr>> wr_queue);

  bool handleHandshake(int fd, const TcpMessageHeader &header);
  bool handleHandshakeAck(int fd, int64_t local_qpn, int64_t remote_qpn);
  bool handlePartialSend(int fd, SendRecvContext &context);
  bool handlePartialRecv(int fd, SendRecvContext &context);
  bool handleSendMessageHeader(int fd, const TcpMessageHeader &header, size_t &bytes_sent);
  bool handleRecvMessageHeader(int fd, TcpMessageHeader &header, size_t &bytes_recv);

private:
  std::mutex ctx_mutex;
  static std::atomic<int64_t> next_mr_id;
  static std::atomic<int64_t> next_qp_id;

  TcpAddress addr;

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

} // namespace pccl

