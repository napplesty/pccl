#include "plugin/sock.h"
#include "component/logging.h"
#include "utils.h"
#include <arpa/inet.h>
#include <cstdint>
#include <cstring>
#include <errno.h>
#include <fcntl.h>
#include <mutex>
#include <stdexcept>
#include <sys/epoll.h>
#include <thread>
#include <unistd.h>

namespace pccl {

std::atomic<int64_t> SockCtx::next_mr_id{0};
std::atomic<int64_t> SockCtx::next_qp_id{0};

PCCL_API SockQp::~SockQp() {
  auto shared_ctx = ctx.lock();
  if (shared_ctx) {
    shared_ctx->unregister_qpn(qpn);
  }
}

PCCL_API SockStatus SockQp::connect(const SockQpInfo &remote_info) {
  this->remote_info = remote_info;
  is_connected = ctx.lock()->qp_connect(remote_info, qpn); // 传递本地QPN
  if (!is_connected) {
    return SockStatus::ERROR_GENERAL;
  }
  return SockStatus::SUCCESS;
}

PCCL_API void SockQp::stageLoad(const SockMr *mr, const SockMrInfo &info,
                                size_t size, uint64_t wrId, uint64_t srcOffset,
                                uint64_t dstOffset, bool signaled) {
  SockWr &wr = stageOp();
  wr.wr_id = wrId;
  wr.op_type = SockOpType::READ;
  wr.local_mr_info = {mr->addr, mr->mr_id, static_cast<uint32_t>(mr->size),
                      mr->is_host_memory};
  wr.remote_mr_info = info;
  wr.size = size;
  wr.local_offset = dstOffset;
  wr.remote_offset = srcOffset;
  wr.imm = 0;
  wr.signaled = signaled;
}

PCCL_API void SockQp::stageSend(const SockMr *mr, const SockMrInfo &info,
                                uint32_t size, uint64_t wrId,
                                uint64_t srcOffset, uint64_t dstOffset,
                                bool signaled) {
  SockWr &wr = stageOp();
  wr.wr_id = wrId;
  wr.op_type = SockOpType::WRITE;
  wr.local_mr_info = {mr->addr, mr->mr_id, static_cast<uint32_t>(mr->size),
                      mr->is_host_memory};
  wr.remote_mr_info = info;
  wr.size = size;
  wr.local_offset = srcOffset;
  wr.remote_offset = dstOffset;
  wr.imm = 0;
  wr.signaled = signaled;
}

PCCL_API void SockQp::stageAtomicAdd(const SockMr *mr, const SockMrInfo &info,
                                     uint64_t wrId, uint64_t dstOffset,
                                     uint64_t addVal, bool signaled) {
  SockWr &wr = stageOp();
  wr.wr_id = wrId;
  wr.op_type = SockOpType::ATOMIC_ADD;
  wr.local_mr_info = {mr->addr, mr->mr_id, static_cast<uint32_t>(mr->size),
                      mr->is_host_memory};
  wr.remote_mr_info = info;
  wr.size = sizeof(uint64_t);
  wr.remote_offset = dstOffset;
  wr.atomic_value = addVal;
  wr.imm = 0;
  wr.signaled = signaled;
}

PCCL_API void SockQp::stageSendWithImm(const SockMr *mr, const SockMrInfo &info,
                                       uint32_t size, uint64_t wrId,
                                       uint64_t srcOffset, uint64_t dstOffset,
                                       bool signaled, unsigned int immData) {
  SockWr &wr = stageOp();
  wr.wr_id = wrId;
  wr.op_type = SockOpType::WRITE_WITH_IMM;
  wr.local_mr_info = {mr->addr, mr->mr_id, static_cast<uint32_t>(mr->size),
                      mr->is_host_memory};
  wr.remote_mr_info = info;
  wr.size = size;
  wr.local_offset = srcOffset;
  wr.remote_offset = dstOffset;
  wr.imm = immData;
  wr.signaled = signaled;
}

PCCL_API SockStatus SockQp::postSend() {
  if (staged_operations->empty()) {
    return SockStatus::SUCCESS;
  }

  auto shared_ctx = ctx.lock();
  if (!shared_ctx || !is_connected) {
    return SockStatus::ERROR_QP_STATE;
  }

  shared_ctx->push_to_global_send_queue(std::move(staged_operations));
  staged_operations = std::make_shared<std::deque<SockWr>>();
  return SockStatus::SUCCESS;
}

PCCL_API int SockQp::pollCq() {
  std::lock_guard<std::mutex> lock(qp_mutex);
  pulled_wcs = std::move(wc_queue);
  wc_queue = std::make_shared<std::deque<SockWc>>();
  return pulled_wcs->size();
}

PCCL_API SockQpInfo SockQp::getInfo() const {
  SockQpInfo info;
  info.qpn = qpn;
  info.addr = local_addr;
  return info;
}

PCCL_API SockWr &SockQp::stageOp() {
  std::lock_guard<std::mutex> lock(qp_mutex);
  if (staged_operations->size() < maxWrqSize) {
    int idx = staged_operations->size();
    staged_operations->emplace_back(SockWr());
    staged_operations->at(idx).local_qpn = qpn;
    staged_operations->at(idx).remote_qpn = remote_info.qpn;
    return staged_operations->at(idx);
  }
  throw std::runtime_error("SockQp::stageOp: staged_operations is full");
}

PCCL_API SockCtx::SockCtx(SockAddr addr) : addr(addr), running(true) {
  std::lock_guard<std::mutex> lock(ctx_mutex);

  listen_socket = socket(addr.family, SOCK_STREAM, 0);
  if (listen_socket < 0) {
    throw std::runtime_error("sockqp_lib create listen socket failed");
  }

  int opt = 1;
  setsockopt(listen_socket, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
  fcntl(listen_socket, F_SETFL, O_NONBLOCK);
  if (addr.family == AF_INET) {
    struct sockaddr_in sockaddr_v4;
    memset(&sockaddr_v4, 0, sizeof(sockaddr_v4));
    sockaddr_v4.sin_family = AF_INET;
    sockaddr_v4.sin_addr = addr.v4.ip;
    sockaddr_v4.sin_port = htons(addr.v4.port);

    if (bind(listen_socket, (struct sockaddr *)&sockaddr_v4,
             sizeof(sockaddr_v4)) < 0) {
      close(listen_socket);
      throw std::runtime_error("sockqp_lib bind address failed");
    }
  } else if (addr.family == AF_INET6) {
    struct sockaddr_in6 sockaddr_v6;
    memset(&sockaddr_v6, 0, sizeof(sockaddr_v6));
    sockaddr_v6.sin6_family = AF_INET6;
    sockaddr_v6.sin6_addr = addr.v6.ip;
    sockaddr_v6.sin6_port = htons(addr.v6.port);

    if (bind(listen_socket, (struct sockaddr *)&sockaddr_v6,
             sizeof(sockaddr_v6)) < 0) {
      close(listen_socket);
      throw std::runtime_error("sockqp_lib bind address failed");
    }
  }

  if (listen(listen_socket, 256) < 0) {
    close(listen_socket);
    throw std::runtime_error("sockqp_lib listen failed");
  }

  recv_epoll_fd = epoll_create1(0);
  send_epoll_fd = epoll_create1(0);

  if (recv_epoll_fd < 0 || send_epoll_fd < 0) {
    close(listen_socket);
    throw std::runtime_error("sockqp_lib create epoll failed");
  }

  struct epoll_event ev;
  ev.events = EPOLLIN;
  ev.data.fd = listen_socket;
  if (epoll_ctl(recv_epoll_fd, EPOLL_CTL_ADD, listen_socket, &ev) < 0) {
    close(listen_socket);
    close(recv_epoll_fd);
    close(send_epoll_fd);
    throw std::runtime_error("sockqp_lib add listen socket to epoll failed");
  }

  recv_thread = std::thread(&SockCtx::recvThreadCycle, this);
  send_thread = std::thread(&SockCtx::sendThreadCycle, this);
}

PCCL_API SockCtx::~SockCtx() {
  for (auto &qpn_qp : qps) {
    int64_t qpn = qpn_qp.first;
    auto qp = qpn_qp.second;
    std::lock_guard<std::mutex> lock(qp->qp_mutex);
    qp->is_connected = false;

    if (qpn_to_send_fd.find(qpn) != qpn_to_send_fd.end()) {
      close(qpn_to_send_fd[qpn]);
      qpn_to_send_fd.erase(qpn);
    }

    if (qpn_to_recv_fd.find(qpn) != qpn_to_recv_fd.end()) {
      close(qpn_to_recv_fd[qpn]);
      qpn_to_recv_fd.erase(qpn);
    }
  }

  std::lock_guard<std::mutex> lock(ctx_mutex);
  running = false;

  if (recv_thread.joinable()) {
    recv_thread.join();
  }
  if (send_thread.joinable()) {
    send_thread.join();
  }

  close(listen_socket);
  close(recv_epoll_fd);
  close(send_epoll_fd);
}

std::shared_ptr<SockQp> SockCtx::createQp(int max_cq_size, int max_wr) {
  auto qp = std::shared_ptr<SockQp>(new SockQp());
  qp->maxCqSize = max_cq_size;
  qp->maxWrqSize = max_wr;
  qp->ctx = weak_from_this();
  qp->qpn = next_qp_id++;
  qp->is_connected = false;
  qp->staged_operations = std::make_shared<std::deque<SockWr>>();
  qp->wc_queue = std::make_shared<std::deque<SockWc>>();
  qp->local_addr = addr;

  register_qpn(qp->qpn, qp);
  return qp;
}

std::shared_ptr<SockMr> SockCtx::registerMr(void *buff, size_t size,
                                            bool is_host_memory) {
  auto mr = std::make_shared<SockMr>();
  mr->addr = reinterpret_cast<uintptr_t>(buff);
  mr->size = size;
  mr->mr_id = next_mr_id++;
  mr->is_host_memory = is_host_memory;
  mr->ctx = weak_from_this();

  std::lock_guard<std::mutex> lock(ctx_mutex);
  mrs[mr->mr_id] = mr;

  return mr;
}

std::shared_ptr<SockQp> SockCtx::findQpByQpn(int64_t qpn) {
  auto it = qps.find(qpn);
  return (it != qps.end()) ? it->second : nullptr;
}

static std::shared_ptr<MessageHeader>
createMessageHeader(const SockWr &wr, const SockAddr &local_addr,
                    const SockAddr &remote_addr) {
  auto header = std::make_shared<MessageHeader>();
  header->op_type = wr.op_type;
  header->src_wr_id = wr.wr_id;
  header->src_qpn = wr.local_qpn;
  header->dst_qpn = wr.remote_qpn;
  if (wr.op_type == SockOpType::HAND_SHAKE) {
    header->sock_addr = remote_addr;
  } else if (wr.op_type == SockOpType::HAND_SHAKE_ACK) {
    header->sock_addr =
  } else {
    header->dst_mr_info = wr.remote_mr_info;
    header->dst_offset = wr.remote_offset;
    header->src_mr_info = wr.local_mr_info;
    header->src_offset = wr.local_offset;
    return header;
  }

} // namespace pccl