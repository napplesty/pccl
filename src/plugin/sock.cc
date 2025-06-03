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

// 前向声明
static std::shared_ptr<MessageHeader>
createMessageHeader(const SockWr &wr, const SockAddr &local_addr,
                    const SockAddr &remote_addr);

PCCL_API SockMr::~SockMr() {
  // SockMr的析构函数，context会自动清理
  auto shared_ctx = ctx.lock();
  if (shared_ctx) {
    shared_ctx->unregisterMr(mr_id);
  }
}

PCCL_API SockQp::~SockQp() {
  auto shared_ctx = ctx.lock();
  if (shared_ctx) {
    shared_ctx->unregister_qpn(qpn);
  }
}

PCCL_API SockStatus SockQp::connect(const SockQpInfo &remote_info) {
  this->remote_info = remote_info;
  is_connected = ctx.lock()->qp_connect(remote_info, qpn);
  if (!is_connected) {
    return SockStatus::ERROR_GENERAL;
  }
  return SockStatus::SUCCESS;
}

PCCL_API void SockQp::stageLoad(const SockMr *mr, const SockMrInfo &info,
                                size_t size, uint64_t wrId, uint64_t srcOffset,
                                uint64_t dstOffset) {
  SockWr &wr = stageOp();
  wr.wr_id = wrId;
  wr.op_type = SockOpType::READ;
  wr.local_mr_info = {mr->addr, mr->mr_id, mr->is_host_memory};
  wr.remote_mr_info = info;
  wr.local_offset = dstOffset;
  wr.remote_offset = srcOffset;
  wr.size = size;
  wr.imm = 0;
}

PCCL_API void SockQp::stageSend(const SockMr *mr, const SockMrInfo &info,
                                uint32_t size, uint64_t wrId,
                                uint64_t srcOffset, uint64_t dstOffset) {
  SockWr &wr = stageOp();
  wr.wr_id = wrId;
  wr.op_type = SockOpType::WRITE;
  wr.local_mr_info = {mr->addr, mr->mr_id, mr->is_host_memory};
  wr.remote_mr_info = info;
  wr.local_offset = srcOffset;
  wr.remote_offset = dstOffset;
  wr.size = size;
  wr.imm = 0;
}

PCCL_API void SockQp::stageAtomicAdd(const SockMr *mr, const SockMrInfo &info,
                                     uint64_t wrId, uint64_t dstOffset,
                                     uint64_t addVal) {
  SockWr &wr = stageOp();
  wr.wr_id = wrId;
  wr.op_type = SockOpType::ATOMIC_ADD;
  wr.local_mr_info = {mr->addr, mr->mr_id, mr->is_host_memory};
  wr.remote_mr_info = info;
  wr.remote_offset = dstOffset;
  wr.atomic_value = addVal;
}

PCCL_API void SockQp::stageSendWithImm(const SockMr *mr, const SockMrInfo &info,
                                       uint32_t size, uint64_t wrId,
                                       uint64_t srcOffset, uint64_t dstOffset,
                                       unsigned int immData) {
  SockWr &wr = stageOp();
  wr.wr_id = wrId;
  wr.op_type = SockOpType::WRITE_WITH_IMM;
  wr.local_mr_info = {mr->addr, mr->mr_id, mr->is_host_memory};
  wr.remote_mr_info = info;
  wr.local_offset = srcOffset;
  wr.remote_offset = dstOffset;
  wr.size = size;
  wr.imm = immData;
}

PCCL_API SockStatus SockQp::postSend() {
  if (staged_operations->empty()) {
    return SockStatus::SUCCESS;
  }

  auto shared_ctx = ctx.lock();
  if (!shared_ctx || !is_connected) {
    return SockStatus::ERROR_QP_STATE;
  }

  std::lock_guard<std::mutex> lock(qp_mutex);
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

PCCL_API SockWc &SockQp::getWcStatus(int idx) {
  if (!pulled_wcs || idx >= pulled_wcs->size()) {
    throw std::runtime_error("SockQp::getWcStatus: invalid index");
  }
  return pulled_wcs->at(idx);
}

PCCL_API SockQpInfo SockQp::getInfo() const {
  SockQpInfo info;
  info.qpn = qpn;
  info.addr = local_addr;
  return info;
}

PCCL_API SockWr &SockQp::stageOp() {
  std::lock_guard<std::mutex> lock(qp_mutex);
  if (staged_operations->size() < (size_t)maxWrqSize) {
    int idx = staged_operations->size();
    staged_operations->emplace_back(SockWr());
    staged_operations->at(idx).local_qpn = qpn;
    staged_operations->at(idx).remote_qpn = remote_info.qpn;
    return staged_operations->at(idx);
  }
  throw std::runtime_error("SockQp::stageOp: staged_operations is full");
}

PCCL_API SockCtx::SockCtx(SockAddr addr) : addr(addr), running(true) {
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
  mr->mr_id = next_mr_id++;
  mr->is_host_memory = is_host_memory;
  mr->ctx = weak_from_this();

  std::lock_guard<std::mutex> lock(ctx_mutex);
  mrs[mr->mr_id] = mr;

  return mr;
}

void SockCtx::unregisterMr(int64_t mr_id) {
  std::lock_guard<std::mutex> lock(ctx_mutex);
  mrs.erase(mr_id);
}

std::shared_ptr<SockQp> SockCtx::findQpByQpn(int64_t qpn) {
  auto it = qps.find(qpn);
  return (it != qps.end()) ? it->second : nullptr;
}

void SockCtx::register_qpn(int64_t qpn, std::shared_ptr<SockQp> qp) {
  std::lock_guard<std::mutex> lock(ctx_mutex);
  qps[qpn] = qp;
}

void SockCtx::unregister_qpn(int64_t qpn) {
  std::lock_guard<std::mutex> lock(ctx_mutex);
  qps.erase(qpn);

  // 清理对应的MR
  if (auto qp = findQpByQpn(qpn)) {
    // 这里可以添加清理QP相关MR的逻辑
  }

  if (qpn_to_send_fd.find(qpn) != qpn_to_send_fd.end()) {
    int fd = qpn_to_send_fd[qpn];
    close(fd);
    qpn_to_send_fd.erase(qpn);
    fd_to_qpn.erase(fd);
    send_recv_contexts.erase(fd);
  }

  if (qpn_to_recv_fd.find(qpn) != qpn_to_recv_fd.end()) {
    int fd = qpn_to_recv_fd[qpn];
    close(fd);
    qpn_to_recv_fd.erase(qpn);
    fd_to_qpn.erase(fd);
    send_recv_contexts.erase(fd);
  }
}

std::shared_ptr<std::deque<SockWr>> SockCtx::pop_from_global_send_queue() {
  std::lock_guard<std::mutex> lock(send_queue_mutex);
  if (global_send_queue.empty()) {
    return nullptr;
  }
  auto wr_queue = global_send_queue.front();
  global_send_queue.pop_front();
  return wr_queue;
}

void SockCtx::push_to_global_send_queue(
    std::shared_ptr<std::deque<SockWr>> wr_queue) {
  std::lock_guard<std::mutex> lock(send_queue_mutex);
  global_send_queue.push_back(wr_queue);
}

bool SockCtx::qp_connect(const SockQpInfo &remote_info, int64_t local_qpn) {
  int sock_fd = socket(remote_info.addr.family, SOCK_STREAM, 0);
  if (sock_fd < 0) {
    return false;
  }

  fcntl(sock_fd, F_SETFL, O_NONBLOCK);

  int result;
  if (remote_info.addr.family == AF_INET) {
    struct sockaddr_in sockaddr_v4;
    memset(&sockaddr_v4, 0, sizeof(sockaddr_v4));
    sockaddr_v4.sin_family = AF_INET;
    sockaddr_v4.sin_addr = remote_info.addr.v4.ip;
    sockaddr_v4.sin_port = htons(remote_info.addr.v4.port);

    result =
        connect(sock_fd, (struct sockaddr *)&sockaddr_v4, sizeof(sockaddr_v4));
  } else {
    struct sockaddr_in6 sockaddr_v6;
    memset(&sockaddr_v6, 0, sizeof(sockaddr_v6));
    sockaddr_v6.sin6_family = AF_INET6;
    sockaddr_v6.sin6_addr = remote_info.addr.v6.ip;
    sockaddr_v6.sin6_port = htons(remote_info.addr.v6.port);

    result =
        connect(sock_fd, (struct sockaddr *)&sockaddr_v6, sizeof(sockaddr_v6));
  }

  if (result < 0 && errno != EINPROGRESS) {
    close(sock_fd);
    return false;
  }

  // 发送握手消息
  SockWr handshake_wr;
  handshake_wr.op_type = SockOpType::HAND_SHAKE;
  handshake_wr.local_qpn = local_qpn;
  handshake_wr.remote_qpn = remote_info.qpn;

  auto header = createMessageHeader(handshake_wr, addr, remote_info.addr);
  if (sendHeader(sock_fd, *header) != SockStatus::SUCCESS) {
    close(sock_fd);
    return false;
  }

  std::lock_guard<std::mutex> lock(ctx_mutex);
  qpn_to_send_fd[local_qpn] = sock_fd;
  qpn_to_recv_fd[local_qpn] = sock_fd;
  fd_to_qpn[sock_fd] = local_qpn;

  struct epoll_event ev;
  ev.events = EPOLLIN | EPOLLOUT;
  ev.data.fd = sock_fd;
  epoll_ctl(recv_epoll_fd, EPOLL_CTL_ADD, sock_fd, &ev);
  epoll_ctl(send_epoll_fd, EPOLL_CTL_ADD, sock_fd, &ev);

  return true;
}

SockStatus SockCtx::sendHeader(int socket_fd, const MessageHeader &header) {
  size_t bytes_sent = 0;
  const char *data = reinterpret_cast<const char *>(&header);
  size_t total_size = sizeof(MessageHeader);

  while (bytes_sent < total_size) {
    ssize_t result = send(socket_fd, data + bytes_sent, total_size - bytes_sent,
                          MSG_NOSIGNAL);
    if (result < 0) {
      if (errno == EAGAIN || errno == EWOULDBLOCK) {
        continue;
      }
      return SockStatus::ERROR_GENERAL;
    }
    bytes_sent += result;
  }

  return SockStatus::SUCCESS;
}

SockStatus SockCtx::sendData(int socket_fd, const MessageHeader &header) {
  if (header.size == 0) {
    return SockStatus::SUCCESS;
  }

  auto mr_it = mrs.find(header.src_mr_info.mr_id);
  if (mr_it == mrs.end()) {
    return SockStatus::ERROR_MR_NOT_FOUND;
  }

  const char *data = reinterpret_cast<const char *>(header.src_mr_info.addr +
                                                    header.src_offset);
  size_t bytes_sent = 0;

  while (bytes_sent < header.size) {
    ssize_t result = send(socket_fd, data + bytes_sent,
                          header.size - bytes_sent, MSG_NOSIGNAL);
    if (result < 0) {
      if (errno == EAGAIN || errno == EWOULDBLOCK) {
        continue;
      }
      return SockStatus::ERROR_GENERAL;
    }
    bytes_sent += result;
  }

  return SockStatus::SUCCESS;
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
    header->sock_addr = local_addr;
  } else {
    header->dst_mr_info = wr.remote_mr_info;
    header->dst_offset = wr.remote_offset;
    header->src_mr_info = wr.local_mr_info;
    header->src_offset = wr.local_offset;
    header->size = wr.size;
  }
  if (wr.op_type == SockOpType::WRITE_WITH_IMM) {
    header->imm_data = wr.imm;
  } else if (wr.op_type == SockOpType::ATOMIC_ADD) {
    header->atomic_value = wr.atomic_value;
  }
  return header;
}

bool SockCtx::handleHandshake(int fd, const MessageHeader &header) {
  if (header.op_type == SockOpType::HAND_SHAKE) {
    return sendHandshakeAck(fd, header.dst_qpn, header.src_qpn);
  } else if (header.op_type == SockOpType::HAND_SHAKE_ACK) {
    std::lock_guard<std::mutex> lock(ctx_mutex);
    fd_to_qpn[fd] = header.dst_qpn;
    return true;
  }
  return false;
}

bool SockCtx::sendHandshakeAck(int fd, int64_t local_qpn, int64_t remote_qpn) {
  SockWr ack_wr;
  ack_wr.op_type = SockOpType::HAND_SHAKE_ACK;
  ack_wr.local_qpn = local_qpn;
  ack_wr.remote_qpn = remote_qpn;

  auto header = createMessageHeader(ack_wr, addr, {});
  return sendHeader(fd, *header) == SockStatus::SUCCESS;
}

bool SockCtx::sendMessageHeader(int fd, const MessageHeader &header,
                                size_t &bytes_sent) {
  const char *data = reinterpret_cast<const char *>(&header);
  size_t total_size = sizeof(MessageHeader);

  ssize_t result =
      send(fd, data + bytes_sent, total_size - bytes_sent, MSG_NOSIGNAL);
  if (result < 0) {
    if (errno == EAGAIN || errno == EWOULDBLOCK) {
      return false; // 稍后重试
    }
    return false; // 错误
  }

  bytes_sent += result;
  return bytes_sent >= total_size;
}

bool SockCtx::sendMessageData(int fd, const SendRecvContext &context,
                              size_t &bytes_sent) {
  if (context.header->size == 0) {
    return true;
  }

  auto mr_it = mrs.find(context.header->src_mr_info.mr_id);
  if (mr_it == mrs.end()) {
    return false;
  }

  const char *data = reinterpret_cast<const char *>(
      context.header->src_mr_info.addr + context.header->src_offset);

  ssize_t result = send(fd, data + bytes_sent,
                        context.header->size - bytes_sent, MSG_NOSIGNAL);
  if (result < 0) {
    if (errno == EAGAIN || errno == EWOULDBLOCK) {
      return false; // 稍后重试
    }
    return false; // 错误
  }

  bytes_sent += result;
  return bytes_sent >= context.header->size;
}

bool SockCtx::recvMessageHeader(int fd, MessageHeader &header,
                                size_t &bytes_recv) {
  char *data = reinterpret_cast<char *>(&header);
  size_t total_size = sizeof(MessageHeader);

  ssize_t result = recv(fd, data + bytes_recv, total_size - bytes_recv, 0);
  if (result <= 0) {
    if (result == 0 || (errno != EAGAIN && errno != EWOULDBLOCK)) {
      return false; // 连接关闭或错误
    }
    return false; // 稍后重试
  }

  bytes_recv += result;
  return bytes_recv >= total_size;
}

bool SockCtx::recvMessageData(int fd, SendRecvContext &context,
                              size_t &bytes_recv) {
  if (context.header->size == 0) {
    return true;
  }

  auto mr_it = mrs.find(context.header->dst_mr_info.mr_id);
  if (mr_it == mrs.end()) {
    return false;
  }

  char *data = reinterpret_cast<char *>(context.header->dst_mr_info.addr +
                                        context.header->dst_offset);

  ssize_t result =
      recv(fd, data + bytes_recv, context.header->size - bytes_recv, 0);
  if (result <= 0) {
    if (result == 0 || (errno != EAGAIN && errno != EWOULDBLOCK)) {
      return false; // 连接关闭或错误
    }
    return false; // 稍后重试
  }

  bytes_recv += result;
  return bytes_recv >= context.header->size;
}

void SockCtx::sendThreadCycle() {
  struct epoll_event events[64];

  while (running.load()) {
    // 处理全局发送队列中的新操作
    auto wr_queue = pop_from_global_send_queue();
    if (wr_queue) {
      for (const auto &wr : *wr_queue) {
        std::lock_guard<std::mutex> lock(ctx_mutex);
        if (qpn_to_send_fd.find(wr.local_qpn) != qpn_to_send_fd.end()) {
          int fd = qpn_to_send_fd[wr.local_qpn];
          auto header = createMessageHeader(wr, addr, {});

          SendRecvContext context;
          context.header = header;
          context.remain_size = sizeof(MessageHeader) + header->size;
          send_recv_contexts[fd] = context;
        }
      }
    }

    // 处理epoll事件
    int nfds = epoll_wait(send_epoll_fd, events, 64, 100);
    for (int i = 0; i < nfds; i++) {
      int fd = events[i].data.fd;
      if (events[i].events & EPOLLOUT) {
        if (send_recv_contexts.find(fd) != send_recv_contexts.end()) {
          processPartialSend(fd, send_recv_contexts[fd]);
        }
      }
    }
  }
}

void SockCtx::recvThreadCycle() {
  struct epoll_event events[64];

  while (running.load()) {
    int nfds = epoll_wait(recv_epoll_fd, events, 64, 100);

    for (int i = 0; i < nfds; i++) {
      int fd = events[i].data.fd;

      if (fd == listen_socket) {
        // 处理新连接
        struct sockaddr_storage client_addr;
        socklen_t addr_len = sizeof(client_addr);
        int client_fd =
            accept(listen_socket, (struct sockaddr *)&client_addr, &addr_len);

        if (client_fd >= 0) {
          fcntl(client_fd, F_SETFL, O_NONBLOCK);

          struct epoll_event ev;
          ev.events = EPOLLIN;
          ev.data.fd = client_fd;
          epoll_ctl(recv_epoll_fd, EPOLL_CTL_ADD, client_fd, &ev);
        }
      } else if (events[i].events & EPOLLIN) {
        // 处理数据接收
        if (send_recv_contexts.find(fd) == send_recv_contexts.end()) {
          // 新的消息，开始接收header
          SendRecvContext context;
          context.header = std::make_shared<MessageHeader>();
          context.remain_size = sizeof(MessageHeader);
          send_recv_contexts[fd] = context;
        }

        processPartialRecv(fd, send_recv_contexts[fd]);
      }
    }
  }
}

bool SockCtx::processPartialSend(int fd, SendRecvContext &context) {
  size_t bytes_sent =
      sizeof(MessageHeader) + context.header->size - context.remain_size;

  if (bytes_sent < sizeof(MessageHeader)) {
    // 还在发送header
    size_t header_bytes_sent = bytes_sent;
    if (sendMessageHeader(fd, *context.header, header_bytes_sent)) {
      context.remain_size -= (header_bytes_sent - bytes_sent);
      bytes_sent = header_bytes_sent;
    }
  }

  if (bytes_sent >= sizeof(MessageHeader) && context.remain_size > 0) {
    // 开始发送数据
    size_t data_bytes_sent = bytes_sent - sizeof(MessageHeader);
    if (sendMessageData(fd, context, data_bytes_sent)) {
      context.remain_size = 0;

      // 发送完成，生成完成事件
      std::lock_guard<std::mutex> lock(ctx_mutex);
      if (fd_to_qpn.find(fd) != fd_to_qpn.end()) {
        int64_t qpn = fd_to_qpn[fd];
        auto qp = findQpByQpn(qpn);
        if (qp) {
          SockWc wc;
          wc.wr_id = context.header->src_wr_id;
          wc.status = SockStatus::SUCCESS;
          wc.imm_data = 0;

          std::lock_guard<std::mutex> qp_lock(qp->qp_mutex);
          qp->wc_queue->push_back(wc);
        }
      }

      send_recv_contexts.erase(fd);
      return true;
    }
  }

  return context.remain_size == 0;
}

bool SockCtx::processPartialRecv(int fd, SendRecvContext &context) {
  size_t bytes_recv =
      sizeof(MessageHeader) + context.header->size - context.remain_size;

  if (bytes_recv < sizeof(MessageHeader)) {
    // 还在接收header
    size_t header_bytes_recv = bytes_recv;
    if (recvMessageHeader(fd, *context.header, header_bytes_recv)) {
      context.remain_size -= (header_bytes_recv - bytes_recv);
      bytes_recv = header_bytes_recv;

      // Header接收完成，检查是否是握手消息
      if (context.header->op_type == SockOpType::HAND_SHAKE ||
          context.header->op_type == SockOpType::HAND_SHAKE_ACK) {
        handleHandshake(fd, *context.header);
        send_recv_contexts.erase(fd);
        return true;
      }

      // 更新剩余大小以包含数据部分
      context.remain_size = context.header->size;
    }
  }

  if (bytes_recv >= sizeof(MessageHeader) && context.remain_size > 0) {
    // 开始接收数据
    size_t data_bytes_recv = bytes_recv - sizeof(MessageHeader);
    if (recvMessageData(fd, context, data_bytes_recv)) {
      context.remain_size = 0;

      // 接收完成，生成完成事件
      std::lock_guard<std::mutex> lock(ctx_mutex);
      if (fd_to_qpn.find(fd) != fd_to_qpn.end()) {
        int64_t qpn = fd_to_qpn[fd];
        auto qp = findQpByQpn(qpn);
        if (qp) {
          SockWc wc;
          wc.wr_id = context.header->src_wr_id;
          wc.status = SockStatus::SUCCESS;
          wc.imm_data = context.header->op_type == SockOpType::WRITE_WITH_IMM
                            ? context.header->imm_data
                            : 0;

          std::lock_guard<std::mutex> qp_lock(qp->qp_mutex);
          qp->wc_queue->push_back(wc);
        }
      }

      send_recv_contexts.erase(fd);
      return true;
    }
  }

  return context.remain_size == 0;
}

} // namespace pccl