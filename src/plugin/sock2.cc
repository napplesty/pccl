#include "plugin/sock2.h"

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/epoll.h>
#include <sys/socket.h>
#include <unistd.h>

#include "component/logging.h"

namespace {

static int setNonBlocking(int fd) {
  int flags = fcntl(fd, F_GETFL, 0);
  if (flags == -1) {
    return -1;
  }
  return fcntl(fd, F_SETFL, flags | O_NONBLOCK);
}

static int addToEpoll(int epoll_fd, int fd, uint32_t events) {
  struct epoll_event ev;
  ev.events = events;
  ev.data.fd = fd;
  return epoll_ctl(epoll_fd, EPOLL_CTL_ADD, fd, &ev);
}

static int removeFromEpoll(int epoll_fd, int fd) {
  return epoll_ctl(epoll_fd, EPOLL_CTL_DEL, fd, nullptr);
}

}  // namespace

namespace pccl {

SockQp::~SockQp() {
  if (send_fd > 0) {
    close(send_fd);
  }
  if (recv_fd > 0) {
    close(recv_fd);
  }
}

SockStatus SockQp::connect(const SockQpInfo &remote_info, std::shared_ptr<SockCtx> ctx) {
  this->remote_info = remote_info;
  this->ctx = ctx;

  int sock_fd = socket(remote_info.addr.family, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
  if (sock_fd < 0) {
    LOG_ERROR << "Failed to create socket for QP connection";
    return SockStatus::ERROR_CONN_FAILED;
  }

  int opt = 1;
  setsockopt(sock_fd, IPPROTO_TCP, TCP_NODELAY, &opt, sizeof(opt));

  int tcp_keepalive = 1;
  setsockopt(sock_fd, SOL_SOCKET, SO_KEEPALIVE, &tcp_keepalive, sizeof(tcp_keepalive));

  // Connect to remote server (non-blocking)
  struct sockaddr_storage addr;
  socklen_t addr_len;

  if (remote_info.addr.family == AF_INET) {
    struct sockaddr_in *sa = reinterpret_cast<struct sockaddr_in *>(&addr);
    memset(sa, 0, sizeof(*sa));
    sa->sin_family = AF_INET;
    sa->sin_addr.s_addr = remote_info.addr.v4.ip;
    sa->sin_port = remote_info.addr.v4.port;
    addr_len = sizeof(*sa);
  } else {
    struct sockaddr_in6 *sa = reinterpret_cast<struct sockaddr_in6 *>(&addr);
    memset(sa, 0, sizeof(*sa));
    sa->sin6_family = AF_INET6;
    sa->sin6_addr = remote_info.addr.v6.ip;
    sa->sin6_port = remote_info.addr.v6.port;
    addr_len = sizeof(*sa);
  }

  if (::connect(sock_fd, reinterpret_cast<struct sockaddr *>(&addr), addr_len) < 0) {
    if (errno != EINPROGRESS) {
      LOG_ERROR << "Failed to connect to remote QP: " << strerror(errno);
      close(sock_fd);
      return SockStatus::ERROR_CONN_FAILED;
    }
  }

  // Wait for connection to be established
  struct pollfd pfd;
  pfd.fd = sock_fd;
  pfd.events = POLLOUT;

  int poll_res = poll(&pfd, 1, 5000);  // 5 seconds timeout

  if (poll_res <= 0) {
    LOG_ERROR << "Connection timeout or error";
    close(sock_fd);
    return SockStatus::ERROR_CONN_FAILED;
  }

  if (pfd.revents & (POLLERR | POLLHUP)) {
    LOG_ERROR << "Connection failed";
    close(sock_fd);
    return SockStatus::ERROR_CONN_FAILED;
  }

  // Check if connection was successful
  int error = 0;
  socklen_t len = sizeof(error);
  if (getsockopt(sock_fd, SOL_SOCKET, SO_ERROR, &error, &len) < 0 || error != 0) {
    LOG_ERROR << "Connect failed: " << strerror(error);
    close(sock_fd);
    return SockStatus::ERROR_CONN_FAILED;
  }

  send_fd = sock_fd;

  // Send CONNECT message to remote
  MessageHeader header;
  memset(&header, 0, sizeof(header));
  header.type = MessageType::CONNECT;
  header.qpn = qpn;
  header.addr = ctx->addr;  // Our address for the remote to connect back

  ssize_t sent = send(send_fd, &header, sizeof(header), MSG_NOSIGNAL);
  if (sent != sizeof(header)) {
    LOG_ERROR << "Failed to send CONNECT message";
    close(send_fd);
    send_fd = -1;
    return SockStatus::ERROR_CONN_FAILED;
  }

  // Receive CONNECT_ACK from remote
  MessageHeader ack_header;
  ssize_t received = 0;

  while (received < sizeof(ack_header)) {
    ssize_t res = recv(send_fd, reinterpret_cast<char *>(&ack_header) + received,
                       sizeof(ack_header) - received, 0);

    if (res <= 0) {
      if (res < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
        // Wait for data to be available
        pfd.fd = send_fd;
        pfd.events = POLLIN;

        if (poll(&pfd, 1, 5000) <= 0) {
          LOG_ERROR << "Timeout waiting for CONNECT_ACK";
          close(send_fd);
          send_fd = -1;
          return SockStatus::ERROR_CONN_FAILED;
        }

        continue;
      }

      LOG_ERROR << "Failed to receive CONNECT_ACK";
      close(send_fd);
      send_fd = -1;
      return SockStatus::ERROR_CONN_FAILED;
    }

    received += res;
  }

  if (ack_header.type != MessageType::CONNECT_ACK) {
    LOG_ERROR << "Received unexpected message type instead of CONNECT_ACK";
    close(send_fd);
    send_fd = -1;
    return SockStatus::ERROR_CONN_FAILED;
  }

  // Keep track of the socket in the context
  recv_fd = send_fd;  // In this implementation, we use the same socket for simplicity
  is_connected = true;

  return SockStatus::SUCCESS;
}

void SockQp::stageLoad(const SockMr *mr, const SockMrInfo &remote_mr_info, size_t size,
                       uint64_t wrId, uint64_t srcOffset, uint64_t dstOffset, bool signaled) {
  if (!is_connected) {
    LOG_ERROR << "QP not connected";
    return;
  }

  SockWr wr;
  wr.wr_id = wrId;
  wr.op_type = SockOpType::READ;

  // Local MR info
  wr.local_mr_info.addr = mr->addr;
  wr.local_mr_info.size = mr->size;
  wr.local_mr_info.mr_id = mr->mr_id;
  wr.local_mr_info.dev_type = mr->is_host_memory ? DeviceType::HOST : DeviceType::CUDA;

  // Remote MR info
  wr.remote_mr_info = remote_mr_info;

  wr.size = size;
  wr.local_offset = srcOffset;
  wr.remote_offset = dstOffset;
  wr.signaled = signaled;

  if (wr_queue->size() >= maxWrqSize) {
    LOG_ERROR << "Work request queue is full";
    return;
  }

  wr_queue->push_back(wr);
}

void SockQp::stageSend(const SockMr *mr, const SockMrInfo &remote_mr_info, uint32_t size,
                       uint64_t wrId, uint64_t srcOffset, uint64_t dstOffset, bool signaled) {
  if (!is_connected) {
    LOG_ERROR << "QP not connected";
    return;
  }

  SockWr wr;
  wr.wr_id = wrId;
  wr.op_type = SockOpType::WRITE;

  // Local MR info
  wr.local_mr_info.addr = mr->addr;
  wr.local_mr_info.size = mr->size;
  wr.local_mr_info.mr_id = mr->mr_id;
  wr.local_mr_info.dev_type = mr->is_host_memory ? DeviceType::HOST : DeviceType::CUDA;

  // Remote MR info
  wr.remote_mr_info = remote_mr_info;

  wr.size = size;
  wr.local_offset = srcOffset;
  wr.remote_offset = dstOffset;
  wr.signaled = signaled;

  if (wr_queue->size() >= maxWrqSize) {
    LOG_ERROR << "Work request queue is full";
    return;
  }

  wr_queue->push_back(wr);
}

void SockQp::stageAtomicAdd(const SockMr *mr, const SockMrInfo &remote_mr_info, uint64_t wrId,
                            uint64_t dstOffset, uint64_t addVal, bool signaled) {
  if (!is_connected) {
    LOG_ERROR << "QP not connected";
    return;
  }

  SockWr wr;
  wr.wr_id = wrId;
  wr.op_type = SockOpType::ATOMIC_ADD;

  // Local MR info (not used for atomic operations, but we fill it anyway)
  wr.local_mr_info.addr = mr->addr;
  wr.local_mr_info.size = mr->size;
  wr.local_mr_info.mr_id = mr->mr_id;
  wr.local_mr_info.dev_type = mr->is_host_memory ? DeviceType::HOST : DeviceType::CUDA;

  // Remote MR info
  wr.remote_mr_info = remote_mr_info;

  wr.size = sizeof(uint64_t);  // Atomic operations are always 8 bytes
  wr.remote_offset = dstOffset;
  wr.signaled = signaled;
  wr.atomic_value = addVal;

  if (wr_queue->size() >= maxWrqSize) {
    LOG_ERROR << "Work request queue is full";
    return;
  }

  wr_queue->push_back(wr);
}

void SockQp::stageSendWithImm(const SockMr *mr, const SockMrInfo &remote_mr_info, uint32_t size,
                              uint64_t wrId, uint64_t srcOffset, uint64_t dstOffset, bool signaled,
                              unsigned int immData) {
  if (!is_connected) {
    LOG_ERROR << "QP not connected";
    return;
  }

  SockWr wr;
  wr.wr_id = wrId;
  wr.op_type = SockOpType::WRITE_WITH_IMM;

  // Local MR info
  wr.local_mr_info.addr = mr->addr;
  wr.local_mr_info.size = mr->size;
  wr.local_mr_info.mr_id = mr->mr_id;
  wr.local_mr_info.dev_type = mr->is_host_memory ? DeviceType::HOST : DeviceType::CUDA;

  // Remote MR info
  wr.remote_mr_info = remote_mr_info;

  wr.size = size;
  wr.local_offset = srcOffset;
  wr.remote_offset = dstOffset;
  wr.signaled = signaled;
  wr.imm_data = immData;

  if (wr_queue->size() >= maxWrqSize) {
    LOG_ERROR << "Work request queue is full";
    return;
  }

  wr_queue->push_back(wr);
}

SockStatus SockQp::postSend() {
  if (!is_connected) {
    return SockStatus::ERROR_QP_STATE;
  }

  if (wr_queue->empty()) {
    return SockStatus::SUCCESS;  // Nothing to do
  }

  for (auto it = wr_queue->begin(); it != wr_queue->end(); /* increment inside loop */) {
    SockWr &wr = *it;

    // Prepare headers
    MessageHeader header;
    memset(&header, 0, sizeof(header));
    header.type = MessageType::REQUEST;
    header.op_type = wr.op_type;
    header.status = SockStatus::SUCCESS;
    header.imm_data = wr.imm_data;
    header.wr_id = wr.wr_id;
    header.qpn = qpn;

    MessagePayloadHeader payload_header;
    memset(&payload_header, 0, sizeof(payload_header));
    payload_header.mr_info = wr.remote_mr_info;
    payload_header.offset = wr.remote_offset;
    payload_header.size = wr.size;

    SockStatus status;

    if (wr.op_type == SockOpType::READ) {
      // Send request to read from remote memory
      status = sendReadRequest(header, payload_header, wr);
    } else if (wr.op_type == SockOpType::WRITE || wr.op_type == SockOpType::WRITE_WITH_IMM) {
      // Send request to write to remote memory
      status = sendWriteRequest(header, payload_header, wr);
    } else if (wr.op_type == SockOpType::ATOMIC_ADD) {
      // Send atomic add request
      status = sendAtomicRequest(header, payload_header, wr);
    } else {
      LOG_ERROR << "Unknown operation type";
      status = SockStatus::ERROR_INVALID_PARAM;
    }

    if (status != SockStatus::SUCCESS) {
      return status;
    }

    // Process completions
    if (wr.signaled) {
      SockWc wc;
      wc.wr_id = wr.wr_id;
      wc.status = SockStatus::SUCCESS;
      wc.imm_data = wr.imm_data;
      wc_queue->push_back(wc);
    }

    // Remove processed WR
    it = wr_queue->erase(it);
  }

  return SockStatus::SUCCESS;
}

int SockQp::pollCq() {
  if (wc_queue->empty()) {
    return 0;
  }

  int count = wc_queue->size();
  wc_queue->clear();
  return count;
}

bool SockQp::active() const { return is_connected; }

int SockQp::sendSockFd() const { return send_fd; }

int SockQp::recvSockFd() const { return recv_fd; }

// Helper methods for SockQp (private)
SockStatus SockQp::sendReadRequest(MessageHeader &header, MessagePayloadHeader &payload_header,
                                   const SockWr &wr) {
  // Send READ request
  ssize_t sent = send(send_fd, &header, sizeof(header), MSG_NOSIGNAL);
  if (sent != sizeof(header)) {
    LOG_ERROR << "Failed to send READ request header";
    return SockStatus::ERROR_SEND_FAILED;
  }

  sent = send(send_fd, &payload_header, sizeof(payload_header), MSG_NOSIGNAL);
  if (sent != sizeof(payload_header)) {
    LOG_ERROR << "Failed to send READ request payload header";
    return SockStatus::ERROR_SEND_FAILED;
  }

  // Wait for response
  MessageHeader resp_header;
  ssize_t received = 0;

  // Receive response header
  while (received < sizeof(resp_header)) {
    ssize_t res = recv(send_fd, reinterpret_cast<char *>(&resp_header) + received,
                       sizeof(resp_header) - received, 0);

    if (res <= 0) {
      if (res < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
        // Wait for data to be available
        struct pollfd pfd;
        pfd.fd = send_fd;
        pfd.events = POLLIN;

        if (poll(&pfd, 1, 5000) <= 0) {
          LOG_ERROR << "Timeout waiting for READ response";
          return SockStatus::ERROR_TIMEOUT;
        }

        continue;
      }

      LOG_ERROR << "Failed to receive READ response header";
      return SockStatus::ERROR_RECV_FAILED;
    }

    received += res;
  }

  if (resp_header.type == MessageType::ERROR) {
    LOG_ERROR << "READ request failed with status: " << static_cast<int>(resp_header.status);
    return resp_header.status;
  }

  if (resp_header.type != MessageType::RESPONSE) {
    LOG_ERROR << "Unexpected response type for READ request";
    return SockStatus::ERROR_GENERAL;
  }

  // Receive the data
  void *data_ptr = reinterpret_cast<void *>(wr.local_mr_info.addr + wr.local_offset);
  received = 0;

  while (received < wr.size) {
    ssize_t res =
        recv(send_fd, reinterpret_cast<char *>(data_ptr) + received, wr.size - received, 0);

    if (res <= 0) {
      if (res < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
        // Wait for data to be available
        struct pollfd pfd;
        pfd.fd = send_fd;
        pfd.events = POLLIN;

        if (poll(&pfd, 1, 5000) <= 0) {
          LOG_ERROR << "Timeout waiting for READ data";
          return SockStatus::ERROR_TIMEOUT;
        }

        continue;
      }

      LOG_ERROR << "Failed to receive READ data";
      return SockStatus::ERROR_RECV_FAILED;
    }

    received += res;
  }

  return SockStatus::SUCCESS;
}

SockStatus SockQp::sendWriteRequest(MessageHeader &header, MessagePayloadHeader &payload_header,
                                    const SockWr &wr) {
  // Send WRITE request
  ssize_t sent = send(send_fd, &header, sizeof(header), MSG_NOSIGNAL);
  if (sent != sizeof(header)) {
    LOG_ERROR << "Failed to send WRITE request header";
    return SockStatus::ERROR_SEND_FAILED;
  }

  sent = send(send_fd, &payload_header, sizeof(payload_header), MSG_NOSIGNAL);
  if (sent != sizeof(payload_header)) {
    LOG_ERROR << "Failed to send WRITE request payload header";
    return SockStatus::ERROR_SEND_FAILED;
  }

  // Send the data
  void *data_ptr = reinterpret_cast<void *>(wr.local_mr_info.addr + wr.local_offset);
  sent = send(send_fd, data_ptr, wr.size, MSG_NOSIGNAL);
  if (sent != wr.size) {
    LOG_ERROR << "Failed to send WRITE data";
    return SockStatus::ERROR_SEND_FAILED;
  }

  // Wait for response
  MessageHeader resp_header;
  ssize_t received = 0;

  // Receive response header
  while (received < sizeof(resp_header)) {
    ssize_t res = recv(send_fd, reinterpret_cast<char *>(&resp_header) + received,
                       sizeof(resp_header) - received, 0);

    if (res <= 0) {
      if (res < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
        // Wait for data to be available
        struct pollfd pfd;
        pfd.fd = send_fd;
        pfd.events = POLLIN;

        if (poll(&pfd, 1, 5000) <= 0) {
          LOG_ERROR << "Timeout waiting for WRITE response";
          return SockStatus::ERROR_TIMEOUT;
        }

        continue;
      }

      LOG_ERROR << "Failed to receive WRITE response header";
      return SockStatus::ERROR_RECV_FAILED;
    }

    received += res;
  }

  if (resp_header.type == MessageType::ERROR) {
    LOG_ERROR << "WRITE request failed with status: " << static_cast<int>(resp_header.status);
    return resp_header.status;
  }

  if (resp_header.type != MessageType::RESPONSE) {
    LOG_ERROR << "Unexpected response type for WRITE request";
    return SockStatus::ERROR_GENERAL;
  }

  return SockStatus::SUCCESS;
}

SockStatus SockQp::sendAtomicRequest(MessageHeader &header, MessagePayloadHeader &payload_header,
                                     const SockWr &wr) {
  // Send ATOMIC_ADD request
  ssize_t sent = send(send_fd, &header, sizeof(header), MSG_NOSIGNAL);
  if (sent != sizeof(header)) {
    LOG_ERROR << "Failed to send ATOMIC_ADD request header";
    return SockStatus::ERROR_SEND_FAILED;
  }

  sent = send(send_fd, &payload_header, sizeof(payload_header), MSG_NOSIGNAL);
  if (sent != sizeof(payload_header)) {
    LOG_ERROR << "Failed to send ATOMIC_ADD request payload header";
    return SockStatus::ERROR_SEND_FAILED;
  }

  // Send the atomic value
  sent = send(send_fd, &wr.atomic_value, sizeof(wr.atomic_value), MSG_NOSIGNAL);
  if (sent != sizeof(wr.atomic_value)) {
    LOG_ERROR << "Failed to send ATOMIC_ADD value";
    return SockStatus::ERROR_SEND_FAILED;
  }

  // Wait for response
  MessageHeader resp_header;
  ssize_t received = 0;

  // Receive response header
  while (received < sizeof(resp_header)) {
    ssize_t res = recv(send_fd, reinterpret_cast<char *>(&resp_header) + received,
                       sizeof(resp_header) - received, 0);

    if (res <= 0) {
      if (res < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
        // Wait for data to be available
        struct pollfd pfd;
        pfd.fd = send_fd;
        pfd.events = POLLIN;

        if (poll(&pfd, 1, 5000) <= 0) {
          LOG_ERROR << "Timeout waiting for ATOMIC_ADD response";
          return SockStatus::ERROR_TIMEOUT;
        }

        continue;
      }

      LOG_ERROR << "Failed to receive ATOMIC_ADD response header";
      return SockStatus::ERROR_RECV_FAILED;
    }

    received += res;
  }

  if (resp_header.type == MessageType::ERROR) {
    LOG_ERROR << "ATOMIC_ADD request failed with status: " << static_cast<int>(resp_header.status);
    return resp_header.status;
  }

  if (resp_header.type != MessageType::RESPONSE) {
    LOG_ERROR << "Unexpected response type for ATOMIC_ADD request";
    return SockStatus::ERROR_GENERAL;
  }

  return SockStatus::SUCCESS;
}

std::atomic<uint32_t> SockCtx::next_mr_id(1);
std::atomic<int> SockCtx::next_qp_id(1);

SockCtx::SockCtx(SockAddr addr) : addr(addr), running(true) {
  recv_epoll_fd = epoll_create1(EPOLL_CLOEXEC);
  if (recv_epoll_fd < 0) {
    LOG_ERROR << "Failed to create recv epoll fd";
    throw std::runtime_error("Failed to create recv epoll fd");
  }

  send_epoll_fd = epoll_create1(EPOLL_CLOEXEC);
  if (send_epoll_fd < 0) {
    close(recv_epoll_fd);
    LOG_ERROR << "Failed to create send epoll fd";
    throw std::runtime_error("Failed to create send epoll fd");
  }

  if (addr.family == AF_INET) {
    listen_socket = socket(AF_INET, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
    if (listen_socket < 0) {
      close(recv_epoll_fd);
      close(send_epoll_fd);
      LOG_ERROR << "Failed to create listening socket";
      throw std::runtime_error("Failed to create listening socket");
    }

    int opt = 1;
    setsockopt(listen_socket, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    setsockopt(listen_socket, SOL_SOCKET, SO_REUSEPORT, &opt, sizeof(opt));

    int tcp_keepalive = 1;
    setsockopt(listen_socket, SOL_SOCKET, SO_KEEPALIVE, &tcp_keepalive, sizeof(tcp_keepalive));

    setsockopt(listen_socket, IPPROTO_TCP, TCP_NODELAY, &opt, sizeof(opt));

    int tcp_quickack = 1;
    setsockopt(listen_socket, IPPROTO_TCP, TCP_QUICKACK, &tcp_quickack, sizeof(tcp_quickack));

    int rcvbuf_size = 1024 * 1024;
    int sndbuf_size = 1024 * 1024;
    setsockopt(listen_socket, SOL_SOCKET, SO_RCVBUF, &rcvbuf_size, sizeof(rcvbuf_size));
    setsockopt(listen_socket, SOL_SOCKET, SO_SNDBUF, &sndbuf_size, sizeof(sndbuf_size));

    if (setNonBlocking(listen_socket) < 0) {
      close(listen_socket);
      close(recv_epoll_fd);
      close(send_epoll_fd);
      LOG_ERROR << "Failed to set non-blocking mode for listen socket";
      throw std::runtime_error("Failed to set non-blocking mode for listen socket");
    }

    struct sockaddr_in server_addr;
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = addr.v4.ip;
    server_addr.sin_port = addr.v4.port;

    if (bind(listen_socket, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
      close(listen_socket);
      close(recv_epoll_fd);
      close(send_epoll_fd);
      LOG_ERROR << "Failed to bind listening socket";
      throw std::runtime_error("Failed to bind listening socket");
    }

    if (listen(listen_socket, SOMAXCONN) < 0) {
      close(listen_socket);
      close(recv_epoll_fd);
      close(send_epoll_fd);
      LOG_ERROR << "Failed to listen on socket";
      throw std::runtime_error("Failed to listen on socket");
    }

    if (addToEpoll(recv_epoll_fd, listen_socket, EPOLLIN | EPOLLET | EPOLLEXCLUSIVE) < 0) {
      close(listen_socket);
      close(recv_epoll_fd);
      close(send_epoll_fd);
      LOG_ERROR << "Failed to add listening socket to epoll";
      throw std::runtime_error("Failed to add listening socket to epoll");
    }
  } else if (addr.family == AF_INET6) {
    listen_socket = socket(AF_INET6, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
    if (listen_socket < 0) {
      close(recv_epoll_fd);
      close(send_epoll_fd);
      LOG_ERROR << "Failed to create IPv6 listening socket";
      throw std::runtime_error("Failed to create IPv6 listening socket");
    }

    int opt = 1;
    setsockopt(listen_socket, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    setsockopt(listen_socket, SOL_SOCKET, SO_REUSEPORT, &opt, sizeof(opt));

    int tcp_keepalive = 1;
    setsockopt(listen_socket, SOL_SOCKET, SO_KEEPALIVE, &tcp_keepalive, sizeof(tcp_keepalive));

    setsockopt(listen_socket, IPPROTO_TCP, TCP_NODELAY, &opt, sizeof(opt));

    int tcp_quickack = 1;
    setsockopt(listen_socket, IPPROTO_TCP, TCP_QUICKACK, &tcp_quickack, sizeof(tcp_quickack));

    int rcvbuf_size = 1024 * 1024;
    int sndbuf_size = 1024 * 1024;
    setsockopt(listen_socket, SOL_SOCKET, SO_RCVBUF, &rcvbuf_size, sizeof(rcvbuf_size));
    setsockopt(listen_socket, SOL_SOCKET, SO_SNDBUF, &sndbuf_size, sizeof(sndbuf_size));

    int v6only = 0;
    setsockopt(listen_socket, IPPROTO_IPV6, IPV6_V6ONLY, &v6only, sizeof(v6only));

    if (setNonBlocking(listen_socket) < 0) {
      close(listen_socket);
      close(recv_epoll_fd);
      close(send_epoll_fd);
      LOG_ERROR << "Failed to set non-blocking mode for IPv6 listen socket";
      throw std::runtime_error("Failed to set non-blocking mode for IPv6 listen socket");
    }

    struct sockaddr_in6 server_addr;
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin6_family = AF_INET6;
    server_addr.sin6_addr = addr.v6.ip;
    server_addr.sin6_port = addr.v6.port;

    if (bind(listen_socket, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
      close(listen_socket);
      close(recv_epoll_fd);
      close(send_epoll_fd);
      LOG_ERROR << "Failed to bind IPv6 listening socket";
      throw std::runtime_error("Failed to bind IPv6 listening socket");
    }

    if (listen(listen_socket, SOMAXCONN) < 0) {
      close(listen_socket);
      close(recv_epoll_fd);
      close(send_epoll_fd);
      LOG_ERROR << "Failed to listen on IPv6 socket";
      throw std::runtime_error("Failed to listen on IPv6 socket");
    }

    if (addToEpoll(recv_epoll_fd, listen_socket, EPOLLIN | EPOLLET | EPOLLEXCLUSIVE) < 0) {
      close(listen_socket);
      close(recv_epoll_fd);
      close(send_epoll_fd);
      LOG_ERROR << "Failed to add IPv6 listening socket to epoll";
      throw std::runtime_error("Failed to add IPv6 listening socket to epoll");
    }
  } else {
    close(recv_epoll_fd);
    close(send_epoll_fd);
    LOG_ERROR << "Unsupported address family";
    throw std::runtime_error("Unsupported address family");
  }

  recv_thread = std::thread(&SockCtx::recvThreadCycle, this);
  send_thread = std::thread(&SockCtx::sendThreadCycle, this);
}

std::shared_ptr<SockQp> SockCtx::createQp(int max_cq_size, int max_wr) {
  auto qp = std::make_shared<SockQp>();
  qp->maxCqSize = max_cq_size;
  qp->maxWrqSize = max_wr;
  qp->ctx = shared_from_this();
  qp->wr_queue = std::make_shared<std::deque<SockWr>>();
  qp->wc_queue = std::make_shared<std::deque<SockWc>>();

  int qpn = next_qp_id.fetch_add(1);
  qps[qpn] = qp;

  return qp;
}

std::shared_ptr<SockMr> SockCtx::registerMr(void *buff, size_t size, bool isHostMemory) {
  auto mr = std::make_shared<SockMr>();
  mr->addr = reinterpret_cast<uintptr_t>(buff);
  mr->size = size;
  mr->mr_id = next_mr_id.fetch_add(1);
  mr->is_host_memory = isHostMemory;

  mrs[mr->mr_id] = mr;

  return mr;
}

std::shared_ptr<SockQp> SockCtx::findQpByQpn(int qpn) {
  auto it = qps.find(qpn);
  if (it != qps.end()) {
    return it->second;
  }
  return nullptr;
}

SockStatus SockCtx::sendData(int socket_fd, MessageHeader &header, MessagePayloadHeader &payload,
                             std::shared_ptr<SockMr> mr, uint32_t offset, uint32_t size) {
  if (offset + size > mr->size) {
    return SockStatus::ERROR_INVALID_PARAM;
  }

  // Send header
  ssize_t sent = send(socket_fd, &header, sizeof(header), MSG_NOSIGNAL);
  if (sent != sizeof(header)) {
    return SockStatus::ERROR_SEND_FAILED;
  }

  // Send payload header
  sent = send(socket_fd, &payload, sizeof(payload), MSG_NOSIGNAL);
  if (sent != sizeof(payload)) {
    return SockStatus::ERROR_SEND_FAILED;
  }

  // Send actual data
  void *data_ptr = reinterpret_cast<void *>(mr->addr + offset);
  sent = send(socket_fd, data_ptr, size, MSG_NOSIGNAL);
  if (sent != size) {
    return SockStatus::ERROR_SEND_FAILED;
  }

  return SockStatus::SUCCESS;
}

void SockCtx::manageConnections() {
  struct epoll_event events[64];

  while (running) {
    int nfds = epoll_wait(recv_epoll_fd, events, 64, 100);
    if (nfds < 0) {
      if (errno == EINTR) continue;
      LOG_ERROR << "Epoll wait failed in manageConnections: " << strerror(errno);
      break;
    }

    for (int i = 0; i < nfds; i++) {
      if (events[i].data.fd == listen_socket) {
        // Handle new connections
        while (true) {
          struct sockaddr_storage client_addr;
          socklen_t client_len = sizeof(client_addr);

          int client_fd = accept4(listen_socket, (struct sockaddr *)&client_addr, &client_len,
                                  SOCK_NONBLOCK | SOCK_CLOEXEC);
          if (client_fd < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
              break;  // No more connections to accept
            }
            LOG_ERROR << "Accept failed: " << strerror(errno);
            break;
          }

          // Set socket options
          int opt = 1;
          setsockopt(client_fd, IPPROTO_TCP, TCP_NODELAY, &opt, sizeof(opt));

          int tcp_keepalive = 1;
          setsockopt(client_fd, SOL_SOCKET, SO_KEEPALIVE, &tcp_keepalive, sizeof(tcp_keepalive));

          // Add to epoll
          if (addToEpoll(recv_epoll_fd, client_fd, EPOLLIN | EPOLLET) < 0) {
            LOG_ERROR << "Failed to add client socket to recv epoll";
            close(client_fd);
            continue;
          }

          // Wait for connection message
          // This will be handled in recvThreadCycle
        }
      }
    }
  }
}

void SockCtx::sendThreadCycle() {
  struct epoll_event events[64];
  std::map<int, std::shared_ptr<SocketTask>> pending_tasks;

  while (running) {
    int nfds = epoll_wait(send_epoll_fd, events, 64, 100);
    if (nfds < 0) {
      if (errno == EINTR) continue;
      LOG_ERROR << "Epoll wait failed in sendThreadCycle: " << strerror(errno);
      break;
    }

    for (int i = 0; i < nfds; i++) {
      int fd = events[i].data.fd;
      auto it = pending_tasks.find(fd);

      if (it == pending_tasks.end()) {
        continue;
      }

      auto task = it->second;

      if (events[i].events & EPOLLOUT) {
        // Continue sending data
        if (task->sended_size < sizeof(MessageHeader)) {
          // Send header
          ssize_t sent = send(fd, reinterpret_cast<char *>(task->header.get()) + task->sended_size,
                              sizeof(MessageHeader) - task->sended_size, MSG_NOSIGNAL);
          if (sent < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
              continue;
            }
            LOG_ERROR << "Failed to send header: " << strerror(errno);
            pending_tasks.erase(it);
            continue;
          }
          task->sended_size += sent;
        } else if (task->sended_size < sizeof(MessageHeader) + sizeof(MessagePayloadHeader)) {
          // Send payload header
          ssize_t sent =
              send(fd,
                   reinterpret_cast<char *>(task->payload_header.get()) +
                       (task->sended_size - sizeof(MessageHeader)),
                   sizeof(MessagePayloadHeader) - (task->sended_size - sizeof(MessageHeader)),
                   MSG_NOSIGNAL);
          if (sent < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
              continue;
            }
            LOG_ERROR << "Failed to send payload header: " << strerror(errno);
            pending_tasks.erase(it);
            continue;
          }
          task->sended_size += sent;
        } else {
          // Task completed, remove from pending
          pending_tasks.erase(it);

          // Process completion if needed
          auto qp = findQpByQpn(task->header->qpn);
          if (qp && task->header->wr_id != 0) {
            SockWc wc;
            wc.wr_id = task->header->wr_id;
            wc.status = SockStatus::SUCCESS;
            wc.imm_data = task->header->imm_data;
            qp->wc_queue->push_back(wc);
          }
        }
      }

      if (events[i].events & (EPOLLERR | EPOLLHUP)) {
        LOG_ERROR << "Socket error in sendThreadCycle";
        pending_tasks.erase(it);
      }
    }

    // Check for new tasks from QPs
    for (auto &qp_entry : qps) {
      auto qp = qp_entry.second;
      if (!qp->wr_queue->empty()) {
        auto &wr = qp->wr_queue->front();
        // Process work request
        // This would be implemented in QP's postSend method
        qp->wr_queue->pop_front();
      }
    }
  }
}

void SockCtx::recvThreadCycle() {
  struct epoll_event events[64];
  std::map<int, std::shared_ptr<SocketTask>> pending_tasks;

  while (running) {
    int nfds = epoll_wait(recv_epoll_fd, events, 64, 100);
    if (nfds < 0) {
      if (errno == EINTR) continue;
      LOG_ERROR << "Epoll wait failed in recvThreadCycle: " << strerror(errno);
      break;
    }

    for (int i = 0; i < nfds; i++) {
      if (events[i].data.fd == listen_socket) {
        continue;  // Handled in manageConnections
      }

      int fd = events[i].data.fd;
      auto it = pending_tasks.find(fd);

      if (it == pending_tasks.end() && (events[i].events & EPOLLIN)) {
        // New message
        auto task = std::make_shared<SocketTask>();
        task->header = std::make_shared<MessageHeader>();
        task->payload_header = std::make_shared<MessagePayloadHeader>();
        task->recv_fd = fd;
        task->recved_size = 0;

        pending_tasks[fd] = task;
        it = pending_tasks.find(fd);
      }

      if (it != pending_tasks.end() && (events[i].events & EPOLLIN)) {
        auto task = it->second;

        if (task->recved_size < sizeof(MessageHeader)) {
          // Receive header
          ssize_t received =
              recv(fd, reinterpret_cast<char *>(task->header.get()) + task->recved_size,
                   sizeof(MessageHeader) - task->recved_size, 0);
          if (received <= 0) {
            if (received == 0 || errno != EAGAIN) {
              LOG_ERROR << "Failed to receive header: "
                        << (received == 0 ? "Connection closed" : strerror(errno));
              pending_tasks.erase(it);
            }
            continue;
          }
          task->recved_size += received;

          if (task->recved_size == sizeof(MessageHeader)) {
            // Header complete, check message type
            if (task->header->type == MessageType::CONNECT) {
              // Handle connection request
              int qpn = task->header->qpn;
              qpn_to_recv_fd[qpn] = fd;

              // Create send socket
              int send_fd =
                  socket(task->header->addr.family, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
              if (send_fd < 0) {
                LOG_ERROR << "Failed to create send socket for QP";
                pending_tasks.erase(it);
                continue;
              }

              // Set socket options
              int opt = 1;
              setsockopt(send_fd, IPPROTO_TCP, TCP_NODELAY, &opt, sizeof(opt));

              struct sockaddr_storage addr;
              socklen_t addr_len;

              if (task->header->addr.family == AF_INET) {
                struct sockaddr_in *sa = reinterpret_cast<struct sockaddr_in *>(&addr);
                memset(sa, 0, sizeof(*sa));
                sa->sin_family = AF_INET;
                sa->sin_addr.s_addr = task->header->addr.v4.ip;
                sa->sin_port = task->header->addr.v4.port;
                addr_len = sizeof(*sa);
              } else {
                struct sockaddr_in6 *sa = reinterpret_cast<struct sockaddr_in6 *>(&addr);
                memset(sa, 0, sizeof(*sa));
                sa->sin6_family = AF_INET6;
                sa->sin6_addr = task->header->addr.v6.ip;
                sa->sin6_port = task->header->addr.v6.port;
                addr_len = sizeof(*sa);
              }

              // Connect (non-blocking)
              if (connect(send_fd, reinterpret_cast<struct sockaddr *>(&addr), addr_len) < 0 &&
                  errno != EINPROGRESS) {
                LOG_ERROR << "Failed to connect send socket: " << strerror(errno);
                close(send_fd);
                pending_tasks.erase(it);
                continue;
              }

              // Add to epoll
              if (addToEpoll(send_epoll_fd, send_fd, EPOLLOUT | EPOLLET) < 0) {
                LOG_ERROR << "Failed to add send socket to epoll";
                close(send_fd);
                pending_tasks.erase(it);
                continue;
              }

              qpn_to_send_fd[qpn] = send_fd;

              // Send CONNECT_ACK
              MessageHeader ack_header = *task->header;
              ack_header.type = MessageType::CONNECT_ACK;
              ack_header.addr = addr;

              send(fd, &ack_header, sizeof(ack_header), MSG_NOSIGNAL);

              pending_tasks.erase(it);
              continue;
            }
          }
        } else if (task->recved_size < sizeof(MessageHeader) + sizeof(MessagePayloadHeader)) {
          // Receive payload header
          ssize_t received =
              recv(fd,
                   reinterpret_cast<char *>(task->payload_header.get()) +
                       (task->recved_size - sizeof(MessageHeader)),
                   sizeof(MessagePayloadHeader) - (task->recved_size - sizeof(MessageHeader)), 0);
          if (received <= 0) {
            if (received == 0 || errno != EAGAIN) {
              LOG_ERROR << "Failed to receive payload header";
              pending_tasks.erase(it);
            }
            continue;
          }
          task->recved_size += received;

          if (task->recved_size == sizeof(MessageHeader) + sizeof(MessagePayloadHeader)) {
            // Headers complete, process request
            int qpn = task->header->qpn;
            auto qp = findQpByQpn(qpn);

            if (!qp) {
              LOG_ERROR << "QP not found for qpn " << qpn;
              pending_tasks.erase(it);
              continue;
            }

            // Handle different message types
            if (task->header->type == MessageType::REQUEST) {
              // Process RDMA-like request
              uint32_t mr_id = task->payload_header->mr_info.mr_id;
              auto mr_it = mrs.find(mr_id);

              if (mr_it == mrs.end()) {
                // Send error response
                MessageHeader err_header = *task->header;
                err_header.type = MessageType::ERROR;
                err_header.status = SockStatus::ERROR_MR_NOT_FOUND;

                send(fd, &err_header, sizeof(err_header), MSG_NOSIGNAL);
                pending_tasks.erase(it);
                continue;
              }

              auto mr = mr_it->second;
              uint32_t offset = task->payload_header->offset;
              uint32_t size = task->payload_header->size;

              if (offset + size > mr->size) {
                // Send error response
                MessageHeader err_header = *task->header;
                err_header.type = MessageType::ERROR;
                err_header.status = SockStatus::ERROR_INVALID_PARAM;

                send(fd, &err_header, sizeof(err_header), MSG_NOSIGNAL);
                pending_tasks.erase(it);
                continue;
              }

              // Process operation
              if (task->header->op_type == SockOpType::READ) {
                // Client wants to read our memory
                void *data_ptr = reinterpret_cast<void *>(mr->addr + offset);

                // Send response header
                MessageHeader resp_header = *task->header;
                resp_header.type = MessageType::RESPONSE;
                resp_header.status = SockStatus::SUCCESS;

                send(fd, &resp_header, sizeof(resp_header), MSG_NOSIGNAL);

                // Send data
                send(fd, data_ptr, size, MSG_NOSIGNAL);

              } else if (task->header->op_type == SockOpType::WRITE ||
                         task->header->op_type == SockOpType::WRITE_WITH_IMM) {
                // Client wants to write to our memory
                // We need to receive the data
                task->mr = mr;
                continue;  // Keep the task alive to receive data
              } else if (task->header->op_type == SockOpType::ATOMIC_ADD) {
                // Atomic add operation
                if (size != sizeof(uint64_t)) {
                  // Send error response
                  MessageHeader err_header = *task->header;
                  err_header.type = MessageType::ERROR;
                  err_header.status = SockStatus::ERROR_INVALID_PARAM;

                  send(fd, &err_header, sizeof(err_header), MSG_NOSIGNAL);
                  pending_tasks.erase(it);
                  continue;
                }

                // Receive the value to add
                uint64_t add_value;
                ssize_t received = recv(fd, &add_value, sizeof(add_value), 0);

                if (received != sizeof(add_value)) {
                  LOG_ERROR << "Failed to receive atomic add value";
                  pending_tasks.erase(it);
                  continue;
                }

                // Perform atomic add
                uint64_t *target = reinterpret_cast<uint64_t *>(mr->addr + offset);
                __sync_fetch_and_add(target, add_value);

                // Send response
                MessageHeader resp_header = *task->header;
                resp_header.type = MessageType::RESPONSE;
                resp_header.status = SockStatus::SUCCESS;

                send(fd, &resp_header, sizeof(resp_header), MSG_NOSIGNAL);
              }

              pending_tasks.erase(it);
            }
          }
        } else {
          // Receiving data for WRITE operations
          auto mr = task->mr;
          auto header = task->header;
          auto payload_header = task->payload_header;

          uint32_t offset = payload_header->offset;
          uint32_t size = payload_header->size;
          uint32_t data_received =
              task->recved_size - sizeof(MessageHeader) - sizeof(MessagePayloadHeader);

          // Receive more data
          void *data_ptr = reinterpret_cast<void *>(mr->addr + offset + data_received);
          ssize_t received = recv(fd, data_ptr, size - data_received, 0);

          if (received <= 0) {
            if (received == 0 || errno != EAGAIN) {
              LOG_ERROR << "Failed to receive write data";
              pending_tasks.erase(it);
            }
            continue;
          }

          task->recved_size += received;

          if (task->recved_size == sizeof(MessageHeader) + sizeof(MessagePayloadHeader) + size) {
            // All data received, send response
            MessageHeader resp_header = *header;
            resp_header.type = MessageType::RESPONSE;
            resp_header.status = SockStatus::SUCCESS;

            send(fd, &resp_header, sizeof(resp_header), MSG_NOSIGNAL);

            // If this was WRITE_WITH_IMM, generate completion event
            if (header->op_type == SockOpType::WRITE_WITH_IMM) {
              auto qp = findQpByQpn(header->qpn);
              if (qp) {
                SockWc wc;
                wc.wr_id = 0;  // Special ID for received completions
                wc.status = SockStatus::SUCCESS;
                wc.imm_data = header->imm_data;
                qp->wc_queue->push_back(wc);
              }
            }

            pending_tasks.erase(it);
          }
        }
      }

      if (events[i].events & (EPOLLERR | EPOLLHUP)) {
        if (it != pending_tasks.end()) {
          pending_tasks.erase(it);
        }
        removeFromEpoll(recv_epoll_fd, fd);
        close(fd);
      }
    }
  }
}

// Destructor for SockCtx
SockCtx::~SockCtx() {
  running = false;

  if (recv_thread.joinable()) {
    recv_thread.join();
  }

  if (send_thread.joinable()) {
    send_thread.join();
  }

  if (connection_thread.joinable()) {
    connection_thread.join();
  }

  // Close all sockets
  for (const auto &pair : qpn_to_recv_fd) {
    close(pair.second);
  }

  for (const auto &pair : qpn_to_send_fd) {
    close(pair.second);
  }

  close(listen_socket);
  close(recv_epoll_fd);
  close(send_epoll_fd);
}

}  // namespace pccl