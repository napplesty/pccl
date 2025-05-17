#include "plugin/sock.h"

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/epoll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <cstring>
#include <stdexcept>

#include "component/logging.h"
#include "utils.h"

namespace pccl {

std::atomic<uint32_t> SockCtx::next_mr_id(1);
std::atomic<int> SockCtx::next_qp_id(1);

SockMr::SockMr(void* buff, size_t size, bool isHostMemory)
    : buff(buff), size(size), isHostMemory(isHostMemory) {}

SockMr::~SockMr() {}

SockMrInfo SockMr::getInfo() const {
  SockMrInfo info;
  info.addr = reinterpret_cast<uint64_t>(buff);
  return info;
}

void* SockMr::getBuff() const { return buff; }

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

SockQp::SockQp(SockCtx* ctx, int socket_fd, const std::string& host, int port, int max_cq_size,
               int max_wr)
    : ctx(ctx),
      socket_fd(socket_fd),
      connected(false),
      num_signaled_posted_items(0),
      num_signaled_staged_items(0),
      num_completed_items(0),
      max_cq_size(max_cq_size),
      max_wr(max_wr),
      recv_bytes(0) {
  info.host = host;
  info.port = port;
  info.qpn = SockCtx::next_qp_id.fetch_add(1);

  wcs.resize(max_cq_size);
  recv_buffer.resize(1024 * 1024);  // 1MB缓冲区
}

SockQp::~SockQp() {
  if (socket_fd >= 0) {
    close(socket_fd);
    socket_fd = -1;
  }
}

SockStatus SockCtx::sendData(int socket_fd, const void* data, size_t size) {
  updateSocketUsage(socket_fd);

  ssize_t sent = 0;
  const char* buffer = static_cast<const char*>(data);

  int retry_count = 0;
  constexpr int MAX_RETRIES = 32;

  while (sent < size) {
    ssize_t bytes = send(socket_fd, buffer + sent, size - sent, MSG_NOSIGNAL);
    if (bytes > 0) {
      sent += bytes;
      retry_count = 0;
    } else if (bytes == 0) {
      LOG_ERROR << "Connection closed while sending data";
      return SockStatus::ERROR_SEND_FAILED;
    } else {
      if (errno == EINTR) {
        continue;
      } else if (errno == EAGAIN || errno == EWOULDBLOCK) {
        retry_count++;
        if (retry_count >= MAX_RETRIES) {
          LOG_ERROR << "Send operation timed out after " << MAX_RETRIES << " retries";
          return SockStatus::ERROR_TIMEOUT;
        }

        struct pollfd pfd;
        pfd.fd = socket_fd;
        pfd.events = POLLOUT;

        int poll_result = poll(&pfd, 1, 10);
        if (poll_result < 0 && errno != EINTR) {
          LOG_ERROR << "Poll failed: " << strerror(errno);
          return SockStatus::ERROR_SEND_FAILED;
        }
      } else {
        LOG_ERROR << "Failed to send data: " << strerror(errno);
        return SockStatus::ERROR_SEND_FAILED;
      }
    }
  }

  return SockStatus::SUCCESS;
}

SockCtx::SockCtx(const std::string& host, int port, int max_connections)
    : host(host), port(port), running(true), max_connections_(max_connections) {
  epoll_fd = epoll_create1(0);
  if (epoll_fd < 0) {
    throw std::runtime_error("Failed to create epoll instance: " + std::string(strerror(errno)));
  }

  listen_socket = socket(AF_INET, SOCK_STREAM, 0);
  if (listen_socket < 0) {
    close(epoll_fd);
    throw std::runtime_error("Failed to create socket: " + std::string(strerror(errno)));
  }

  int opt = 1;
  if (setsockopt(listen_socket, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
    close(listen_socket);
    close(epoll_fd);
    throw std::runtime_error("Failed to set socket options: " + std::string(strerror(errno)));
  }

  if (setNonBlocking(listen_socket) < 0) {
    close(listen_socket);
    close(epoll_fd);
    throw std::runtime_error("Failed to set non-blocking mode: " + std::string(strerror(errno)));
  }

  struct sockaddr_in address;
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = inet_addr(host.c_str());
  address.sin_port = htons(port);

  if (bind(listen_socket, (struct sockaddr*)&address, sizeof(address)) < 0) {
    close(listen_socket);
    close(epoll_fd);
    throw std::runtime_error("Failed to bind socket: " + std::string(strerror(errno)));
  }

  if (listen(listen_socket, SOMAXCONN) < 0) {
    close(listen_socket);
    close(epoll_fd);
    throw std::runtime_error("Failed to listen on socket: " + std::string(strerror(errno)));
  }

  if (addToEpoll(epoll_fd, listen_socket, EPOLLIN) < 0) {
    close(listen_socket);
    close(epoll_fd);
    throw std::runtime_error("Failed to add socket to epoll: " + std::string(strerror(errno)));
  }

  worker_thread = std::thread(&SockCtx::processEvents, this);
}

SockCtx::~SockCtx() {
  running = false;

  if (epoll_fd >= 0) {
    close(epoll_fd);
  }
  if (listen_socket >= 0) {
    close(listen_socket);
  }

  if (worker_thread.joinable()) {
    worker_thread.join();
  }

  qps.clear();
  mrs.clear();
}

void SockCtx::updateSocketUsage(int fd) {
  std::lock_guard<std::mutex> lock(qps_mutex);
  auto it = socket_infos.find(fd);
  if (it != socket_infos.end()) {
    it->second.incrementUsage();
  }
}

std::shared_ptr<SockQp> SockCtx::findQpByFd(int fd) {
  std::lock_guard<std::mutex> lock(qps_mutex);
  auto it = socket_infos.find(fd);
  if (it == socket_infos.end() || it->second.type != SocketType::QP_SOCKET ||
      it->second.qp.expired()) {
    return nullptr;
  }
  return it->second.qp.lock();
}

std::shared_ptr<SockQp> SockCtx::findQpById(int qp_id) {
  std::lock_guard<std::mutex> lock(qps_mutex);

  auto fd_it = qp_id_to_fd.find(qp_id);
  if (fd_it == qp_id_to_fd.end()) {
    for (const auto& qp : qps) {
      if (qp->getInfo().qpn == qp_id) {
        return qp;
      }
    }
    return nullptr;
  }

  auto info_it = socket_infos.find(fd_it->second);
  if (info_it == socket_infos.end() || info_it->second.qp.expired()) {
    return nullptr;
  }

  return info_it->second.qp.lock();
}

SockStatus SockCtx::addSocketToEpoll(int fd) {
  if (addToEpoll(epoll_fd, fd, EPOLLIN) < 0) {
    LOG_ERROR << "Failed to add socket to epoll: " << strerror(errno);
    return SockStatus::ERROR_GENERAL;
  }
  return SockStatus::SUCCESS;
}

void SockCtx::updateSocketMapping(int fd, SockQp* qp) {
  std::lock_guard<std::mutex> lock(qps_mutex);
  for (const auto& q : qps) {
    if (q.get() == qp) {
      socket_infos.insert_or_assign(fd, SocketInfo(SocketType::QP_SOCKET, q));
      qp_id_to_fd[q->getInfo().qpn] = fd;
      return;
    }
  }
}

std::shared_ptr<SockQp> SockCtx::createQp(int max_cq_size, int max_wr) {
  int socket_fd = socket(AF_INET, SOCK_STREAM, 0);
  if (socket_fd < 0) {
    throw std::runtime_error("Failed to create socket: " + std::string(strerror(errno)));
  }

  int flag = 1;
  if (setsockopt(socket_fd, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(int)) < 0) {
    close(socket_fd);
    throw std::runtime_error("Failed to set TCP_NODELAY: " + std::string(strerror(errno)));
  }

  if (setNonBlocking(socket_fd) < 0) {
    close(socket_fd);
    throw std::runtime_error("Failed to set non-blocking mode: " + std::string(strerror(errno)));
  }

  qps.push_back(std::make_shared<SockQp>(this, socket_fd, host, port, max_cq_size, max_wr));
  std::shared_ptr<SockQp> qp = qps.back();

  if (addToEpoll(epoll_fd, socket_fd, EPOLLIN) < 0) {
    qps.pop_back();
    throw std::runtime_error("Failed to add socket to epoll: " + std::string(strerror(errno)));
  }

  std::lock_guard<std::mutex> lock(qps_mutex);
  socket_infos.insert_or_assign(socket_fd, SocketInfo(SocketType::QP_SOCKET, qp));
  qp_id_to_fd[qp->getInfo().qpn] = socket_fd;

  return qp;
}

std::shared_ptr<SockMr> SockCtx::registerMr(void* buff, size_t size, bool isHostMemory) {
  if (!buff || size == 0) {
    throw std::invalid_argument("Invalid buffer or size");
  }

  mrs.emplace_back(std::make_shared<SockMr>(buff, size, isHostMemory));
  return mrs.back();
}

void SockCtx::handleSocketEvent(int fd, uint32_t events) {
  std::lock_guard<std::mutex> lock(qps_mutex);

  auto it = socket_infos.find(fd);
  if (it == socket_infos.end()) {
    LOG_ERROR << "Event for unknown socket: " << fd;
    removeFromEpoll(epoll_fd, fd);
    close(fd);
    return;
  }

  it->second.incrementUsage();

  if (events & (EPOLLERR | EPOLLHUP)) {
    LOG_ERROR << "Socket error or closed: " << fd;
    if (it->second.type == SocketType::QP_SOCKET && !it->second.qp.expired()) {
      auto qp = it->second.qp.lock();
      qp->disconnect();
      int qp_id = qp->getInfo().qpn;
      qp_id_to_fd.erase(qp_id);
    } else {
      close(fd);
    }
    socket_infos.erase(it);
    removeFromEpoll(epoll_fd, fd);
    return;
  }

  if (events & EPOLLIN) {
    char buffer[4096];
    ssize_t bytes_read = recv(fd, buffer, sizeof(buffer), 0);

    if (bytes_read <= 0) {
      if (bytes_read == 0 || (errno != EAGAIN && errno != EWOULDBLOCK)) {
        LOG_ERROR << "Connection closed or error: "
                  << (bytes_read == 0 ? "Connection closed" : strerror(errno));

        // 根据socket类型处理
        if (it->second.type == SocketType::QP_SOCKET && !it->second.qp.expired()) {
          // QP socket: 断开连接但不删除QP对象
          auto qp = it->second.qp.lock();
          qp->disconnect();
          int qp_id = qp->getInfo().qpn;
          qp_id_to_fd.erase(qp_id);
        } else {
          // 普通socket: 直接关闭
          close(fd);
        }

        socket_infos.erase(it);
        removeFromEpoll(epoll_fd, fd);
        return;
      }
      return;
    }

    // 根据socket类型处理接收到的数据
    if (it->second.type == SocketType::QP_SOCKET && !it->second.qp.expired()) {
      // QP socket: 交给QP处理
      auto qp = it->second.qp.lock();
      qp->handleReceivedData(buffer, bytes_read);
    } else {
      // 普通socket: 可以根据需要处理，例如协议识别
      LOG_INFO << "Received " << bytes_read << " bytes from non-QP socket: " << fd;
      // 这里可以添加对普通socket数据的处理逻辑
    }
  }
}

// 重置socket使用计数
void SockCtx::resetUsageCounts() {
  std::lock_guard<std::mutex> lock(qps_mutex);

  for (auto& pair : socket_infos) {
    pair.second.resetUsage();
  }

  LOG_INFO << "Reset usage counts for all sockets";
}

// 关闭指定socket连接
void SockCtx::closeConnection(int fd) {
  std::lock_guard<std::mutex> lock(qps_mutex);

  auto it = socket_infos.find(fd);
  if (it == socket_infos.end()) {
    return;
  }

  // 根据socket类型处理
  if (it->second.type == SocketType::QP_SOCKET && !it->second.qp.expired()) {
    // QP socket: 断开连接但不删除QP对象
    auto qp = it->second.qp.lock();
    qp->disconnect();
    int qp_id = qp->getInfo().qpn;
    qp_id_to_fd.erase(qp_id);
    LOG_INFO << "Closed QP connection for QP ID: " << qp_id << ", fd: " << fd;
  } else {
    // 普通socket: 直接关闭
    close(fd);
    LOG_INFO << "Closed non-QP connection, fd: " << fd;
  }

  socket_infos.erase(it);
  removeFromEpoll(epoll_fd, fd);
}

void SockCtx::processEvents() {
  bindToCpu();
  const int MAX_EVENTS = 64;
  struct epoll_event events[MAX_EVENTS];

  auto last_reset_time = std::chrono::steady_clock::now();
  auto last_manage_time = std::chrono::steady_clock::now();

  while (running) {
    int num_events = epoll_wait(epoll_fd, events, MAX_EVENTS, 0);

    if (num_events < 0) {
      if (errno == EINTR) {
        continue;
      }
      LOG_ERROR << "epoll_wait error: " << strerror(errno);
      break;
    }

    for (int i = 0; i < num_events; i++) {
      int fd = events[i].data.fd;
      uint32_t event_flags = events[i].events;
      if (fd == listen_socket) {
        acceptConnections();
      } else {
        handleSocketEvent(fd, event_flags);
      }
    }

    auto now = std::chrono::steady_clock::now();
    if (std::chrono::duration_cast<std::chrono::hours>(now - last_reset_time).count() >= 1) {
      resetUsageCounts();
      last_reset_time = now;
    }

    if (std::chrono::duration_cast<std::chrono::seconds>(now - last_manage_time).count() >= 5) {
      manageConnections();
      last_manage_time = now;
    }
  }
}

void SockCtx::acceptConnections() {
  struct sockaddr_in client_addr;
  socklen_t client_len = sizeof(client_addr);

  while (true) {
    int client_socket = accept(listen_socket, (struct sockaddr*)&client_addr, &client_len);
    if (client_socket < 0) {
      if (errno == EAGAIN || errno == EWOULDBLOCK) {
        break;
      }
      LOG_ERROR << "Failed to accept connection: " << strerror(errno);
      continue;
    }

    int flag = 1;
    if (setsockopt(client_socket, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(int)) < 0) {
      close(client_socket);
      LOG_ERROR << "Failed to set TCP_NODELAY: " << strerror(errno);
      continue;
    }

    if (setNonBlocking(client_socket) < 0) {
      close(client_socket);
      LOG_ERROR << "Failed to set non-blocking mode: " << strerror(errno);
      continue;
    }

    if (addToEpoll(epoll_fd, client_socket, EPOLLIN) < 0) {
      close(client_socket);
      LOG_ERROR << "Failed to add socket to epoll: " << strerror(errno);
      continue;
    }

    char client_ip[INET_ADDRSTRLEN];
    inet_ntop(AF_INET, &(client_addr.sin_addr), client_ip, INET_ADDRSTRLEN);
    int client_port = ntohs(client_addr.sin_port);

    try {
      auto qp = std::make_shared<SockQp>(this, client_socket, client_ip, client_port, 64, 16);

      std::lock_guard<std::mutex> lock(qps_mutex);
      qps.push_back(qp);
      socket_infos.insert_or_assign(client_socket,
                                    std::move(SocketInfo(SocketType::QP_SOCKET, qp)));
      qp_id_to_fd[qp->getInfo().qpn] = client_socket;

      LOG_INFO << "Accepted connection from " << client_ip << ":" << client_port
               << ", socket: " << client_socket << ", QP ID: " << qp->getInfo().qpn;
    } catch (const std::exception& e) {
      LOG_ERROR << "Failed to create QP for accepted connection: " << e.what();

      std::lock_guard<std::mutex> lock(qps_mutex);
      socket_infos.insert_or_assign(client_socket, std::move(SocketInfo()));

      LOG_INFO << "Accepted connection (non-QP) from " << client_ip << ":" << client_port
               << ", socket: " << client_socket;
    }
  }
}

SockStatus SockQp::handleReceivedData(const char* data, size_t length) {
  if (recv_bytes + length > recv_buffer.size()) {
    recv_buffer.resize(recv_bytes + length);
  }
  memcpy(recv_buffer.data() + recv_bytes, data, length);
  recv_bytes += length;

  size_t processed = 0;
  while (recv_bytes - processed >= sizeof(MessageHeader)) {
    const MessageHeader* header =
        reinterpret_cast<const MessageHeader*>(recv_buffer.data() + processed);
    size_t message_size = sizeof(MessageHeader);
    if (header->type == MessageType::RESPONSE || header->type == MessageType::REQUEST) {
      if (header->op_type == SockOpType::RDMA_WRITE ||
          header->op_type == SockOpType::RDMA_WRITE_WITH_IMM) {
        message_size += header->size;
      } else if (header->op_type == SockOpType::RDMA_READ &&
                 header->type == MessageType::RESPONSE) {
        message_size += header->size;
      } else if (header->op_type == SockOpType::RDMA_ATOMIC_ADD &&
                 header->type == MessageType::RESPONSE) {
        message_size += sizeof(uint64_t);
      }
    }

    if (recv_bytes - processed < message_size) {
      break;
    }

    SockStatus status = handleMessageHeader(recv_buffer.data() + processed);
    if (status != SockStatus::SUCCESS) {
      LOG_ERROR << "Error handling message";
    }

    processed += message_size;
  }

  if (processed > 0) {
    if (processed < recv_bytes) {
      memmove(recv_buffer.data(), recv_buffer.data() + processed, recv_bytes - processed);
    }
    recv_bytes -= processed;
  }

  return SockStatus::SUCCESS;
}

SockStatus SockQp::handleMessageHeader(const void* header_data) {
  const MessageHeader* header = static_cast<const MessageHeader*>(header_data);
  char* data = const_cast<char*>(static_cast<const char*>(header_data)) + sizeof(MessageHeader);

  // 更新使用计数
  ctx->updateSocketUsage(socket_fd);

  // 对于响应和完成通知，需要路由到正确的QP
  if (header->type == MessageType::RESPONSE || header->type == MessageType::COMPLETION ||
      header->type == MessageType::ERROR) {
    if (header->conn_id != info.qpn) {
      // 消息不是给当前QP的，需要路由
      auto target_qp = ctx->findQpById(header->conn_id);
      if (target_qp) {
        // 转发给正确的QP
        return target_qp->handleMessageHeader(header_data);
      } else {
        LOG_ERROR << "Cannot find QP with ID: " << header->conn_id;
        return SockStatus::ERROR_GENERAL;
      }
    }
  }

  switch (header->type) {
    case MessageType::REQUEST: {
      switch (header->op_type) {
        case SockOpType::RDMA_READ: {
          // 找到请求的内存区域
          SockMr* mr = nullptr;
          for (const auto& pair : mr_map) {
            if (pair.first == header->mr_id) {
              mr = pair.second;
              break;
            }
          }

          if (!mr) {
            MessageHeader err_header = *header;
            err_header.type = MessageType::ERROR;
            err_header.status = SockStatus::ERROR_MR_NOT_FOUND;

            if (ctx->sendData(socket_fd, &err_header, sizeof(MessageHeader)) !=
                SockStatus::SUCCESS) {
              LOG_ERROR << "Failed to send error response";
            }

            if (header->signaled) {
              MessageHeader completion = *header;
              completion.type = MessageType::COMPLETION;
              completion.status = SockStatus::ERROR_MR_NOT_FOUND;

              if (ctx->sendData(socket_fd, &completion, sizeof(MessageHeader)) !=
                  SockStatus::SUCCESS) {
                LOG_ERROR << "Failed to send completion notification";
              }
            }
            return SockStatus::ERROR_MR_NOT_FOUND;
          }

          char* src = static_cast<char*>(mr->getBuff()) +
                      (header->dst_addr - reinterpret_cast<uint64_t>(mr->getBuff()));

          MessageHeader response_header = *header;
          response_header.type = MessageType::RESPONSE;

          std::vector<char> response(sizeof(MessageHeader) + header->size);
          memcpy(response.data(), &response_header, sizeof(MessageHeader));
          memcpy(response.data() + sizeof(MessageHeader), src, header->size);

          if (ctx->sendData(socket_fd, response.data(), response.size()) != SockStatus::SUCCESS) {
            LOG_ERROR << "Failed to send RDMA_READ response";

            if (header->signaled) {
              MessageHeader completion = *header;
              completion.type = MessageType::COMPLETION;
              completion.status = SockStatus::ERROR_SEND_FAILED;

              if (ctx->sendData(socket_fd, &completion, sizeof(MessageHeader)) !=
                  SockStatus::SUCCESS) {
                LOG_ERROR << "Failed to send completion notification";
              }
            }
            return SockStatus::ERROR_SEND_FAILED;
          }

          if (header->signaled) {
            MessageHeader completion = *header;
            completion.type = MessageType::COMPLETION;
            completion.status = SockStatus::SUCCESS;

            if (ctx->sendData(socket_fd, &completion, sizeof(MessageHeader)) !=
                SockStatus::SUCCESS) {
              LOG_ERROR << "Failed to send completion notification";
              return SockStatus::ERROR_SEND_FAILED;
            }
          }
          break;
        }

        case SockOpType::RDMA_WRITE:
        case SockOpType::RDMA_WRITE_WITH_IMM: {
          // 找到目标内存区域
          SockMr* mr = nullptr;
          for (const auto& pair : mr_map) {
            if (pair.first == header->mr_id) {
              mr = pair.second;
              break;
            }
          }

          if (!mr) {
            MessageHeader err_header = *header;
            err_header.type = MessageType::ERROR;
            err_header.status = SockStatus::ERROR_MR_NOT_FOUND;

            if (ctx->sendData(socket_fd, &err_header, sizeof(MessageHeader)) !=
                SockStatus::SUCCESS) {
              LOG_ERROR << "Failed to send error response";
            }

            if (header->signaled) {
              MessageHeader completion = *header;
              completion.type = MessageType::COMPLETION;
              completion.status = SockStatus::ERROR_MR_NOT_FOUND;

              if (ctx->sendData(socket_fd, &completion, sizeof(MessageHeader)) !=
                  SockStatus::SUCCESS) {
                LOG_ERROR << "Failed to send completion notification";
              }
            }
            return SockStatus::ERROR_MR_NOT_FOUND;
          }

          // 计算目标地址并复制数据
          char* dst = static_cast<char*>(mr->getBuff()) +
                      (header->dst_addr - reinterpret_cast<uint64_t>(mr->getBuff()));
          memcpy(dst, data, header->size);

          // 如果需要信号通知，发送完成通知
          if (header->signaled) {
            MessageHeader completion = *header;
            completion.type = MessageType::COMPLETION;
            completion.status = SockStatus::SUCCESS;

            if (ctx->sendData(socket_fd, &completion, sizeof(MessageHeader)) !=
                SockStatus::SUCCESS) {
              LOG_ERROR << "Failed to send completion notification";
              return SockStatus::ERROR_SEND_FAILED;
            }
          }
          break;
        }

        case SockOpType::RDMA_ATOMIC_ADD: {
          // 找到目标内存区域
          SockMr* mr = nullptr;
          for (const auto& pair : mr_map) {
            if (pair.first == header->mr_id) {
              mr = pair.second;
              break;
            }
          }

          if (!mr) {
            // 错误：未找到MR，发送错误响应
            MessageHeader err_header = *header;
            err_header.type = MessageType::ERROR;
            err_header.status = SockStatus::ERROR_MR_NOT_FOUND;

            if (ctx->sendData(socket_fd, &err_header, sizeof(MessageHeader)) !=
                SockStatus::SUCCESS) {
              LOG_ERROR << "Failed to send error response";
            }

            // 如果需要信号通知，发送带错误的完成通知
            if (header->signaled) {
              MessageHeader completion = *header;
              completion.type = MessageType::COMPLETION;
              completion.status = SockStatus::ERROR_MR_NOT_FOUND;

              if (ctx->sendData(socket_fd, &completion, sizeof(MessageHeader)) !=
                  SockStatus::SUCCESS) {
                LOG_ERROR << "Failed to send completion notification";
              }
            }
            return SockStatus::ERROR_MR_NOT_FOUND;
          }

          // 计算目标地址
          uint64_t* dst = reinterpret_cast<uint64_t*>(
              static_cast<char*>(mr->getBuff()) +
              (header->dst_addr - reinterpret_cast<uint64_t>(mr->getBuff())));

          // 执行原子加操作并获取旧值
          uint64_t old_value = __sync_fetch_and_add(dst, header->atomic_val);

          // 准备响应消息
          MessageHeader response_header = *header;
          response_header.type = MessageType::RESPONSE;
          response_header.status = SockStatus::SUCCESS;

          // 创建响应缓冲区
          std::vector<char> response(sizeof(MessageHeader) + sizeof(uint64_t));
          memcpy(response.data(), &response_header, sizeof(MessageHeader));
          memcpy(response.data() + sizeof(MessageHeader), &old_value, sizeof(uint64_t));

          // 发送响应
          if (ctx->sendData(socket_fd, response.data(), response.size()) != SockStatus::SUCCESS) {
            LOG_ERROR << "Failed to send ATOMIC_ADD response";

            // 如果需要信号通知，发送带错误的完成通知
            if (header->signaled) {
              MessageHeader completion = *header;
              completion.type = MessageType::COMPLETION;
              completion.status = SockStatus::ERROR_SEND_FAILED;

              if (ctx->sendData(socket_fd, &completion, sizeof(MessageHeader)) !=
                  SockStatus::SUCCESS) {
                LOG_ERROR << "Failed to send completion notification";
              }
            }
            return SockStatus::ERROR_SEND_FAILED;
          }

          // 如果需要信号通知，发送完成通知
          if (header->signaled) {
            MessageHeader completion = *header;
            completion.type = MessageType::COMPLETION;
            completion.status = SockStatus::SUCCESS;

            if (ctx->sendData(socket_fd, &completion, sizeof(MessageHeader)) !=
                SockStatus::SUCCESS) {
              LOG_ERROR << "Failed to send completion notification";
              return SockStatus::ERROR_SEND_FAILED;
            }
          }
          break;
        }

        default:
          LOG_ERROR << "Unknown operation type";
          return SockStatus::ERROR_INVALID_PARAM;
      }
      break;
    }

    case MessageType::ERROR: {
      // 处理错误响应
      std::lock_guard<std::mutex> lock(mutex);
      int idx = num_completed_items++;
      if (idx < wcs.size()) {
        wcs[idx].wr_id = header->wr_id;
        wcs[idx].status = header->status;
        wcs[idx].byte_len = 0;
        wcs[idx].imm_data = 0;
      }

      // 减少已发布的信号项
      if (num_signaled_posted_items > 0) {
        num_signaled_posted_items--;
      }

      // 通知等待线程
      cv.notify_all();
      break;
    }

    case MessageType::RESPONSE: {
      // 处理响应
      switch (header->op_type) {
        case SockOpType::RDMA_READ: {
          // 查找本地内存区域
          const SockMr* mr = nullptr;
          for (const auto& wr : pending_wrs) {
            if (wr.wr_id == header->wr_id) {
              mr = wr.local_mr;
              break;
            }
          }

          if (!mr) {
            LOG_ERROR << "Warning: received response for unknown WR ID: " << header->wr_id;
            break;
          }

          char* dst = static_cast<char*>(const_cast<void*>(mr->getBuff())) +
                      (header->src_addr - reinterpret_cast<uint64_t>(mr->getBuff()));
          memcpy(dst, data, header->size);
          break;
        }

        case SockOpType::RDMA_ATOMIC_ADD: {
          // 查找本地内存区域
          const SockMr* mr = nullptr;
          for (const auto& wr : pending_wrs) {
            if (wr.wr_id == header->wr_id) {
              mr = wr.local_mr;
              break;
            }
          }

          if (!mr) {
            LOG_ERROR << "Warning: received response for unknown WR ID: " << header->wr_id;
            break;
          }

          // 存储旧值
          uint64_t old_value;
          memcpy(&old_value, data, sizeof(uint64_t));
          uint64_t* dst = reinterpret_cast<uint64_t*>(const_cast<void*>(mr->getBuff()));
          *dst = old_value;
          break;
        }

        default:
          break;
      }
      break;
    }

    case MessageType::COMPLETION: {
      std::lock_guard<std::mutex> lock(mutex);
      int idx = num_completed_items++;
      if (idx < wcs.size()) {
        wcs[idx].wr_id = header->wr_id;
        wcs[idx].status = header->status;
        wcs[idx].byte_len = header->size;
        wcs[idx].imm_data = header->imm_data;
      }

      if (num_signaled_posted_items > 0) {
        num_signaled_posted_items--;
      }

      // 通知等待线程
      cv.notify_all();
      break;
    }

    case MessageType::CONNECT:
    case MessageType::CONNECT_ACK:
      // 这些在connect方法中处理
      break;

    default:
      LOG_ERROR << "Unknown message type";
      return SockStatus::ERROR_INVALID_PARAM;
  }

  return SockStatus::SUCCESS;
}

SockStatus SockQp::connect(const SockQpInfo& remote_info) {
  MessageHeader header;
  header.type = MessageType::CONNECT;
  header.status = SockStatus::SUCCESS;

  std::vector<char> buffer(sizeof(MessageHeader));
  memcpy(buffer.data(), &header, sizeof(MessageHeader));

  std::string info_str =
      info.host + ":" + std::to_string(info.port) + ":" + std::to_string(info.qpn);
  buffer.resize(sizeof(MessageHeader) + info_str.size());
  memcpy(buffer.data() + sizeof(MessageHeader), info_str.c_str(), info_str.size());

  if (ctx->sendData(socket_fd, buffer.data(), buffer.size()) != SockStatus::SUCCESS) {
    LOG_ERROR << "Failed to send connection information";
    return SockStatus::ERROR_CONN_FAILED;
  }

  buffer.resize(sizeof(MessageHeader));
  int retry = 0;
  while (retry < 5) {
    fd_set readfds;
    FD_ZERO(&readfds);
    FD_SET(socket_fd, &readfds);

    struct timeval tv = {1, 0};  // 1秒超时
    int result = select(socket_fd + 1, &readfds, nullptr, nullptr, &tv);

    if (result > 0) {
      ssize_t received = recv(socket_fd, buffer.data(), sizeof(MessageHeader), 0);
      if (received == sizeof(MessageHeader)) {
        break;
      }
    }

    retry++;
  }

  if (retry >= 5) {
    LOG_ERROR << "Timeout waiting for connection acknowledgment";
    return SockStatus::ERROR_TIMEOUT;
  }

  memcpy(&header, buffer.data(), sizeof(MessageHeader));
  if (header.type != MessageType::CONNECT_ACK) {
    LOG_ERROR << "Invalid connection acknowledgment";
    return SockStatus::ERROR_CONN_FAILED;
  }

  connected = true;
  return SockStatus::SUCCESS;
}

SockStatus SockQp::stageLoad(const SockMr* mr, const SockMrInfo& info, size_t size, uint64_t wrId,
                             uint64_t srcOffset, uint64_t dstOffset, bool signaled) {
  if (!mr) {
    return SockStatus::ERROR_INVALID_PARAM;
  }

  if (pending_wrs.size() >= max_wr) {
    return SockStatus::ERROR_WR_OVERFLOW;
  }

  WrInfo wr;
  wr.opcode = SockOpType::RDMA_READ;
  wr.wr_id = wrId;
  wr.signaled = signaled;
  wr.local_mr = mr;
  wr.local_offset = dstOffset;
  wr.remote_mr = info;
  wr.remote_offset = srcOffset;
  wr.size = size;

  pending_wrs.push_back(wr);
  if (signaled) {
    num_signaled_staged_items++;
  }
  return SockStatus::SUCCESS;
}

SockStatus SockQp::stageSend(const SockMr* mr, const SockMrInfo& info, uint32_t size, uint64_t wrId,
                             uint64_t srcOffset, uint64_t dstOffset, bool signaled) {
  if (!mr) {
    return SockStatus::ERROR_INVALID_PARAM;
  }

  if (pending_wrs.size() >= max_wr) {
    return SockStatus::ERROR_WR_OVERFLOW;
  }

  WrInfo wr;
  wr.opcode = SockOpType::RDMA_WRITE;
  wr.wr_id = wrId;
  wr.signaled = signaled;
  wr.local_mr = mr;
  wr.local_offset = srcOffset;
  wr.remote_mr = info;
  wr.remote_offset = dstOffset;
  wr.size = size;

  pending_wrs.push_back(wr);
  if (signaled) {
    num_signaled_staged_items++;
  }
  return SockStatus::SUCCESS;
}

SockStatus SockQp::stageAtomicAdd(const SockMr* mr, const SockMrInfo& info, uint64_t wrId,
                                  uint64_t dstOffset, uint64_t addVal, bool signaled) {
  if (!mr) {
    return SockStatus::ERROR_INVALID_PARAM;
  }

  if (pending_wrs.size() >= max_wr) {
    return SockStatus::ERROR_WR_OVERFLOW;
  }

  WrInfo wr;
  wr.opcode = SockOpType::RDMA_ATOMIC_ADD;
  wr.wr_id = wrId;
  wr.signaled = signaled;
  wr.local_mr = mr;
  wr.local_offset = 0;  // 结果将存储在缓冲区开头
  wr.remote_mr = info;
  wr.remote_offset = dstOffset;
  wr.size = sizeof(uint64_t);  // 原子操作总是作用于uint64_t
  wr.add_val = addVal;

  pending_wrs.push_back(wr);
  if (signaled) {
    num_signaled_staged_items++;
  }
  return SockStatus::SUCCESS;
}

SockStatus SockQp::stageSendWithImm(const SockMr* mr, const SockMrInfo& info, uint32_t size,
                                    uint64_t wrId, uint64_t srcOffset, uint64_t dstOffset,
                                    bool signaled, unsigned int immData) {
  if (!mr) {
    return SockStatus::ERROR_INVALID_PARAM;
  }

  if (pending_wrs.size() >= max_wr) {
    return SockStatus::ERROR_WR_OVERFLOW;
  }

  WrInfo wr;
  wr.opcode = SockOpType::RDMA_WRITE_WITH_IMM;
  wr.wr_id = wrId;
  wr.signaled = signaled;
  wr.local_mr = mr;
  wr.local_offset = srcOffset;
  wr.remote_mr = info;
  wr.remote_offset = dstOffset;
  wr.size = size;
  wr.imm_data = immData;

  pending_wrs.push_back(wr);
  if (signaled) {
    num_signaled_staged_items++;
  }
  return SockStatus::SUCCESS;
}

SockStatus SockQp::sendRequest(const WrInfo& wr) {
  // 确保连接存在
  SockStatus status = ensureConnected();
  if (status != SockStatus::SUCCESS) {
    return status;
  }

  // 更新socket使用计数
  ctx->updateSocketUsage(socket_fd);

  MessageHeader header;
  header.type = MessageType::REQUEST;
  header.op_type = wr.opcode;
  header.wr_id = wr.wr_id;
  header.src_addr = reinterpret_cast<uint64_t>(wr.local_mr->getBuff()) + wr.local_offset;
  header.dst_addr = wr.remote_mr.addr + wr.remote_offset;
  header.size = wr.size;
  header.mr_id = wr.remote_mr.id;
  header.atomic_val = wr.add_val;
  header.imm_data = wr.imm_data;
  header.signaled = wr.signaled;
  header.status = SockStatus::SUCCESS;
  header.conn_id = this->info.qpn;  // 设置发送方的QP ID

  std::vector<char> buffer(sizeof(MessageHeader));
  memcpy(buffer.data(), &header, sizeof(MessageHeader));

  // 对于WRITE操作，附加要发送的数据
  if (wr.opcode == SockOpType::RDMA_WRITE || wr.opcode == SockOpType::RDMA_WRITE_WITH_IMM) {
    const char* data = static_cast<const char*>(wr.local_mr->getBuff()) + wr.local_offset;
    buffer.resize(sizeof(MessageHeader) + wr.size);
    memcpy(buffer.data() + sizeof(MessageHeader), data, wr.size);
  }

  return ctx->sendData(socket_fd, buffer.data(), buffer.size());
}

SockStatus SockQp::postSend() {
  if (pending_wrs.empty()) {
    return SockStatus::SUCCESS;
  }

  if (!connected) {
    return SockStatus::ERROR_QP_STATE;
  }

  std::lock_guard<std::mutex> lock(mutex);

  SockStatus overall_status = SockStatus::SUCCESS;

  for (const auto& wr : pending_wrs) {
    SockStatus status = sendRequest(wr);
    if (status != SockStatus::SUCCESS) {
      overall_status = status;
      // 记录错误但继续尝试发送其他请求
      LOG_ERROR << "Failed to send request, wr_id: " << wr.wr_id;
    }
  }

  num_signaled_posted_items += num_signaled_staged_items;
  num_signaled_staged_items = 0;
  pending_wrs.clear();

  return overall_status;
}

int SockQp::pollCq() {
  std::lock_guard<std::mutex> lock(mutex);

  int completed = num_completed_items.exchange(0);
  return completed;
}

SockStatus SockQp::getWcStatus(int idx) const {
  if (idx >= 0 && idx < wcs.size()) {
    return wcs[idx].status;
  }
  return SockStatus::ERROR_GENERAL;
}

int SockQp::getNumCqItems() const { return num_signaled_posted_items; }

// QP断开连接
void SockQp::disconnect() {
  std::lock_guard<std::mutex> lock(mutex);
  if (socket_fd >= 0) {
    close(socket_fd);
    socket_fd = -1;
  }
  connected = false;
  // 保留pending_wrs，以便重连后继续
}

// QP确保连接
SockStatus SockQp::ensureConnected() {
  if (connected) {
    return SockStatus::SUCCESS;
  }

  std::lock_guard<std::mutex> lock(mutex);
  if (connected) {  // 再次检查，避免竞态条件
    return SockStatus::SUCCESS;
  }

  // 创建新的socket
  socket_fd = socket(AF_INET, SOCK_STREAM, 0);
  if (socket_fd < 0) {
    LOG_ERROR << "Failed to create socket: " << strerror(errno);
    return SockStatus::ERROR_CONN_FAILED;
  }

  // 设置socket选项
  int flag = 1;
  if (setsockopt(socket_fd, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(int)) < 0) {
    close(socket_fd);
    socket_fd = -1;
    LOG_ERROR << "Failed to set TCP_NODELAY: " << strerror(errno);
    return SockStatus::ERROR_CONN_FAILED;
  }

  if (setNonBlocking(socket_fd) < 0) {
    close(socket_fd);
    socket_fd = -1;
    LOG_ERROR << "Failed to set non-blocking mode: " << strerror(errno);
    return SockStatus::ERROR_CONN_FAILED;
  }

  // 连接到远程主机
  struct sockaddr_in addr;
  memset(&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET;
  addr.sin_port = htons(info.port);

  if (inet_pton(AF_INET, info.host.c_str(), &addr.sin_addr) <= 0) {
    close(socket_fd);
    socket_fd = -1;
    LOG_ERROR << "Invalid address: " << info.host;
    return SockStatus::ERROR_CONN_FAILED;
  }

  if (::connect(socket_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
    if (errno != EINPROGRESS) {
      close(socket_fd);
      socket_fd = -1;
      LOG_ERROR << "Failed to connect: " << strerror(errno);
      return SockStatus::ERROR_CONN_FAILED;
    }
  }

  // 将socket添加到epoll
  SockStatus status = ctx->addSocketToEpoll(socket_fd);
  if (status != SockStatus::SUCCESS) {
    close(socket_fd);
    socket_fd = -1;
    return SockStatus::ERROR_CONN_FAILED;
  }

  // 更新ctx的映射
  ctx->updateSocketMapping(socket_fd, this);

  connected = true;
  return SockStatus::SUCCESS;
}

// 管理连接数量
void SockCtx::manageConnections() {
  std::lock_guard<std::mutex> lock(qps_mutex);

  // 如果连接数未超过限制，则不需处理
  if (socket_infos.size() <= static_cast<size_t>(max_connections_)) {
    return;
  }

  // 收集所有socket信息用于LFU分析
  std::vector<std::pair<int, SocketInfo>> all_sockets;
  for (const auto& entry : socket_infos) {
    all_sockets.push_back(
        std::make_pair(entry.first, std::move(SocketInfo(entry.second.type, entry.second.qp))));
  }

  std::sort(all_sockets.begin(), all_sockets.end(), SocketLfuComparator());

  // 优先关闭ACCEPT_SOCKET类型的连接
  int to_close = socket_infos.size() - max_connections_;

  // 第一遍：关闭非QP的socket
  for (const auto& entry : all_sockets) {
    if (to_close <= 0) break;

    if (entry.second.type == SocketType::ACCEPT_SOCKET) {
      closeConnection(entry.first);
      to_close--;
    }
  }

  // 如果仍需关闭连接，关闭最不常用的QP连接
  if (to_close > 0) {
    for (const auto& entry : all_sockets) {
      if (to_close <= 0) break;

      if (entry.second.type == SocketType::QP_SOCKET && !entry.second.qp.expired()) {
        auto qp = entry.second.qp.lock();
        LOG_INFO << "Closing least frequently used QP connection, fd: " << entry.first
                 << ", usage count: " << entry.second.getUsageCount();

        // 关闭QP连接但不删除QP对象
        qp->disconnect();
        removeFromEpoll(epoll_fd, entry.first);
        socket_infos.erase(entry.first);
        qp_id_to_fd.erase(qp->getInfo().qpn);

        to_close--;
      }
    }
  }
}

}  // namespace pccl
