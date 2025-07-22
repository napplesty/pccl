#include "net/tcp/tcp.h"
#include <netinet/in.h>
#include <unistd.h>
#include <cstring>
#include <fcntl.h>
#include <sys/epoll.h>
#include <arpa/inet.h>
#include <stdexcept>
#include <iostream>

namespace pccl {

// ==================== 全局静态变量初始化 ====================
constexpr static int qpn_maximum = (1u << 20) + 1;
std::atomic<int> TcpCtx::next_mrn(0);
std::atomic<int> TcpCtx::next_qpn(0);

// ==================== TcpQp 成员函数实现 ====================
TcpQp::~TcpQp() {
  if (auto ctx = ctx_.lock()) {
    ctx->removeQp(local_handle.qpn);
  }
}

void TcpQp::rtr() {
  // TODO, setup a socket fd in ctx to recv
  rtr_ = true;
}

void TcpQp::rts() {
  // TODO, setup a socket fd in ctx to send
  rts_ = true;
}

TcpWr& TcpQp::stageOperation() {
  staged_operations->emplace_back();
  return staged_operations->back();
}

void TcpQp::stageLoad(const TcpMr &local_mr, const TcpMr &remote_mr, size_t size,
                      OperatorId op_id, OperationId operation_id,
                      size_t src_offset, size_t dst_offset) {
  auto& wr = stageOperation();
  wr.op_type = TcpOpType::READ;
  wr.operator_id = op_id;
  wr.operation_id = operation_id;
  wr.local_mr_info = local_mr;
  wr.remote_mr_info = remote_mr;
  wr.size = size;
  wr.local_offset = src_offset;
  wr.remote_offset = dst_offset;
}

void TcpQp::stageSend(const TcpMr &local_mr, const TcpMr &remote_mr, size_t size,
                      OperatorId op_id, OperationId operation_id,
                      size_t srcOffset, size_t dstOffset) {
  auto& wr = stageOperation();
  wr.op_type = TcpOpType::WRITE;
  wr.operator_id = op_id;
  wr.operation_id = operation_id;
  wr.local_mr_info = local_mr;
  wr.remote_mr_info = remote_mr;
  wr.size = size;
  wr.local_offset = srcOffset;
  wr.remote_offset = dstOffset;
}

void TcpQp::stageAtomicAdd(const TcpMr &local_mr, const TcpMr &remote_mr,
                           OperatorId op_id, OperationId operation_id,
                           size_t dst_offset, int add_val) {
  auto& wr = stageOperation();
  wr.op_type = TcpOpType::ATOMIC_ADD;
  wr.operator_id = op_id;
  wr.operation_id = operation_id;
  wr.local_mr_info = local_mr;
  wr.remote_mr_info = remote_mr;
  wr.remote_offset = dst_offset;
  wr.atomic_value = add_val;
}

TcpStatus TcpQp::postOperations() {
  if (!rts_ || !rtr_) return TcpStatus::ERROR_QP_STATE;
  if (auto ctx = ctx_.lock()) {
    ctx->push_to_global_send_queue(staged_operations);
    staged_operations = std::make_shared<std::deque<TcpWr>>();
    return TcpStatus::SUCCESS;
  }
  return TcpStatus::ERROR;
}

int TcpQp::pollCq() {
  std::lock_guard<std::mutex> lock(qp_mutex);
  return pulled_wcs->size();
}

TcpWc& TcpQp::getWcStatus(int idx) {
  std::lock_guard<std::mutex> lock(qp_mutex);
  if (idx < 0 || static_cast<size_t>(idx) >= pulled_wcs->size()) {
      throw std::out_of_range("Invalid CQ index");
  }
  return (*pulled_wcs)[idx];
}

// ==================== TcpCtx 成员函数实现 ====================
TcpCtx::TcpCtx(TcpAddress addr) : addr(addr), running(true) {
  listen_socket = socket(addr.family, SOCK_STREAM, IPPROTO_TCP);
  if (listen_socket < 0) {
    throw std::runtime_error("Failed to create socket");
  }

  int opt = 1;
  setsockopt(listen_socket, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

  sockaddr_storage sa;
  memset(&sa, 0, sizeof(sa));
  if (addr.family == AF_INET) {
    auto* sa_in = reinterpret_cast<sockaddr_in*>(&sa);
    sa_in->sin_family = AF_INET;
    sa_in->sin_port = htons(addr.port);
    sa_in->sin_addr = addr.v4_ip;
    if (bind(listen_socket, reinterpret_cast<sockaddr*>(sa_in), sizeof(*sa_in)) < 0) {
        throw std::runtime_error("Bind failed");
    }
  } else {
    auto* sa_in6 = reinterpret_cast<sockaddr_in6*>(&sa);
    sa_in6->sin6_family = AF_INET6;
    sa_in6->sin6_port = htons(addr.port);
    sa_in6->sin6_addr = addr.v6_ip;
    if (bind(listen_socket, reinterpret_cast<sockaddr*>(sa_in6), sizeof(*sa_in6)) < 0) {
      throw std::runtime_error("Bind failed");
    }
  }

  int status = listen(listen_socket, 32);
  if(status == -1) {
    throw std::runtime_error("Bind failed");
  }

  recv_epoll_fd = epoll_create1(0);
  send_epoll_fd = epoll_create1(0);
  if (recv_epoll_fd < 0 || send_epoll_fd < 0) {
    throw std::runtime_error("Failed to create epoll");
  }

  recv_thread = std::thread(&TcpCtx::recvThreadCycle, this);
  send_thread = std::thread(&TcpCtx::sendThreadCycle, this);
}

TcpCtx::~TcpCtx() {
    running = false;
    
    // 关闭所有socket
    close(listen_socket);
    for (auto& pair : qpn_to_send_fd) close(pair.second);
    for (auto& pair : qpn_to_recv_fd) close(pair.second);
    
    // 等待线程退出
    if (recv_thread.joinable()) recv_thread.join();
    if (send_thread.joinable()) send_thread.join();
    
    close(recv_epoll_fd);
    close(send_epoll_fd);
}

std::shared_ptr<TcpQp> TcpCtx::createQp(int max_cq_size, int max_wr) {
    std::lock_guard<std::mutex> lock(ctx_mutex);
    int qpn = next_qp_id++;
    
    auto qp = std::make_shared<TcpQp>();
    qp->ctx_ = shared_from_this();
    qp->maxCqSize = max_cq_size;
    qp->maxWrqSize = max_wr;
    qp->pulled_wcs = std::make_shared<std::deque<TcpWc>>();
    qp->staged_operations = std::make_shared<std::deque<TcpWr>>();
    qp->local_handle.qpn = qpn;
    qp->local_handle.addr = addr;
    
    qps[qpn] = qp;
    return qp;
}

std::shared_ptr<TcpMr> TcpCtx::registerMr(RegisteredMemory memory) {
    std::lock_guard<std::mutex> lock(ctx_mutex);
    int mr_id = next_mr_id++;
    
    auto mr = std::make_shared<TcpMr>();
    mr->addr = reinterpret_cast<uintptr_t>(memory.addr);
    mr->component_flag = memory.flags;
    mr->mr_id = mr_id;
    
    mrs[mr_id] = mr;
    return mr;
}

std::shared_ptr<TcpQp> TcpCtx::findQpByQpn(int qpn) {
    std::lock_guard<std::mutex> lock(ctx_mutex);
    auto it = qps.find(qpn);
    if (it != qps.end()) return it->second;
    return nullptr;
}

void TcpCtx::removeQp(int64_t qpn) {
    std::lock_guard<std::mutex> lock(ctx_mutex);
    auto qp_it = qps.find(qpn);
    if (qp_it != qps.end()) {
        auto fd_it_send = qpn_to_send_fd.find(qpn);
        if (fd_it_send != qpn_to_send_fd.end()) {
            close(fd_it_send->second);
            qpn_to_send_fd.erase(fd_it_send);
        }
        auto fd_it_recv = qpn_to_recv_fd.find(qpn);
        if (fd_it_recv != qpn_to_recv_fd.end()) {
            close(fd_it_recv->second);
            qpn_to_recv_fd.erase(fd_it_recv);
        }
        qps.erase(qp_it);
    }
}

void TcpCtx::push_to_global_send_queue(std::shared_ptr<std::deque<TcpWr>> wr_queue) {
    std::lock_guard<std::mutex> lock(send_queue_mutex);
    global_send_queue.push_back(wr_queue);
}

std::shared_ptr<std::deque<TcpWr>> TcpCtx::pop_from_global_send_queue() {
    std::lock_guard<std::mutex> lock(send_queue_mutex);
    if (global_send_queue.empty()) return nullptr;
    
    auto q = global_send_queue.front();
    global_send_queue.pop_front();
    return q;
}

void TcpCtx::sendThreadCycle() {
    while (running) {
        auto work_queue = pop_from_global_send_queue();
        if (!work_queue) {
            usleep(100); // 短暂休眠避免忙等待
            continue;
        }

        for (auto& wr : *work_queue) {
            TcpMessageHeader header;
            memset(&header, 0, sizeof(header));
            header.op_type = wr.op_type;
            header.src_qpn = wr.local_qpn;
            header.dst_qpn = wr.remote_qpn;
            
            switch (wr.op_type) {
                case TcpOpType::READ:
                case TcpOpType::WRITE: {
                    header.read_load.op_id = wr.operator_id;
                    header.read_load.operation_id = wr.operation_id;
                    header.read_load.dst_mr_info = wr.remote_mr_info;
                    header.read_load.src_mr_info = wr.local_mr_info;
                    header.read_load.dst_offset = wr.remote_offset;
                    header.read_load.src_offset = wr.local_offset;
                    header.read_load.size = wr.size;
                    break;
                }
                case TcpOpType::ATOMIC_ADD: {
                    header.atomic_add.op_id = wr.operator_id;
                    header.atomic_add.operation_id = wr.operation_id;
                    header.atomic_add.dst_mr_info = wr.remote_mr_info;
                    header.atomic_add.atomic_value = wr.atomic_value;
                    break;
                }
                default:
                    continue; // 跳过不支持的操作类型
            }

            // 查找发送fd
            int send_fd = -1;
            {
                std::lock_guard<std::mutex> lock(ctx_mutex);
                auto it = qpn_to_send_fd.find(wr.local_qpn);
                if (it != qpn_to_send_fd.end()) send_fd = it->second;
            }

            if (send_fd > 0) {
                size_t bytes_sent = 0;
                if (!handleSendMessageHeader(send_fd, header, bytes_sent)) {
                    // 发送失败处理
                    std::cerr << "Failed to send message" << std::endl;
                }
            }
        }
    }
}

void TcpCtx::recvThreadCycle() {
    const int MAX_EVENTS = 10;
    epoll_event events[MAX_EVENTS];
    
    // 将监听socket加入epoll
    epoll_event ev;
    ev.events = EPOLLIN;
    ev.data.fd = listen_socket;
    epoll_ctl(recv_epoll_fd, EPOLL_CTL_ADD, listen_socket, &ev);

    while (running) {
        int nfds = epoll_wait(recv_epoll_fd, events, MAX_EVENTS, 100);
        if (nfds < 0) {
            if (errno == EINTR) continue;
            break;
        }

        for (int i = 0; i < nfds; ++i) {
            if (events[i].data.fd == listen_socket) {
                // 接受新连接
                sockaddr_storage client_addr;
                socklen_t addrlen = sizeof(client_addr);
                int conn_fd = accept(listen_socket, 
                                   reinterpret_cast<sockaddr*>(&client_addr),
                                   &addrlen);
                if (conn_fd < 0) continue;
                
                // 设置为非阻塞
                int flags = fcntl(conn_fd, F_GETFL, 0);
                fcntl(conn_fd, F_SETFL, flags | O_NONBLOCK);

                // 接收握手消息
                TcpMessageHeader handshake;
                size_t bytes_recv = 0;
                if (!handleRecvMessageHeader(conn_fd, handshake, bytes_recv)) {
                    close(conn_fd);
                    continue;
                }

                // 处理握手
                if (!handleHandshake(conn_fd, handshake)) {
                    close(conn_fd);
                }
            } else {
                // 处理普通数据接收
                int conn_fd = events[i].data.fd;
                auto it = send_recv_contexts.find(conn_fd);
                if (it != send_recv_contexts.end()) {
                    handlePartialRecv(conn_fd, it->second);
                } else {
                    SendRecvContext context;
                    if (handleRecvMessageHeader(conn_fd, *context.header, context.bytes_recv)) {
                        send_recv_contexts[conn_fd] = context;
                    }
                }
            }
        }
    }
}

bool TcpCtx::handleHandshake(int fd, const TcpMessageHeader &header) {
    if (header.op_type != TcpOpType::HAND_SHAKE) return false;
    
    // 创建新QP
    int qpn = next_qp_id++;
    auto qp = std::make_shared<TcpQp>();
    qp->remote_handle = header.src_handle;
    qp->local_handle.qpn = qpn;
    qp->local_handle.addr = addr;
    
    // 更新QP映射
    {
        std::lock_guard<std::mutex> lock(ctx_mutex);
        qps[qpn] = qp;
        qpn_to_recv_fd[qpn] = fd;
        fd_to_qpn[fd] = qpn;
    }

    // 发送ACK
    TcpMessageHeader ack;
    ack.op_type = TcpOpType::HAND_SHAKE_ACK;
    ack.src_handle = qp->local_handle;
    size_t bytes_sent = 0;
    return handleSendMessageHeader(fd, ack, bytes_sent);
}

bool TcpCtx::handleHandshakeAck(int fd, int64_t local_qpn, int64_t remote_qpn) {
    std::lock_guard<std::mutex> lock(ctx_mutex);
    auto it = qps.find(local_qpn);
    if (it == qps.end()) return false;
    
    auto qp = it->second;
    qp->remote_handle.qpn = remote_qpn;
    qpn_to_send_fd[local_qpn] = fd;
    fd_to_qpn[fd] = local_qpn;
    return true;
}

bool TcpCtx::handleSendMessageHeader(int fd, const TcpMessageHeader &header, size_t &bytes_sent) {
    const char* data = reinterpret_cast<const char*>(&header);
    size_t total = sizeof(TcpMessageHeader);
    
    ssize_t n = send(fd, data + bytes_sent, total - bytes_sent, MSG_DONTWAIT);
    if (n < 0) {
        if (errno == EAGAIN || errno == EWOULDBLOCK) return true;
        return false;
    }
    
    bytes_sent += n;
    return bytes_sent == total;
}

bool TcpCtx::handleRecvMessageHeader(int fd, TcpMessageHeader &header, size_t &bytes_recv) {
    char* data = reinterpret_cast<char*>(&header);
    size_t total = sizeof(TcpMessageHeader);
    
    ssize_t n = recv(fd, data + bytes_recv, total - bytes_recv, MSG_DONTWAIT);
    if (n < 0) {
        if (errno == EAGAIN || errno == EWOULDBLOCK) return true;
        return false;
    }
    if (n == 0) return false; // 连接关闭
    
    bytes_recv += n;
    return bytes_recv == total;
}

bool TcpCtx::handlePartialSend(int fd, SendRecvContext &context) {
    // 简化的部分发送处理
    return true;
}

bool TcpCtx::handlePartialRecv(int fd, SendRecvContext &context) {
    // 简化的部分接收处理
    return true;
}

} // namespace pccl