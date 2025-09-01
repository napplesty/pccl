#pragma once

#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <string>
#include <functional>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <memory>
#include <map>
#include <atomic>
#include <iostream>
#include <list>
#include <poll.h>

namespace tcp_verbs {

enum Status {
  SUCCESS = 0,
  FAILURE = -1,
  TIMEOUT = -2,
  CONNECTION_CLOSED = -3
};

struct MemoryRegion {
  void* addr;
  size_t length;
  uint32_t lkey;
  uint32_t rkey;
};

struct EndpointConfig {
  std::string ip;
  int port;
  int max_msg_size = 4096;
  int timeout_ms = 5000;
};

enum CompletionType {
  SEND_COMPLETION,
  RECV_COMPLETION,
  CONNECTION_EVENT
};

struct CompletionEvent {
  CompletionType type;
  Status status;
  size_t byte_count;
  void* context;
  int endpoint_id;
};

using CompletionCallback = std::function<void(const CompletionEvent&)>;

class TcpManager;

class Endpoint {
public:
  Endpoint(TcpManager* manager, int id);
  ~Endpoint();

  Status init(const EndpointConfig& config);
  Status register_memory_region(void* addr, size_t length, MemoryRegion& mr);
  Status deregister_memory_region(MemoryRegion& mr);
  Status connect(const std::string& remote_ip, int remote_port);
  Status listen(int backlog = 10);
  Status accept();
  Status disconnect();
  Status post_send(const MemoryRegion& mr, size_t offset, size_t length, void* context = nullptr);
  Status post_recv(const MemoryRegion& mr, size_t offset, size_t length, void* context = nullptr);
  Status write(const MemoryRegion& local_mr, size_t local_offset, 
             const MemoryRegion& remote_mr, size_t remote_offset, size_t length, void* context = nullptr);
  Status read(const MemoryRegion& local_mr, size_t local_offset, 
            const MemoryRegion& remote_mr, size_t remote_offset, size_t length, void* context = nullptr);

  // 供TcpManager调用的处理函数
  void process_io();
  bool has_pending_io() const;

  int get_id() const { return id_; }
  int get_socket() const { return sockfd_; }
  bool is_connected() const { return connected_; }

private:
  friend class TcpManager;
  
  TcpManager* manager_;
  int id_;
  int sockfd_;
  int listenfd_;
  bool connected_;
  EndpointConfig config_;
  
  // 待处理的IO操作
  struct PendingIO {
    enum Type { SEND, RECV } type;
    MemoryRegion mr;
    size_t offset;
    size_t length;
    void* context;
  };
  
  std::queue<PendingIO> pending_io_;
  std::mutex io_mutex_;
};

class TcpManager {
public:
  TcpManager(int io_threads = 1);
  ~TcpManager();

  static TcpManager& instance() {
    static TcpManager instance;
    return instance;
  }

  Endpoint* create_endpoint();
  void destroy_endpoint(int endpoint_id);
  Endpoint* get_endpoint(int endpoint_id);

  void set_completion_callback(CompletionCallback callback);
  void add_completion_event(const CompletionEvent& event);
  Status get_completion_event(CompletionEvent& event, int timeout_ms = -1);

  void start();
  void stop();

private:
  void io_thread_func(int thread_id);
  void process_endpoints();

  std::atomic<bool> stop_io_threads_;
  std::vector<std::thread> io_threads_;
  std::mutex endpoints_mutex_;
  std::map<int, std::unique_ptr<Endpoint>> endpoints_;
  int next_endpoint_id_;

  std::mutex completion_mutex_;
  std::condition_variable completion_cv_;
  std::queue<CompletionEvent> completion_queue_;
  CompletionCallback completion_callback_;
};

// Endpoint implementation
Endpoint::Endpoint(TcpManager* manager, int id) : manager_(manager), id_(id), sockfd_(-1), listenfd_(-1), connected_(false) {}

Endpoint::~Endpoint() {
  disconnect();
}

Status Endpoint::init(const EndpointConfig& config) {
  config_ = config;
  return SUCCESS;
}

Status Endpoint::register_memory_region(void* addr, size_t length, MemoryRegion& mr) {
  mr.addr = addr;
  mr.length = length;
  static uint32_t key_counter = 1;
  mr.lkey = key_counter++;
  mr.rkey = mr.lkey;
  return SUCCESS;
}

Status Endpoint::deregister_memory_region(MemoryRegion& mr) {
  mr = MemoryRegion{};
  return SUCCESS;
}

Status Endpoint::connect(const std::string& remote_ip, int remote_port) {
  if (connected_) {
    return FAILURE;
  }

  sockfd_ = socket(AF_INET, SOCK_STREAM, 0);
  if (sockfd_ < 0) {
    return FAILURE;
  }

  // 设置为非阻塞
  int flags = fcntl(sockfd_, F_GETFL, 0);
  fcntl(sockfd_, F_SETFL, flags | O_NONBLOCK);

  struct sockaddr_in server_addr;
  memset(&server_addr, 0, sizeof(server_addr));
  server_addr.sin_family = AF_INET;
  server_addr.sin_port = htons(remote_port);
  inet_pton(AF_INET, remote_ip.c_str(), &server_addr.sin_addr);

  // 非阻塞连接
  int result = ::connect(sockfd_, (struct sockaddr*)&server_addr, sizeof(server_addr));
  if (result < 0 && errno != EINPROGRESS) {
    close(sockfd_);
    sockfd_ = -1;
    return FAILURE;
  }

  // 连接正在进行中，等待IO线程处理
  return SUCCESS;
}

Status Endpoint::listen(int backlog) {
  listenfd_ = socket(AF_INET, SOCK_STREAM, 0);
  if (listenfd_ < 0) {
    return FAILURE;
  }

  int optval = 1;
  setsockopt(listenfd_, SOL_SOCKET, SO_REUSEADDR, &optval, sizeof(optval));

  struct sockaddr_in server_addr;
  memset(&server_addr, 0, sizeof(server_addr));
  server_addr.sin_family = AF_INET;
  server_addr.sin_port = htons(config_.port);
  server_addr.sin_addr.s_addr = INADDR_ANY;

  if (bind(listenfd_, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
    close(listenfd_);
    listenfd_ = -1;
    return FAILURE;
  }

  if (::listen(listenfd_, backlog) < 0) {
    close(listenfd_);
    listenfd_ = -1;
    return FAILURE;
  }

  // 设置为非阻塞
  int flags = fcntl(listenfd_, F_GETFL, 0);
  fcntl(listenfd_, F_SETFL, flags | O_NONBLOCK);

  return SUCCESS;
}

Status Endpoint::accept() {
  if (listenfd_ < 0) {
    return FAILURE;
  }

  struct sockaddr_in client_addr;
  socklen_t client_len = sizeof(client_addr);
  
  int client_fd = ::accept(listenfd_, (struct sockaddr*)&client_addr, &client_len);
  if (client_fd < 0) {
    if (errno == EWOULDBLOCK || errno == EAGAIN) {
      return SUCCESS; // 没有连接待接受，但不是错误
    }
    return FAILURE;
  }

  // 设置为非阻塞
  int flags = fcntl(client_fd, F_GETFL, 0);
  fcntl(client_fd, F_SETFL, flags | O_NONBLOCK);

  sockfd_ = client_fd;
  connected_ = true;
  
  CompletionEvent event;
  event.type = CONNECTION_EVENT;
  event.status = SUCCESS;
  event.endpoint_id = id_;
  manager_->add_completion_event(event);
  
  return SUCCESS;
}

Status Endpoint::disconnect() {
  if (sockfd_ >= 0) {
    shutdown(sockfd_, SHUT_RDWR);
    close(sockfd_);
    sockfd_ = -1;
  }
  
  if (listenfd_ >= 0) {
    close(listenfd_);
    listenfd_ = -1;
  }
  
  connected_ = false;
  return SUCCESS;
}

Status Endpoint::post_send(const MemoryRegion& mr, size_t offset, size_t length, void* context) {
  if (!connected_ || sockfd_ < 0) {
    return FAILURE;
  }

  if (offset + length > mr.length) {
    return FAILURE;
  }

  std::lock_guard<std::mutex> lock(io_mutex_);
  pending_io_.push({PendingIO::SEND, mr, offset, length, context});
  return SUCCESS;
}

Status Endpoint::post_recv(const MemoryRegion& mr, size_t offset, size_t length, void* context) {
  if (!connected_ || sockfd_ < 0) {
    return FAILURE;
  }

  if (offset + length > mr.length) {
    return FAILURE;
  }

  std::lock_guard<std::mutex> lock(io_mutex_);
  pending_io_.push({PendingIO::RECV, mr, offset, length, context});
  return SUCCESS;
}

Status Endpoint::write(const MemoryRegion& local_mr, size_t local_offset, 
                       const MemoryRegion& remote_mr, size_t remote_offset, 
                       size_t length, void* context) {
  return post_send(local_mr, local_offset, length, context);
}

Status Endpoint::read(const MemoryRegion& local_mr, size_t local_offset, 
                      const MemoryRegion& remote_mr, size_t remote_offset, 
                      size_t length, void* context) {
  return post_recv(local_mr, local_offset, length, context);
}

void Endpoint::process_io() {
  // 处理连接状态
  if (sockfd_ >= 0 && !connected_) {
    // 检查非阻塞连接是否完成
    int error = 0;
    socklen_t len = sizeof(error);
    if (getsockopt(sockfd_, SOL_SOCKET, SO_ERROR, &error, &len) < 0) {
      disconnect();
      return;
    }
    
    if (error == 0) {
      connected_ = true;
      CompletionEvent event;
      event.type = CONNECTION_EVENT;
      event.status = SUCCESS;
      event.endpoint_id = id_;
      manager_->add_completion_event(event);
    } else {
      disconnect();
      CompletionEvent event;
      event.type = CONNECTION_EVENT;
      event.status = FAILURE;
      event.endpoint_id = id_;
      manager_->add_completion_event(event);
      return;
    }
  }
  
  // 处理监听socket
  if (listenfd_ >= 0) {
    accept();
  }
  
  // 处理待处理的IO操作
  std::queue<PendingIO> io_ops;
  {
    std::lock_guard<std::mutex> lock(io_mutex_);
    io_ops = std::move(pending_io_);
  }
  
  while (!io_ops.empty()) {
    auto& io = io_ops.front();
    
    if (io.type == PendingIO::SEND) {
      char* data = static_cast<char*>(io.mr.addr) + io.offset;
      
      // 先发送长度信息
      uint32_t net_length = htonl(static_cast<uint32_t>(io.length));
      ssize_t sent = send(sockfd_, &net_length, sizeof(net_length), MSG_NOSIGNAL | MSG_DONTWAIT);
      
      if (sent == sizeof(net_length)) {
        // 发送实际数据
        sent = send(sockfd_, data, io.length, MSG_NOSIGNAL | MSG_DONTWAIT);
        
        if (sent == static_cast<ssize_t>(io.length)) {
          CompletionEvent event;
          event.type = SEND_COMPLETION;
          event.status = SUCCESS;
          event.byte_count = io.length;
          event.context = io.context;
          event.endpoint_id = id_;
          manager_->add_completion_event(event);
        } else if (sent < 0 && (errno == EWOULDBLOCK || errno == EAGAIN)) {
          // 发送缓冲区满，重新加入队列
          std::lock_guard<std::mutex> lock(io_mutex_);
          pending_io_.push(io);
        } else {
          // 发送失败
          CompletionEvent event;
          event.type = SEND_COMPLETION;
          event.status = FAILURE;
          event.byte_count = 0;
          event.context = io.context;
          event.endpoint_id = id_;
          manager_->add_completion_event(event);
        }
      } else if (sent < 0 && (errno == EWOULDBLOCK || errno == EAGAIN)) {
        // 发送缓冲区满，重新加入队列
        std::lock_guard<std::mutex> lock(io_mutex_);
        pending_io_.push(io);
      } else {
        // 发送失败
        CompletionEvent event;
        event.type = SEND_COMPLETION;
        event.status = FAILURE;
        event.byte_count = 0;
        event.context = io.context;
        event.endpoint_id = id_;
        manager_->add_completion_event(event);
      }
    } else if (io.type == PendingIO::RECV) {
      char* data = static_cast<char*>(io.mr.addr) + io.offset;
      
      // 先接收长度信息
      uint32_t net_length;
      ssize_t received = recv(sockfd_, &net_length, sizeof(net_length), MSG_DONTWAIT);
      
      if (received == sizeof(net_length)) {
        uint32_t msg_length = ntohl(net_length);
        
        if (msg_length <= io.length) {
          // 接收实际数据
          received = recv(sockfd_, data, msg_length, MSG_DONTWAIT);
          
          if (received == static_cast<ssize_t>(msg_length)) {
            CompletionEvent event;
            event.type = RECV_COMPLETION;
            event.status = SUCCESS;
            event.byte_count = msg_length;
            event.context = io.context;
            event.endpoint_id = id_;
            manager_->add_completion_event(event);
          } else if (received < 0 && (errno == EWOULDBLOCK || errno == EAGAIN)) {
            // 数据未就绪，重新加入队列
            std::lock_guard<std::mutex> lock(io_mutex_);
            pending_io_.push(io);
          } else {
            // 接收失败
            CompletionEvent event;
            event.type = RECV_COMPLETION;
            event.status = FAILURE;
            event.byte_count = 0;
            event.context = io.context;
            event.endpoint_id = id_;
            manager_->add_completion_event(event);
          }
        } else {
          // 消息太大，无法放入缓冲区
          CompletionEvent event;
          event.type = RECV_COMPLETION;
          event.status = FAILURE;
          event.byte_count = 0;
          event.context = io.context;
          event.endpoint_id = id_;
          manager_->add_completion_event(event);
        }
      } else if (received < 0 && (errno == EWOULDBLOCK || errno == EAGAIN)) {
        // 数据未就绪，重新加入队列
        std::lock_guard<std::mutex> lock(io_mutex_);
        pending_io_.push(io);
      } else if (received == 0) {
        // 连接关闭
        CompletionEvent event;
        event.type = RECV_COMPLETION;
        event.status = CONNECTION_CLOSED;
        event.byte_count = 0;
        event.context = io.context;
        event.endpoint_id = id_;
        manager_->add_completion_event(event);
        disconnect();
      } else {
        // 接收失败
        CompletionEvent event;
        event.type = RECV_COMPLETION;
        event.status = FAILURE;
        event.byte_count = 0;
        event.context = io.context;
        event.endpoint_id = id_;
        manager_->add_completion_event(event);
      }
    }
    
    io_ops.pop();
  }
}

bool Endpoint::has_pending_io() const {
  std::lock_guard<std::mutex> lock(io_mutex_);
  return !pending_io_.empty() || 
         (sockfd_ >= 0 && !connected_) || 
         (listenfd_ >= 0);
}

// TcpManager implementation
TcpManager::TcpManager(int io_threads) : stop_io_threads_(false), next_endpoint_id_(0) {
  for (int i = 0; i < io_threads; i++) {
    io_threads_.emplace_back(&TcpManager::io_thread_func, this, i);
  }
}

TcpManager::~TcpManager() {
  stop();
  
  std::lock_guard<std::mutex> lock(endpoints_mutex_);
  endpoints_.clear();
}

Endpoint* TcpManager::create_endpoint() {
  std::lock_guard<std::mutex> lock(endpoints_mutex_);
  int id = next_endpoint_id_++;
  auto endpoint = std::make_unique<Endpoint>(this, id);
  auto* ptr = endpoint.get();
  endpoints_[id] = std::move(endpoint);
  return ptr;
}

void TcpManager::destroy_endpoint(int endpoint_id) {
  std::lock_guard<std::mutex> lock(endpoints_mutex_);
  endpoints_.erase(endpoint_id);
}

Endpoint* TcpManager::get_endpoint(int endpoint_id) {
  std::lock_guard<std::mutex> lock(endpoints_mutex_);
  auto it = endpoints_.find(endpoint_id);
  if (it != endpoints_.end()) {
    return it->second.get();
  }
  return nullptr;
}

void TcpManager::set_completion_callback(CompletionCallback callback) {
  completion_callback_ = callback;
}

void TcpManager::add_completion_event(const CompletionEvent& event) {
  std::lock_guard<std::mutex> lock(completion_mutex_);
  completion_queue_.push(event);
  completion_cv_.notify_one();
  
  if (completion_callback_) {
    completion_callback_(event);
  }
}

Status TcpManager::get_completion_event(CompletionEvent& event, int timeout_ms) {
  std::unique_lock<std::mutex> lock(completion_mutex_);
  
  if (completion_queue_.empty()) {
    if (timeout_ms < 0) {
      completion_cv_.wait(lock, [this] { return !completion_queue_.empty() || stop_io_threads_; });
    } else {
      auto status = completion_cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                                          [this] { return !completion_queue_.empty() || stop_io_threads_; });
      if (!status) {
        return TIMEOUT;
      }
    }
  }
  
  if (stop_io_threads_) {
    return FAILURE;
  }
  
  if (!completion_queue_.empty()) {
    event = completion_queue_.front();
    completion_queue_.pop();
    return SUCCESS;
  }
  
  return FAILURE;
}

void TcpManager::start() {
  stop_io_threads_ = false;
}

void TcpManager::stop() {
  stop_io_threads_ = true;
  for (auto& thread : io_threads_) {
    if (thread.joinable()) {
      thread.join();
    }
  }
  io_threads_.clear();
}

void TcpManager::io_thread_func(int thread_id) {
  // 准备poll文件描述符
  std::vector<pollfd> fds;
  std::vector<Endpoint*> endpoints;
  
  while (!stop_io_threads_) {
    // 获取所有端点
    {
      std::lock_guard<std::mutex> lock(endpoints_mutex_);
      endpoints.clear();
      fds.clear();
      
      for (auto& pair : endpoints_) {
        Endpoint* ep = pair.second.get();
        endpoints.push_back(ep);
        
        if (ep->get_socket() >= 0) {
          pollfd pfd;
          pfd.fd = ep->get_socket();
          pfd.events = POLLIN | POLLOUT;
          pfd.revents = 0;
          fds.push_back(pfd);
        }
        
        if (ep->has_pending_io()) {
          ep->process_io();
        }
      }
    }
    
    // 如果没有文件描述符需要监视，休眠一会儿
    if (fds.empty()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      continue;
    }
    
    // 使用poll等待IO事件
    int result = poll(fds.data(), fds.size(), 10);
    if (result > 0) {
      // 处理有事件的端点
      for (size_t i = 0; i < fds.size(); i++) {
        if (fds[i].revents & (POLLIN | POLLOUT | POLLERR | POLLHUP)) {
          endpoints[i]->process_io();
        }
      }
    } else if (result < 0) {
      // poll错误
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }
}

} // namespace tcp_verbs
