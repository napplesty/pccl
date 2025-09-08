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

  int get_id() const { return id_; }
  int get_socket() const { return sockfd_; }
  bool is_connected() const { return connected_; }

private:
  TcpManager* manager_;
  int id_;
  int sockfd_;
  int listenfd_;
  bool connected_;
  EndpointConfig config_;
  
  Status establish_connection(const std::string& ip, int port);
  Status send_data(const void* data, size_t length);
  Status recv_data(void* buffer, size_t length);
};

class TcpManager {
public:
  TcpManager();
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

private:
  std::mutex endpoints_mutex_;
  std::map<int, std::unique_ptr<Endpoint>> endpoints_;
  int next_endpoint_id_;
  CompletionCallback completion_callback_;
};

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

Status Endpoint::establish_connection(const std::string& ip, int port) {
  sockfd_ = socket(AF_INET, SOCK_STREAM, 0);
  if (sockfd_ < 0) {
    return FAILURE;
  }

  struct sockaddr_in server_addr;
  memset(&server_addr, 0, sizeof(server_addr));
  server_addr.sin_family = AF_INET;
  server_addr.sin_port = htons(port);
  inet_pton(AF_INET, ip.c_str(), &server_addr.sin_addr);

  if (::connect(sockfd_, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
    close(sockfd_);
    sockfd_ = -1;
    return FAILURE;
  }

  const uint32_t magic = 0x52444D41;
  if (send_data(&magic, sizeof(magic)) != SUCCESS) {
    close(sockfd_);
    sockfd_ = -1;
    return FAILURE;
  }

  uint32_t response;
  if (recv_data(&response, sizeof(response)) != SUCCESS || response != magic) {
    close(sockfd_);
    sockfd_ = -1;
    return FAILURE;
  }

  connected_ = true;
  return SUCCESS;
}

Status Endpoint::connect(const std::string& remote_ip, int remote_port) {
  if (connected_) {
    return FAILURE;
  }
  return establish_connection(remote_ip, remote_port);
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

  return SUCCESS;
}

Status Endpoint::accept() {
  if (listenfd_ < 0) {
    return FAILURE;
  }

  struct sockaddr_in client_addr;
  socklen_t client_len = sizeof(client_addr);
  
  sockfd_ = ::accept(listenfd_, (struct sockaddr*)&client_addr, &client_len);
  if (sockfd_ < 0) {
    return FAILURE;
  }

  uint32_t magic;
  if (recv_data(&magic, sizeof(magic)) != SUCCESS || magic != 0x52444D41) {
    close(sockfd_);
    sockfd_ = -1;
    return FAILURE;
  }

  if (send_data(&magic, sizeof(magic)) != SUCCESS) {
    close(sockfd_);
    sockfd_ = -1;
    return FAILURE;
  }

  connected_ = true;
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

Status Endpoint::send_data(const void* data, size_t length) {
  ssize_t total_sent = 0;
  while (total_sent < static_cast<ssize_t>(length)) {
    ssize_t sent = send(sockfd_, static_cast<const char*>(data) + total_sent, length - total_sent, 0);
    if (sent < 0) {
      return FAILURE;
    }
    total_sent += sent;
  }
  return SUCCESS;
}

Status Endpoint::recv_data(void* buffer, size_t length) {
  ssize_t total_received = 0;
  while (total_received < static_cast<ssize_t>(length)) {
    ssize_t received = recv(sockfd_, static_cast<char*>(buffer) + total_received, length - total_received, 0);
    if (received <= 0) {
      return received == 0 ? CONNECTION_CLOSED : FAILURE;
    }
    total_received += received;
  }
  return SUCCESS;
}

Status Endpoint::post_send(const MemoryRegion& mr, size_t offset, size_t length, void* context) {
  if (!connected_ || sockfd_ < 0) {
    return FAILURE;
  }

  if (offset + length > mr.length) {
    return FAILURE;
  }

  uint32_t net_length = htonl(static_cast<uint32_t>(length));
  if (send_data(&net_length, sizeof(net_length)) != SUCCESS) {
    disconnect();
    return FAILURE;
  }

  if (send_data(static_cast<char*>(mr.addr) + offset, length) != SUCCESS) {
    disconnect();
    return FAILURE;
  }

  CompletionEvent event;
  event.type = SEND_COMPLETION;
  event.status = SUCCESS;
  event.byte_count = length;
  event.context = context;
  event.endpoint_id = id_;
  manager_->add_completion_event(event);

  return SUCCESS;
}

Status Endpoint::post_recv(const MemoryRegion& mr, size_t offset, size_t length, void* context) {
  if (!connected_ || sockfd_ < 0) {
    return FAILURE;
  }

  if (offset + length > mr.length) {
    return FAILURE;
  }

  uint32_t net_length;
  if (recv_data(&net_length, sizeof(net_length)) != SUCCESS) {
    disconnect();
    return FAILURE;
  }

  uint32_t msg_length = ntohl(net_length);
  if (msg_length > length) {
    return FAILURE;
  }

  if (recv_data(static_cast<char*>(mr.addr) + offset, msg_length) != SUCCESS) {
    disconnect();
    return FAILURE;
  }

  CompletionEvent event;
  event.type = RECV_COMPLETION;
  event.status = SUCCESS;
  event.byte_count = msg_length;
  event.context = context;
  event.endpoint_id = id_;
  manager_->add_completion_event(event);

  return SUCCESS;
}

Status Endpoint::write(const MemoryRegion& local_mr, size_t local_offset, 
                       const MemoryRegion& remote_mr, size_t remote_offset, 
                       size_t length, void* context) {
  Status status = establish_connection(config_.ip, config_.port);
  if (status != SUCCESS) {
    return status;
  }

  status = post_send(local_mr, local_offset, length, context);
  disconnect();
  return status;
}

Status Endpoint::read(const MemoryRegion& local_mr, size_t local_offset, 
                      const MemoryRegion& remote_mr, size_t remote_offset, 
                      size_t length, void* context) {
  Status status = establish_connection(config_.ip, config_.port);
  if (status != SUCCESS) {
    return status;
  }

  status = post_recv(local_mr, local_offset, length, context);
  disconnect();
  return status;
}

TcpManager::TcpManager() : next_endpoint_id_(0) {}

TcpManager::~TcpManager() {
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
  if (completion_callback_) {
    completion_callback_(event);
  }
}

} // namespace tcp_verbs