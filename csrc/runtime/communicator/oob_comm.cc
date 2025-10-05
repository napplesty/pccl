#include "runtime/communicator/oob_comm.h"
#include "utils/logging.h"
#include <thread>
#include <chrono>

namespace pccl::communicator {

AsioOobChannel::AsioOobChannel() : running_(false) {}

AsioOobChannel::~AsioOobChannel() {
  shutdown();
}

bool AsioOobChannel::init(const Endpoint& self_endpoint) {
  if (running_) {
    PCCL_LOG_WARN("OOB channel already initialized");
    return false;
  }
  
  self_endpoint_ = self_endpoint;
  
  try {
    auto ip_str = self_endpoint.attributes_.at("ip");
    auto port_str = self_endpoint.attributes_.at("port");
    uint16_t port = std::stoi(port_str);
    
    asio::ip::tcp::endpoint endpoint(asio::ip::make_address(ip_str), port);
    acceptor_.open(endpoint.protocol());
    acceptor_.set_option(asio::ip::tcp::acceptor::reuse_address(true));
    acceptor_.bind(endpoint);
    acceptor_.listen();
    
    running_ = true;
    io_thread_ = std::make_unique<std::thread>([this]() {
      startAccept();
      io_context_.run();
    });
    
    PCCL_LOG_INFO("OOB channel initialized on {}:{}", ip_str, port);
    return true;
  } catch (const std::exception& e) {
    PCCL_LOG_ERROR("Failed to initialize OOB channel: {}", e.what());
    return false;
  }
}

void AsioOobChannel::shutdown() {
  if (!running_) return;
  
  running_ = false;
  io_context_.stop();
  
  if (io_thread_ && io_thread_->joinable()) {
    io_thread_->join();
  }
  
  {
    std::lock_guard<std::shared_mutex> lock(connections_mutex_);
    connections_.clear();
  }
  
  PCCL_LOG_INFO("OOB channel shutdown");
}

bool AsioOobChannel::send(const OobMessage& msg, const Endpoint& dst) {
  if (!running_) return false;
  
  try {
    auto key = endpointToKey(dst);
    std::shared_lock<std::shared_mutex> lock(connections_mutex_);
    auto it = connections_.find(key);
    if (it == connections_.end()) {
      if (!connectToEndpoint(dst)) {
        return false;
      }
      it = connections_.find(key);
      if (it == connections_.end()) return false;
    }
    
    auto conn = it->second;
    auto msg_data = msg.toJson().dump();
    std::vector<uint8_t> data(msg_data.begin(), msg_data.end());
    
    std::lock_guard<std::mutex> conn_lock(conn->send_queue_mutex_);
    conn->send_queue_.push(std::move(data));
    
    startSend(conn);
    return true;
  } catch (const std::exception& e) {
    PCCL_LOG_ERROR("Failed to send OOB message: {}", e.what());
    return false;
  }
}

bool AsioOobChannel::broadcast(const OobMessage& msg, const std::vector<Endpoint>& targets) {
  bool success = true;
  for (const auto& target : targets) {
    if (!send(msg, target)) {
      success = false;
    }
  }
  return success;
}

bool AsioOobChannel::poll(OobMessage* msg, uint32_t timeout_ms) {
  if (!running_ || !msg) return false;
  
  std::unique_lock<std::mutex> lock(queue_mutex_);
  if (message_queue_.empty()) {
    if (timeout_ms == 0) return false;
    
    queue_cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                      [this]() { return !message_queue_.empty() || !running_; });
    
    if (!running_ || message_queue_.empty()) return false;
  }
  
  *msg = message_queue_.front();
  message_queue_.pop();
  return true;
}

bool AsioOobChannel::registerHandler(OobMsgType type, std::function<void(const OobMessage&)> handler) {
  std::lock_guard<std::mutex> lock(handlers_mutex_);
  handlers_[type] = handler;
  return true;
}

std::vector<Endpoint> AsioOobChannel::getConnectedNodes() const {
  std::vector<Endpoint> endpoints;
  std::shared_lock<std::shared_mutex> lock(connections_mutex_);
  
  for (const auto& pair : connections_) {
    if (pair.second->connected_) {
      Endpoint ep;
      ep.attributes_["ip"] = pair.second->remote_endpoint_.address().to_string();
      ep.attributes_["port"] = std::to_string(pair.second->remote_endpoint_.port());
      endpoints.push_back(ep);
    }
  }
  
  return endpoints;
}

bool AsioOobChannel::isConnected(const Endpoint& endpoint) const {
  auto key = endpointToKey(endpoint);
  std::shared_lock<std::shared_mutex> lock(connections_mutex_);
  auto it = connections_.find(key);
  return it != connections_.end() && it->second->connected_;
}

void AsioOobChannel::startAccept() {
  if (!running_) return;
  
  auto socket = std::make_shared<asio::ip::tcp::socket>(io_context_);
  acceptor_.async_accept(*socket, [this, socket](asio::error_code ec) {
    if (!ec) {
      handleAccept(socket, ec);
    } else {
      PCCL_LOG_ERROR("Accept error: {}", ec.message());
    }
    
    if (running_) {
      startAccept();
    }
  });
}

void AsioOobChannel::handleAccept(std::shared_ptr<asio::ip::tcp::socket> socket, 
                                 const asio::error_code& error) {
  if (error) {
    PCCL_LOG_ERROR("Accept handler error: {}", error.message());
    return;
  }
  
  auto remote_ep = socket->remote_endpoint();
  auto conn = std::make_shared<Connection>();
  conn->socket_ = socket;
  conn->remote_endpoint_ = remote_ep;
  conn->connected_ = true;
  conn->last_activity_ = std::chrono::steady_clock::now();
  
  auto key = endpointToKey(remote_ep.address().to_string(), remote_ep.port());
  {
    std::lock_guard<std::shared_mutex> lock(connections_mutex_);
    connections_[key] = conn;
  }
  
  startReceive(conn);
  PCCL_LOG_INFO("New connection from {}:{}", 
               remote_ep.address().to_string(), remote_ep.port());
}

void AsioOobChannel::startReceive(std::shared_ptr<Connection> conn) {
  if (!conn->connected_) return;
  
  conn->recv_buffer_.resize(1024);
  conn->socket_->async_read_some(
    asio::buffer(conn->recv_buffer_),
    [this, conn](asio::error_code ec, size_t bytes_transferred) {
      handleReceive(conn, ec, bytes_transferred);
    }
  );
}

void AsioOobChannel::handleReceive(std::shared_ptr<Connection> conn, 
                                  const asio::error_code& error, 
                                  size_t bytes_transferred) {
  if (error) {
    PCCL_LOG_ERROR("Receive error: {}", error.message());
    conn->connected_ = false;
    return;
  }
  
  conn->last_activity_ = std::chrono::steady_clock::now();
  
  try {
    std::string msg_str(conn->recv_buffer_.begin(), 
                       conn->recv_buffer_.begin() + bytes_transferred);
    auto json_data = nlohmann::json::parse(msg_str);
    OobMessage msg = OobMessage::fromJson(json_data);
    
    {
      std::lock_guard<std::mutex> lock(queue_mutex_);
      message_queue_.push(msg);
      queue_cv_.notify_one();
    }
    
    handleMessage(msg);
  } catch (const std::exception& e) {
    PCCL_LOG_ERROR("Failed to parse OOB message: {}", e.what());
  }
  
  if (conn->connected_) {
    startReceive(conn);
  }
}

void AsioOobChannel::startSend(std::shared_ptr<Connection> conn) {
  if (!conn->connected_) return;
  
  std::lock_guard<std::mutex> lock(conn->send_queue_mutex_);
  if (conn->send_queue_.empty()) return;
  
  auto& data = conn->send_queue_.front();
  asio::async_write(*conn->socket_, asio::buffer(data),
                   [this, conn](asio::error_code ec, size_t bytes_transferred) {
                     handleSend(conn, ec, bytes_transferred);
                   });
}

void AsioOobChannel::handleSend(std::shared_ptr<Connection> conn, 
                               const asio::error_code& error, 
                               size_t bytes_transferred) {
  if (error) {
    PCCL_LOG_ERROR("Send error: {}", error.message());
    conn->connected_ = false;
    return;
  }
  
  conn->last_activity_ = std::chrono::steady_clock::now();
  
  std::lock_guard<std::mutex> lock(conn->send_queue_mutex_);
  if (!conn->send_queue_.empty()) {
    conn->send_queue_.pop();
  }
  
  if (!conn->send_queue_.empty()) {
    startSend(conn);
  }
}

bool AsioOobChannel::connectToEndpoint(const Endpoint& endpoint) {
  try {
    auto tcp_ep = endpointToTcpEndpoint(endpoint);
    auto socket = std::make_shared<asio::ip::tcp::socket>(io_context_);
    
    socket->async_connect(tcp_ep, [this, socket, endpoint](asio::error_code ec) {
      if (!ec) {
        auto conn = std::make_shared<Connection>();
        conn->socket_ = socket;
        conn->remote_endpoint_ = socket->remote_endpoint();
        conn->connected_ = true;
        conn->last_activity_ = std::chrono::steady_clock::now();
        
        auto key = endpointToKey(endpoint);
        {
          std::lock_guard<std::shared_mutex> lock(connections_mutex_);
          connections_[key] = conn;
        }
        
        startReceive(conn);
        PCCL_LOG_INFO("Connected to {}:{}", 
                     endpoint.attributes_.at("ip"), endpoint.attributes_.at("port"));
      } else {
        PCCL_LOG_ERROR("Connect error: {}", ec.message());
      }
    });
    
    return true;
  } catch (const std::exception& e) {
    PCCL_LOG_ERROR("Connect exception: {}", e.what());
    return false;
  }
}

void AsioOobChannel::handleMessage(const OobMessage& msg) {
  std::lock_guard<std::mutex> lock(handlers_mutex_);
  auto it = handlers_.find(msg.type_);
  if (it != handlers_.end()) {
    it->second(msg);
  }
}

void AsioOobChannel::cleanupStaleConnections() {
  auto now = std::chrono::steady_clock::now();
  std::lock_guard<std::shared_mutex> lock(connections_mutex_);
  
  for (auto it = connections_.begin(); it != connections_.end(); ) {
    auto duration = std::chrono::duration_cast<std::chrono::minutes>(
      now - it->second->last_activity_);
    
    if (duration > std::chrono::minutes(5) || !it->second->connected_) {
      it = connections_.erase(it);
    } else {
      ++it;
    }
  }
}

std::string AsioOobChannel::endpointToKey(const Endpoint& endpoint) const {
  return endpoint.attributes_.at("ip") + ":" + endpoint.attributes_.at("port");
}

std::string AsioOobChannel::endpointToKey(const std::string& ip, uint16_t port) const {
  return ip + ":" + std::to_string(port);
}

asio::ip::tcp::endpoint AsioOobChannel::endpointToTcpEndpoint(const Endpoint& endpoint) const {
  auto ip_str = endpoint.attributes_.at("ip");
  auto port_str = endpoint.attributes_.at("port");
  uint16_t port = std::stoi(port_str);
  
  return asio::ip::tcp::endpoint(asio::ip::make_address(ip_str), port);
}

} // namespace pccl::communicator
