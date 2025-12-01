#include <cluster/daemon.h>
#include <common/time.h>
#include <common/serialize.h>
#include <nlohmann/json.hpp>

namespace engine_c {

std::string ClusterMessage::serialize() const {
  nlohmann::json j;
  j["event_type"] = static_cast<int>(event_type);
  j["data"] = data;
  j["source_rank"] = source_rank;
  j["target_rank"] = target_rank;
  j["timestamp"] = timestamp;
  return j.dump();
}

ClusterMessage ClusterMessage::deserialize(const std::string& data) {
  nlohmann::json j = nlohmann::json::parse(data);
  ClusterMessage msg;
  msg.event_type = static_cast<ClusterEventType>(j["event_type"].get<int>());
  msg.data = j["data"].get<std::string>();
  msg.source_rank = j["source_rank"].get<int>();
  msg.target_rank = j["target_rank"].get<int>();
  msg.timestamp = j["timestamp"].get<uint64_t>();
  return msg;
}

Daemon::Daemon() 
  : rank_(-1), is_master_(false), port_(0), running_(false), connected_to_master_(false) {
}

Daemon::~Daemon() {
  stop();
}

bool Daemon::start(int rank, const std::string& host, int port, bool is_master) {
  if (running_) {
    return false;
  }

  rank_ = rank;
  host_ = host;
  port_ = port;
  is_master_ = is_master;

  io_context_ = std::make_unique<asio::io_context>();
  
  if (is_master_) {
    acceptor_ = std::make_unique<asio::ip::tcp::acceptor>(
      *io_context_, 
      asio::ip::tcp::endpoint(asio::ip::make_address(host), port)
    );
  }

  running_ = true;

  io_thread_ = std::make_unique<std::thread>([this]() {
    if (is_master_) {
      startAccept();
    }
    io_context_->run();
  });

  worker_thread_ = std::make_unique<std::thread>([this]() {
    workerThread();
  });

  return true;
}

bool Daemon::stop() {
  if (!running_) {
    return false;
  }

  running_ = false;
  connected_to_master_ = false;

  if (io_context_) {
    io_context_->stop();
  }

  queue_cv_.notify_all();

  if (io_thread_ && io_thread_->joinable()) {
    io_thread_->join();
  }

  if (worker_thread_ && worker_thread_->joinable()) {
    worker_thread_->join();
  }

  {
    std::lock_guard<std::mutex> lock(sockets_mutex_);
    slave_sockets_.clear();
  }

  master_socket_.reset();
  acceptor_.reset();
  io_context_.reset();

  return true;
}

bool Daemon::isRunning() const {
  return running_;
}

bool Daemon::connectToMaster(const std::string& master_host, int master_port) {
  if (is_master_ || connected_to_master_) {
    return false;
  }

  master_socket_ = std::make_shared<asio::ip::tcp::socket>(*io_context_);
  
  asio::ip::tcp::resolver resolver(*io_context_);
  auto endpoints = resolver.resolve(master_host, std::to_string(master_port));
  
  asio::connect(*master_socket_, endpoints);
  connected_to_master_ = true;

  ClusterMessage join_msg;
  join_msg.event_type = ClusterEventType::JoinSelf;
  join_msg.source_rank = rank_;
  join_msg.target_rank = 0;
  join_msg.timestamp = utils::getCurrentTimeNanos();
  join_msg.data = host_ + ":" + std::to_string(port_);

  sendMessageToSocket(master_socket_, join_msg);
  startReceive();

  return true;
}

bool Daemon::disconnectFromMaster() {
  if (!connected_to_master_ || !master_socket_) {
    return false;
  }

  ClusterMessage quit_msg;
  quit_msg.event_type = ClusterEventType::QuitSelf;
  quit_msg.source_rank = rank_;
  quit_msg.target_rank = 0;
  quit_msg.timestamp = utils::getCurrentTimeNanos();

  sendMessageToSocket(master_socket_, quit_msg);
  
  master_socket_->close();
  master_socket_.reset();
  connected_to_master_ = false;

  return true;
}

bool Daemon::sendMessage(const ClusterMessage& message) {
  if (!running_) {
    return false;
  }

  if (is_master_) {
    std::lock_guard<std::mutex> lock(sockets_mutex_);
    auto it = slave_sockets_.find(message.target_rank);
    if (it != slave_sockets_.end() && it->second) {
      sendMessageToSocket(it->second, message);
      return true;
    }
  } else {
    if (connected_to_master_ && master_socket_) {
      sendMessageToSocket(master_socket_, message);
      return true;
    }
  }

  return false;
}

bool Daemon::broadcastMessage(const ClusterMessage& message) {
  if (!running_ || !is_master_) {
    return false;
  }

  std::lock_guard<std::mutex> lock(sockets_mutex_);
  bool success = true;
  
  for (auto& [rank, socket] : slave_sockets_) {
    if (socket) {
      sendMessageToSocket(socket, message);
    }
  }

  return success;
}

void Daemon::setEventCallback(std::function<void(const ClusterMessage&)> callback) {
  event_callback_ = callback;
}

void Daemon::startAccept() {
  auto socket = std::make_shared<asio::ip::tcp::socket>(*io_context_);
  
  acceptor_->async_accept(*socket, [this, socket](const std::error_code& ec) {
    if (!ec) {
      handleAccept(socket);
    }
    
    if (running_) {
      startAccept();
    }
  });
}

void Daemon::startReceive() {
  if (!is_master_ && master_socket_) {
    auto buffer = std::make_shared<std::vector<char>>(4096);
    
    master_socket_->async_read_some(
      asio::buffer(*buffer),
      [this, buffer](const std::error_code& ec, std::size_t bytes_transferred) {
        if (!ec) {
          std::string data(buffer->data(), bytes_transferred);
          ClusterMessage msg = ClusterMessage::deserialize(data);
          processMessage(msg);
        }
        
        if (running_ && connected_to_master_) {
          startReceive();
        }
      }
    );
  }
}

void Daemon::handleAccept(std::shared_ptr<asio::ip::tcp::socket> socket) {
  auto buffer = std::make_shared<std::vector<char>>(4096);
  
  socket->async_read_some(
    asio::buffer(*buffer),
    [this, socket, buffer](const std::error_code& ec, std::size_t bytes_transferred) {
      if (!ec) {
        std::string data(buffer->data(), bytes_transferred);
        ClusterMessage msg = ClusterMessage::deserialize(data);
        
        if (msg.event_type == ClusterEventType::JoinSelf) {
          std::lock_guard<std::mutex> lock(sockets_mutex_);
          slave_sockets_[msg.source_rank] = socket;
        }
        
        processMessage(msg);
        
        if (socket->is_open()) {
          handleAccept(socket);
        }
      }
    }
  );
}

void Daemon::handleMasterConnection() {
}

void Daemon::processMessage(const ClusterMessage& message) {
  {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    message_queue_.push(message);
  }
  queue_cv_.notify_one();
}

void Daemon::sendMessageToSocket(std::shared_ptr<asio::ip::tcp::socket> socket, 
                                const ClusterMessage& message) {
  if (!socket || !socket->is_open()) {
    return;
  }

  std::string data = message.serialize();
  asio::write(*socket, asio::buffer(data));
}

void Daemon::workerThread() {
  while (running_) {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    queue_cv_.wait(lock, [this]() { return !message_queue_.empty() || !running_; });
    
    while (!message_queue_.empty()) {
      ClusterMessage msg = message_queue_.front();
      message_queue_.pop();
      lock.unlock();
      
      if (event_callback_) {
        event_callback_(msg);
      }
      
      lock.lock();
    }
  }
}

}
