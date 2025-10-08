#include "runtime/communicator/oob_comm.h"
#include "utils/logging.h"
#include <nlohmann/json.hpp>

namespace pccl::communicator {

nlohmann::json OobMessage::toJson() const {
  nlohmann::json j;
  j["type"] = static_cast<int>(type_);
  j["src_rank"] = src_rank;
  j["payload"] = payload;
  j["timestamp"] = timestamp;
  return j;
}

OobMessage OobMessage::fromJson(const nlohmann::json& json_data) {
  OobMessage msg;
  msg.type_ = static_cast<OobMsgType>(json_data["type"].get<int>());
  msg.src_rank = json_data["src_rank"].get<int>();
  msg.payload = json_data["payload"].get<std::string>();
  msg.timestamp = json_data.value("timestamp", 0);
  return msg;
}

AsioOobChannel::AsioOobChannel() 
  : acceptor_(io_context_) {
}

AsioOobChannel::~AsioOobChannel() {
  shutdown();
}

bool AsioOobChannel::init(const Endpoint& self_endpoint) {
  try {
    local_ip_ = self_endpoint.attributes_.at("pccl.oob.ip");
    local_port_ = std::stoi(self_endpoint.attributes_.at("pccl.oob.port"));
    
    asio::ip::tcp::endpoint endpoint(asio::ip::make_address(local_ip_), local_port_);
    acceptor_.open(endpoint.protocol());
    acceptor_.set_option(asio::ip::tcp::acceptor::reuse_address(true));
    acceptor_.bind(endpoint);
    acceptor_.listen();
    
    running_ = true;
    startAccept();
    
    io_thread_ = std::thread([this]() {
      io_context_.run();
    });
    
    PCCL_LOG_INFO("OOB channel initialized on {}:{}", local_ip_, local_port_);
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
  
  if (io_thread_.joinable()) {
    io_thread_.join();
  }
  
  asio::error_code ec;
  acceptor_.close(ec);
  
  PCCL_LOG_INFO("OOB channel shutdown");
}

bool AsioOobChannel::send(const OobMessage& msg, const Endpoint& dst) {
  try {
    std::string dst_ip = dst.attributes_.at("pccl.oob.ip");
    uint16_t dst_port = std::stoi(dst.attributes_.at("pccl.oob.port"));
    
    asio::ip::tcp::endpoint endpoint(asio::ip::make_address(dst_ip), dst_port);
    asio::ip::tcp::socket socket(io_context_);
    socket.connect(endpoint);
    
    auto json_str = msg.toJson().dump();
    uint32_t length = htonl(static_cast<uint32_t>(json_str.size()));
    
    std::vector<asio::const_buffer> buffers;
    buffers.push_back(asio::buffer(&length, sizeof(length)));
    buffers.push_back(asio::buffer(json_str));
    
    asio::write(socket, buffers);
    socket.close();
    
    return true;
  } catch (const std::exception& e) {
    PCCL_LOG_ERROR("OOB send failed: {}", e.what());
    return false;
  }
}

bool AsioOobChannel::broadcast(const OobMessage& msg, const std::vector<Endpoint>& destinations) {
  bool all_success = true;
  for (const auto& dst : destinations) {
    if (!send(msg, dst)) {
      all_success = false;
    }
  }
  return all_success;
}

bool AsioOobChannel::registerHandler(OobMsgType type, std::function<void(const OobMessage&)> handler) {
  std::unique_lock lock(handlers_mutex_);
  handlers_[type] = handler;
  return true;
}

std::queue<OobMessage> AsioOobChannel::getPendingMessages() {
  std::lock_guard lock(queue_mutex_);
  std::queue<OobMessage> result = std::move(message_queue_);
  return result;
}

void AsioOobChannel::startAccept() {
  if (!running_) return;
  
  auto socket = std::make_shared<asio::ip::tcp::socket>(io_context_);
  acceptor_.async_accept(*socket, 
    [this, socket](const asio::error_code& error) {
      handleAccept(error, socket);
    });
}

void AsioOobChannel::handleAccept(const asio::error_code& error, std::shared_ptr<asio::ip::tcp::socket> socket) {
  if (!error) {
    startReceive(socket);
  }
  
  if (running_) {
    startAccept();
  }
}

void AsioOobChannel::startReceive(std::shared_ptr<asio::ip::tcp::socket> socket) {
  auto buffer = std::make_shared<std::vector<char>>(sizeof(uint32_t));
  
  asio::async_read(*socket, asio::buffer(*buffer),
    [this, socket, buffer](const asio::error_code& error, size_t bytes_transferred) {
      if (!error && bytes_transferred == sizeof(uint32_t)) {
        uint32_t msg_length = ntohl(*reinterpret_cast<uint32_t*>(buffer->data()));
        buffer->resize(msg_length);
        
        asio::async_read(*socket, asio::buffer(*buffer),
          [this, socket, buffer](const asio::error_code& error, size_t bytes_transferred) {
            handleReceive(socket, buffer, error, bytes_transferred);
          });
      } else {
        PCCL_LOG_ERROR("OOB receive header failed");
      }
    });
}

void AsioOobChannel::handleReceive(std::shared_ptr<asio::ip::tcp::socket> socket, 
                                  std::shared_ptr<std::vector<char>> buffer,
                                  const asio::error_code& error, size_t bytes_transferred) {
  if (!error && bytes_transferred == buffer->size()) {
    processMessage(*buffer);
  } else {
    PCCL_LOG_ERROR("OOB receive body failed");
  }
  
  socket->close();
}

void AsioOobChannel::processMessage(const std::vector<char>& data) {
  try {
    std::string json_str(data.begin(), data.end());
    auto json_data = nlohmann::json::parse(json_str);
    OobMessage msg = OobMessage::fromJson(json_data);
    
    {
      std::lock_guard lock(queue_mutex_);
      message_queue_.push(msg);
    }
    
    while (true) {
      std::shared_lock lock(handlers_mutex_);
      auto it = handlers_.find(msg.type_);
      if (it != handlers_.end()) {
        it->second(msg);
        break;
      }
    }
    
    if (msg.type_ == OobMsgType::CONFIG_SYNC) {
      processConfigSync(msg);
    }
    
  } catch (const std::exception& e) {
    PCCL_LOG_ERROR("OOB message processing failed: {}", e.what());
  }
}

void AsioOobChannel::processConfigSync(const OobMessage& msg) {
  PCCL_LOG_INFO("Received config sync from rank {}", msg.src_rank);
}

} // namespace pccl::communicator
