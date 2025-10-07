#pragma once

#include <runtime/communicator/channel.h>
#include <asio.hpp>
#include <asio/ts/buffer.hpp>
#include <asio/ts/internet.hpp>
#include <thread>
#include <shared_mutex>
#include <queue>

namespace pccl::communicator {

enum class OobMsgType {
  CONFIG_SYNC,
  BUFFER_UPDATE,
  NODE_JOIN,
  NODE_LEAVE,
  HEARTBEAT
};

struct OobMessage {
  OobMsgType type_;
  int src_rank;
  std::string payload;
  uint64_t timestamp;

  nlohmann::json toJson() const;
  static OobMessage fromJson(const nlohmann::json& json_data);
};

class OobChannel {
public:
  virtual ~OobChannel() = default;
  
  virtual bool init(const Endpoint& self_endpoint) = 0;
  virtual void shutdown() = 0;
  virtual bool send(const OobMessage& msg, const Endpoint& dst) = 0;
  virtual bool broadcast(const OobMessage& msg, const std::vector<Endpoint>& destinations) = 0;
  virtual bool registerHandler(OobMsgType type, std::function<void(const OobMessage&)> handler) = 0;
  virtual std::queue<OobMessage> getPendingMessages() = 0;
};

class AsioOobChannel : public OobChannel {
public:
  AsioOobChannel();
  ~AsioOobChannel();

  bool init(const Endpoint& self_endpoint) override;
  void shutdown() override;
  bool send(const OobMessage& msg, const Endpoint& dst) override;
  bool broadcast(const OobMessage& msg, const std::vector<Endpoint>& destinations) override;
  bool registerHandler(OobMsgType type, std::function<void(const OobMessage&)> handler) override;
  std::queue<OobMessage> getPendingMessages() override;

private:
  void startAccept();
  void handleAccept(const asio::error_code& error, std::shared_ptr<asio::ip::tcp::socket> socket);
  void startReceive(std::shared_ptr<asio::ip::tcp::socket> socket);
  void handleReceive(std::shared_ptr<asio::ip::tcp::socket> socket, 
                    std::shared_ptr<std::vector<char>> buffer,
                    const asio::error_code& error, size_t bytes_transferred);
  void processMessage(const std::vector<char>& data);
  void processConfigSync(const OobMessage& msg);

  asio::io_context io_context_;
  asio::ip::tcp::acceptor acceptor_;
  std::thread io_thread_;
  std::atomic<bool> running_{false};
  
  std::unordered_map<OobMsgType, std::function<void(const OobMessage&)>> handlers_;
  std::shared_mutex handlers_mutex_;
  
  std::queue<OobMessage> message_queue_;
  std::mutex queue_mutex_;
  
  std::string local_ip_;
  uint16_t local_port_;
};

} // namespace pccl::communicator
