#pragma once

#include <runtime/communicator/channel.h>
#include <asio.hpp>
#include <asio/ts/buffer.hpp>
#include <asio/ts/internet.hpp>
#include <queue>
#include <thread>
#include <shared_mutex>
#include <condition_variable>

namespace pccl::communicator {

class AsioOobChannel : public OobChannel {
public:
  AsioOobChannel();
  ~AsioOobChannel();

  bool init(const Endpoint& self_endpoint) override;
  void shutdown() override;
  
  bool send(const OobMessage& msg, const Endpoint& dst) override;
  bool broadcast(const OobMessage& msg, const std::vector<Endpoint>& targets) override;
  bool poll(OobMessage* msg, uint32_t timeout_ms) override;
  bool registerHandler(OobMsgType type, std::function<void(const OobMessage&)> handler) override;
  std::vector<Endpoint> getConnectedNodes() const override;
  bool isConnected(const Endpoint& endpoint) const override;

private:
  struct Connection {
    std::shared_ptr<asio::ip::tcp::socket> socket_;
    asio::ip::tcp::endpoint remote_endpoint_;
    std::queue<std::vector<uint8_t>> send_queue_;
    std::mutex send_queue_mutex_;
    std::vector<uint8_t> recv_buffer_;
    bool connected_;
    std::chrono::steady_clock::time_point last_activity_;
  };

  void startAccept();
  void handleAccept(std::shared_ptr<asio::ip::tcp::socket> socket, 
                   const asio::error_code& error);
  void startReceive(std::shared_ptr<Connection> conn);
  void handleReceive(std::shared_ptr<Connection> conn, 
                    const asio::error_code& error, 
                    size_t bytes_transferred);
  void startSend(std::shared_ptr<Connection> conn);
  void handleSend(std::shared_ptr<Connection> conn, 
                 const asio::error_code& error, 
                 size_t bytes_transferred);
  
  bool connectToEndpoint(const Endpoint& endpoint);
  void handleMessage(const OobMessage& msg);
  void cleanupStaleConnections();

  asio::io_context io_context_;
  asio::ip::tcp::acceptor acceptor_{io_context_};
  std::unique_ptr<std::thread> io_thread_;
  std::atomic<bool> running_;
  
  Endpoint self_endpoint_;
  std::unordered_map<std::string, std::shared_ptr<Connection>> connections_;
  mutable std::shared_mutex connections_mutex_;
  
  std::queue<OobMessage> message_queue_;
  mutable std::mutex queue_mutex_;
  std::condition_variable queue_cv_;
  
  std::unordered_map<OobMsgType, std::function<void(const OobMessage&)>> handlers_;
  mutable std::mutex handlers_mutex_;
  
  std::string endpointToKey(const Endpoint& endpoint) const;
  std::string endpointToKey(const std::string& ip, uint16_t port) const;
  asio::ip::tcp::endpoint endpointToTcpEndpoint(const Endpoint& endpoint) const;
};

} // namespace pccl::communicator
