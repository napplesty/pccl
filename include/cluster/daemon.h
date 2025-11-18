#pragma once

#include <string>
#include <memory>
#include <functional>
#include <thread>
#include <atomic>
#include <unordered_map>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <asio.hpp>

namespace engine_c {

enum class ClusterEventType : int {
  ResizeWorld,
  JoinSelf,
  QuitSelf,
  ReplaceOld,
  BeReplaced,
  JoinNode,
  QuitNode,
  UpdateNode,
};

struct ClusterMessage {
  ClusterEventType event_type;
  std::string data;
  int source_rank;
  int target_rank;
  uint64_t timestamp;

  std::string serialize() const;
  static ClusterMessage deserialize(const std::string& data);
};

class Daemon {
public:
  Daemon();
  ~Daemon();

  bool start(int rank, const std::string& host, int port, bool is_master = false);
  bool stop();
  bool isRunning() const;

  bool connectToMaster(const std::string& master_host, int master_port);
  bool disconnectFromMaster();

  bool sendMessage(const ClusterMessage& message);
  bool broadcastMessage(const ClusterMessage& message);

  void setEventCallback(std::function<void(const ClusterMessage&)> callback);

  int getRank() const { return rank_; }
  bool isMaster() const { return is_master_; }

private:
  void startAccept();
  void startReceive();
  void handleAccept(std::shared_ptr<asio::ip::tcp::socket> socket);
  void handleReceive(std::shared_ptr<asio::ip::tcp::socket> socket, 
                     std::shared_ptr<std::vector<char>> buffer);
  void handleMasterConnection();

  void processMessage(const ClusterMessage& message);
  void sendMessageToSocket(std::shared_ptr<asio::ip::tcp::socket> socket, 
                           const ClusterMessage& message);

  void workerThread();

  int rank_;
  bool is_master_;
  std::string host_;
  int port_;

  std::unique_ptr<asio::io_context> io_context_;
  std::unique_ptr<asio::ip::tcp::acceptor> acceptor_;
  std::shared_ptr<asio::ip::tcp::socket> master_socket_;
  std::unique_ptr<std::thread> io_thread_;
  std::unique_ptr<std::thread> worker_thread_;

  std::unordered_map<int, std::shared_ptr<asio::ip::tcp::socket>> slave_sockets_;
  std::mutex sockets_mutex_;

  std::queue<ClusterMessage> message_queue_;
  std::mutex queue_mutex_;
  std::condition_variable queue_cv_;

  std::function<void(const ClusterMessage&)> event_callback_;

  std::atomic<bool> running_;
  std::atomic<bool> connected_to_master_;
};

}
