#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace pccl {

enum class SockOpType : int8_t { READ, WRITE, ATOMIC_ADD, WRITE_WITH_IMM };

enum class SockStatus : int8_t {
  SUCCESS,
  ERROR_GENERAL,
  ERROR_MR_NOT_FOUND,
  ERROR_CONN_FAILED,
  ERROR_TIMEOUT,
  ERROR_INVALID_PARAM,
  ERROR_OUT_OF_MEMORY,
  ERROR_QP_STATE,
  ERROR_SEND_FAILED,
  ERROR_RECV_FAILED,
  ERROR_WR_OVERFLOW
};

enum class SocketType : int8_t { QP_SOCKET, ACCEPT_SOCKET };

struct SockMrInfo {
  uint64_t addr;
};

class SockMr {
 public:
  ~SockMr();

  SockMrInfo getInfo() const;
  void* getBuff() const;
  bool isHost() const;

 private:
  SockMr(void* buff, size_t size, bool isHostMemory);
  void* buff;
  size_t size;
  bool isHostMemory;
  friend class SockCtx;
};

struct SockQpInfo {
  std::string host;
  int port;
  int qpn;
};

struct SockWc {
  uint64_t wr_id;
  SockStatus status;
  uint32_t byte_len;
  uint32_t imm_data;
};

struct SocketInfo {
  SocketType type;
  std::weak_ptr<class SockQp> qp;
  std::atomic<int> usage_count_{0};
  std::chrono::steady_clock::time_point last_active_time_;

  SocketInfo() : type(SocketType::ACCEPT_SOCKET) {
    last_active_time_ = std::chrono::steady_clock::now();
  }

  SocketInfo(SocketType t, std::weak_ptr<class SockQp> q) : type(t), qp(q) {
    last_active_time_ = std::chrono::steady_clock::now();
  }

  SocketInfo(SocketInfo&& other) noexcept
      : type(other.type),
        qp(std::move(other.qp)),
        usage_count_(other.usage_count_.load()),
        last_active_time_(other.last_active_time_) {}

  SocketInfo& operator=(SocketInfo&& other) noexcept {
    if (this != &other) {
      type = other.type;
      qp = std::move(other.qp);
      usage_count_.store(other.usage_count_.load());
      last_active_time_ = other.last_active_time_;
    }
    return *this;
  }

  inline void incrementUsage() {
    usage_count_++;
    last_active_time_ = std::chrono::steady_clock::now();
  }

  inline void resetUsage() { usage_count_ = 0; }
  inline int getUsageCount() const { return usage_count_; }

  inline std::chrono::steady_clock::time_point getLastActiveTime() const {
    return last_active_time_;
  }
};

class SockCtx;

class SockQp {
 public:
  ~SockQp();
  SockStatus connect(const SockQpInfo& remote_info);
  SockStatus stageLoad(const SockMr* mr, const SockMrInfo& info, size_t size, uint64_t wrId,
                       uint64_t srcOffset, uint64_t dstOffset, bool signaled);
  SockStatus stageSend(const SockMr* mr, const SockMrInfo& info, uint32_t size, uint64_t wrId,
                       uint64_t srcOffset, uint64_t dstOffset, bool signaled);
  SockStatus stageAtomicAdd(const SockMr* mr, const SockMrInfo& info, uint64_t wrId,
                            uint64_t dstOffset, uint64_t addVal, bool signaled);
  SockStatus stageSendWithImm(const SockMr* mr, const SockMrInfo& info, uint32_t size,
                              uint64_t wrId, uint64_t srcOffset, uint64_t dstOffset, bool signaled,
                              unsigned int immData);
  SockStatus postSend();
  int pollCq();
  void clearCq(int num);
  void clearCq(uint64_t wrId);
  SockQpInfo& getInfo() { return this->info; }

 private:
  SockStatus getWcStatus(uint64_t wrId) const;
  int getNumCqItems() const;

  void disconnect();
  SockStatus ensureConnected();
  int getSocketFd() const { return socket_fd; }

 protected:
  struct WrInfo {
    SockOpType opcode;
    bool signaled;
    uint64_t wr_id;
    const SockMr* local_mr;
    uint64_t local_offset;
    SockMrInfo remote_mr;
    uint64_t remote_offset;
    size_t size;
    uint64_t add_val;
    unsigned int imm_data;
  };

  SockQp(SockCtx* ctx, int socket_fd, const std::string& host, int port, int max_cq_size,
         int max_wr);
  SockStatus sendRequest(const WrInfo& wr);
  SockStatus handleRequest(const std::vector<char>& request);
  SockStatus handleMessageHeader(const void* header_data);

  SockCtx* ctx;
  SockQpInfo info;
  int socket_fd;
  bool connected;
  std::mutex mutex;
  std::condition_variable cv;

  std::list<WrInfo> pending_wrs;
  std::list<SockWc> wcs;
  std::atomic<int> num_signaled_posted_items;
  std::atomic<int> num_signaled_staged_items;
  std::atomic<int> num_completed_items;

  const int max_cq_size;
  const int max_wr;

  std::vector<char> recv_buffer;
  size_t recv_bytes;
  std::unordered_map<uint32_t, SockMr*> mr_map;

  friend class SockCtx;
};

struct SocketLfuComparator {
  bool operator()(const std::pair<int, SocketInfo>& a, const std::pair<int, SocketInfo>& b) const {
    if (a.second.getUsageCount() == b.second.getUsageCount()) {
      return a.second.getLastActiveTime() < b.second.getLastActiveTime();
    }
    return a.second.getUsageCount() < b.second.getUsageCount();
  }
};

enum class MessageType : int8_t {
  REQUEST,      // Operation request
  RESPONSE,     // Operation response
  COMPLETION,   // Work completion notification
  CONNECT,      // Connection information exchange
  CONNECT_ACK,  // Connection acknowledgment
  ERROR         // Error response
};

struct MessageHeader {
  MessageType type;
  SockOpType op_type;
  SockStatus status;
  bool signaled;
  unsigned int imm_data;
  uint64_t wr_id;
  uint64_t src_addr;
  uint64_t dst_addr;
  uint64_t size;
  uint32_t mr_id;
  uint64_t atomic_val;
};

class SockCtx {
 public:
  SockCtx(const std::string& host, int port, int max_connections = 100);
  ~SockCtx();

  std::shared_ptr<SockQp> createQp(int max_cq_size, int max_wr);
  std::shared_ptr<SockMr> registerMr(void* buff, size_t size, bool isHostMemory);
  SockStatus sendData(int socket_fd, const void* header, size_t header_size, const void* payload,
                      size_t payload_size);
  static std::atomic<uint32_t> next_mr_id;
  static std::atomic<int> next_qp_id;

 private:
  std::shared_ptr<SockQp> findQpByFd(int fd);
  std::shared_ptr<SockQp> findQpById(int qp_id);
  void manageConnections();
  void resetUsageCounts();
  void closeConnection(int fd);
  SockStatus addSocketToEpoll(int fd);
  void updateSocketMapping(int fd, SockQp* qp);
  void updateSocketUsage(int fd);

  void setMaxConnections(int max_connections) { max_connections_ = max_connections; }
  int getConnectionCount() const { return socket_infos.size(); }

 private:
  std::string host;
  int port;
  int listen_socket;
  int epoll_fd;
  bool running;
  int max_connections_;

  std::thread worker_thread;
  std::mutex qps_mutex;

  std::unordered_map<int, SocketInfo> socket_infos;
  std::unordered_map<int, int> qp_id_to_fd;

  std::list<std::shared_ptr<SockQp>> qps;
  std::list<std::shared_ptr<SockMr>> mrs;

  void acceptConnections();
  void processEvents();
  void handleSocketEvent(int fd, uint32_t events);

  friend class SockQp;
};

}  // namespace pccl
