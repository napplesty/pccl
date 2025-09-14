#include <sys/types.h>
#include <unordered_map>
#include <memory>
#include <string>
#include <mutex>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <queue>
#include <thread>
#include <vector>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <ifaddrs.h>
#include <unistd.h>
#include <stdexcept>
#include <sys/epoll.h>
#include <fcntl.h>
#include <errno.h>
#include <optional>

namespace pccl {

using GID = std::string;
using Token = std::string;

struct TcpRemoteConnectionMetadata {
  std::string ip;
  uint16_t port;
  TcpRemoteConnectionMetadata() : port(0) {}
  TcpRemoteConnectionMetadata(const std::string& ip, uint16_t port) : ip(ip), port(port) {}
  TcpRemoteConnectionMetadata(const TcpRemoteConnectionMetadata&) = default;
  TcpRemoteConnectionMetadata(TcpRemoteConnectionMetadata&&) noexcept = default;
  TcpRemoteConnectionMetadata& operator=(const TcpRemoteConnectionMetadata&) = default;
  TcpRemoteConnectionMetadata& operator=(TcpRemoteConnectionMetadata&&) noexcept = default;
};

class TcpGIDMap {
public:
  void registerGID(const GID& gid, const std::string& ip, uint16_t port) {
    std::lock_guard<std::mutex> lock(mutex_);
    map_[gid] = {ip, port};
  }
  std::optional<std::pair<std::string, uint16_t>> lookupGID(const GID& gid) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = map_.find(gid);
    if (it != map_.end()) return it->second;
    return std::nullopt;
  }
private:
  std::unordered_map<GID, std::pair<std::string, uint16_t>> map_;
  std::mutex mutex_;
};

class TcpDeviceList {
public:
  TcpDeviceList() {
    struct ifaddrs* ifaddr;
    if (getifaddrs(&ifaddr) == -1) throw std::runtime_error("");
    for (struct ifaddrs* ifa = ifaddr; ifa; ifa = ifa->ifa_next) {
      if (!ifa->ifa_addr) continue;
      if (ifa->ifa_addr->sa_family == AF_INET) {
        struct sockaddr_in* sin = reinterpret_cast<struct sockaddr_in*>(ifa->ifa_addr);
        char buf[INET_ADDRSTRLEN];
        if (inet_ntop(AF_INET, &sin->sin_addr, buf, sizeof(buf))) devices_.emplace_back(buf);
      }
    }
    freeifaddrs(ifaddr);
  }
  int count() const { return static_cast<int>(devices_.size()); }
  std::string operator[](int index) const {
    if (index < 0 || index >= count()) throw std::runtime_error("");
    return devices_[index];
  }
private:
  std::vector<std::string> devices_;
};

class TcpContext {
public:
  explicit TcpContext(const std::string& local_ip) : local_ip_(local_ip) {}
  const std::string& getLocalIp() const { return local_ip_; }
private:
  std::string local_ip_;
};

class TcpMemoryRegion {
public:
  TcpMemoryRegion(void* addr, size_t length, uint32_t lkey, uint32_t rkey)
    : addr_(addr), length_(length), lkey_(lkey), rkey_(rkey) {}
  void* getAddr() const { return addr_; }
  size_t getLength() const { return length_; }
  uint32_t getLkey() const { return lkey_; }
  uint32_t getRkey() const { return rkey_; }
private:
  void* addr_;
  size_t length_;
  uint32_t lkey_;
  uint32_t rkey_;
};

class TcpProtectionDomain {
public:
  std::shared_ptr<TcpMemoryRegion> registerMR(void* addr, size_t length) {
    std::lock_guard<std::mutex> lock(mutex_);
    uint32_t lkey = next_key_++;
    uint32_t rkey = next_key_++;
    auto mr = std::make_shared<TcpMemoryRegion>(addr, length, lkey, rkey);
    mrs_.emplace(lkey, mr);
    mrs_.emplace(rkey, mr);
    return mr;
  }
  void deregisterMR(const std::shared_ptr<TcpMemoryRegion>& mr) {
    std::lock_guard<std::mutex> lock(mutex_);
    mrs_.erase(mr->getLkey());
    mrs_.erase(mr->getRkey());
  }
  std::shared_ptr<TcpMemoryRegion> lookUpMR(uint32_t key) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = mrs_.find(key);
    return it != mrs_.end() ? it->second : nullptr;
  }
private:
  std::unordered_map<uint32_t, std::shared_ptr<TcpMemoryRegion>> mrs_;
  std::mutex mutex_;
  std::atomic<uint32_t> next_key_{1};
};

enum class TcpWCStatus { Success, Failure, ConnectionReset };
enum class TcpWCOpcode { Send, Recv, Write, Read };
enum class TcpQPState { Reset, Init, Rtr, Rts, Err };

struct TcpSGE { void* addr; size_t length; uint32_t lkey; };
struct TcpSendWR { TcpSendWR* next; TcpSGE* sg_list; int num_sge; };
struct TcpRecvWR { TcpRecvWR* next; TcpSGE* sg_list; int num_sge; };
struct TcpWC { TcpWCStatus status; TcpWCOpcode opcode; size_t byte_len; };

struct TcpRemoteMRMetadata {
  GID remote_gid;
  uint64_t remote_addr;
  uint32_t remote_rkey;
};

struct TcpCtrlHeader {
  enum class Op { Token, Data, Write, Read };
  Op op;
  Token token;
  GID gid;
  uint64_t remote_addr;
  uint32_t remote_rkey;
  size_t length;
};

struct TcpWriteWR {
  TcpWriteWR* next;
  TcpSGE* sg_list;
  int num_sge;
  TcpRemoteMRMetadata remote_mr;
};

struct TcpReadWR {
  TcpReadWR* next;
  TcpSGE* sg_list;
  int num_sge;
  TcpRemoteMRMetadata remote_mr;
};

struct SendTaskContext {
  std::vector<TcpSGE> sg_list;
  size_t curr_sge_idx = 0;
  size_t curr_sge_offset = 0;
  size_t total_bytes = 0;
  explicit SendTaskContext(const TcpSendWR* wr) {
    if (!wr) return;
    for (const auto* curr_wr = wr; curr_wr; curr_wr = curr_wr->next) {
      if (!curr_wr->sg_list) continue;
      for (int i = 0; i < curr_wr->num_sge; ++i) {
        sg_list.push_back(curr_wr->sg_list[i]);
        total_bytes += curr_wr->sg_list[i].length;
      }
    }
  }
  bool is_completed() const { return curr_sge_idx >= sg_list.size(); }
  std::pair<void*, size_t> get_current_data() const {
    if (is_completed()) return {nullptr, 0};
    const auto& sge = sg_list[curr_sge_idx];
    return {static_cast<char*>(sge.addr) + curr_sge_offset, sge.length - curr_sge_offset};
  }
  bool update_sent(size_t bytes) {
    if (is_completed()) return false;
    curr_sge_offset += bytes;
    if (curr_sge_offset >= sg_list[curr_sge_idx].length) {
      curr_sge_idx++;
      curr_sge_offset = 0;
      return true;
    }
    return false;
  }
};

struct RecvTaskContext {
  std::vector<TcpSGE> sg_list;
  size_t curr_sge_idx = 0;
  size_t curr_sge_offset = 0;
  size_t total_bytes = 0;
  explicit RecvTaskContext(const TcpRecvWR* wr) {
    if (!wr) return;
    for (const auto* curr_wr = wr; curr_wr; curr_wr = curr_wr->next) {
      if (!curr_wr->sg_list) continue;
      for (int i = 0; i < curr_wr->num_sge; ++i) {
        sg_list.push_back(curr_wr->sg_list[i]);
        total_bytes += curr_wr->sg_list[i].length;
      }
    }
  }
  bool is_completed() const { return curr_sge_idx >= sg_list.size(); }
  std::pair<void*, size_t> get_current_buffer() const {
    if (is_completed()) return {nullptr, 0};
    const auto& sge = sg_list[curr_sge_idx];
    return {static_cast<char*>(sge.addr) + curr_sge_offset, sge.length - curr_sge_offset};
  }
  bool update_received(size_t bytes) {
    if (is_completed()) return false;
    curr_sge_offset += bytes;
    if (curr_sge_offset >= sg_list[curr_sge_idx].length) {
      curr_sge_idx++;
      curr_sge_offset = 0;
      return true;
    }
    return false;
  }
};

struct WriteTaskContext {
  std::vector<TcpSGE> sg_list;
  TcpRemoteMRMetadata remote_mr;
  size_t curr_sge_idx = 0;
  size_t curr_sge_offset = 0;
  size_t total_bytes = 0;
  bool ctrl_sent = false;
  explicit WriteTaskContext(const TcpWriteWR* wr) {
    if (!wr) return;
    remote_mr = wr->remote_mr;
    for (const auto* curr_wr = wr; curr_wr; curr_wr = curr_wr->next) {
      if (!curr_wr->sg_list) continue;
      for (int i = 0; i < curr_wr->num_sge; ++i) {
        sg_list.push_back(curr_wr->sg_list[i]);
        total_bytes += curr_wr->sg_list[i].length;
      }
    }
  }
  bool is_completed() const { return curr_sge_idx >= sg_list.size() && ctrl_sent; }
  std::vector<char> get_ctrl_header(const Token& token) const {
    TcpCtrlHeader header;
    header.op = TcpCtrlHeader::Op::Write;
    header.token = token;
    header.gid = remote_mr.remote_gid;
    header.remote_addr = remote_mr.remote_addr;
    header.remote_rkey = remote_mr.remote_rkey;
    header.length = total_bytes;
    std::vector<char> buf(sizeof(header));
    memcpy(buf.data(), &header, sizeof(header));
    return buf;
  }
  std::pair<void*, size_t> get_current_data() const {
    if (is_completed()) return {nullptr, 0};
    const auto& sge = sg_list[curr_sge_idx];
    return {static_cast<char*>(sge.addr) + curr_sge_offset, sge.length - curr_sge_offset};
  }
  bool update_sent(size_t bytes) {
    if (is_completed()) return false;
    curr_sge_offset += bytes;
    if (curr_sge_offset >= sg_list[curr_sge_idx].length) {
      curr_sge_idx++;
      curr_sge_offset = 0;
      return true;
    }
    return false;
  }
};

struct ReadTaskContext {
  std::vector<TcpSGE> sg_list;
  TcpRemoteMRMetadata remote_mr;
  size_t curr_sge_idx = 0;
  size_t curr_sge_offset = 0;
  size_t total_bytes = 0;
  bool ctrl_sent = false;
  bool data_received = false;
  explicit ReadTaskContext(const TcpReadWR* wr) {
    if (!wr) return;
    remote_mr = wr->remote_mr;
    for (const auto* curr_wr = wr; curr_wr; curr_wr = curr_wr->next) {
      if (!curr_wr->sg_list) continue;
      for (int i = 0; i < curr_wr->num_sge; ++i) {
        sg_list.push_back(curr_wr->sg_list[i]);
        total_bytes += curr_wr->sg_list[i].length;
      }
    }
  }
  bool is_completed() const { return curr_sge_idx >= sg_list.size() && ctrl_sent && data_received; }
  std::vector<char> get_ctrl_header(const Token& token) const {
    TcpCtrlHeader header;
    header.op = TcpCtrlHeader::Op::Read;
    header.token = token;
    header.gid = remote_mr.remote_gid;
    header.remote_addr = remote_mr.remote_addr;
    header.remote_rkey = remote_mr.remote_rkey;
    header.length = total_bytes;
    std::vector<char> buf(sizeof(header));
    memcpy(buf.data(), &header, sizeof(header));
    return buf;
  }
  std::pair<void*, size_t> get_current_buffer() const {
    if (is_completed()) return {nullptr, 0};
    const auto& sge = sg_list[curr_sge_idx];
    return {static_cast<char*>(sge.addr) + curr_sge_offset, sge.length - curr_sge_offset};
  }
  bool update_received(size_t bytes) {
    if (is_completed()) return false;
    curr_sge_offset += bytes;
    if (curr_sge_offset >= sg_list[curr_sge_idx].length) {
      curr_sge_idx++;
      curr_sge_offset = 0;
      return true;
    }
    return false;
  }
};

class TcpQueuePair {
public:
  TcpQueuePair(const Token& token) : sock_fd_(-1), state_(TcpQPState::Reset), token_(token) {}
  ~TcpQueuePair() { if (sock_fd_ != -1) close(sock_fd_); }
  bool modifyToInit() {
    if (state_ != TcpQPState::Reset) return false;
    sock_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (sock_fd_ == -1) return false;
    state_ = TcpQPState::Init;
    return true;
  }
  bool modifyToRTR(const GID& gid, const TcpGIDMap& gid_map) {
    if (state_ != TcpQPState::Init) return false;
    auto addr = gid_map.lookupGID(gid);
    if (!addr) return false;
    struct sockaddr_in serv_addr{};
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(addr->second);
    if (inet_pton(AF_INET, addr->first.c_str(), &serv_addr.sin_addr) != 1) return false;
    if (connect(sock_fd_, reinterpret_cast<struct sockaddr*>(&serv_addr), sizeof(serv_addr)) == -1) {
      if (errno != EINPROGRESS) return false;
    }
    TcpCtrlHeader header{.op = TcpCtrlHeader::Op::Token, .token = token_, .gid = gid};
    std::vector<char> ctrl_buf(sizeof(header));
    memcpy(ctrl_buf.data(), &header, sizeof(header));
    if (send(sock_fd_, ctrl_buf.data(), ctrl_buf.size(), MSG_NOSIGNAL) != ctrl_buf.size()) return false;
    char resp[sizeof(TcpCtrlHeader)];
    ssize_t recvd = recv(sock_fd_, resp, sizeof(resp), MSG_WAITALL);
    if (recvd != sizeof(resp)) return false;
    TcpCtrlHeader resp_header;
    memcpy(&resp_header, resp, sizeof(resp_header));
    if (resp_header.op != TcpCtrlHeader::Op::Token || resp_header.token != token_) return false;
    state_ = TcpQPState::Rtr;
    return true;
  }
  bool modifyToRTS() {
    if (state_ != TcpQPState::Rtr) return false;
    int flags = fcntl(sock_fd_, F_GETFL, 0);
    if (flags == -1 || fcntl(sock_fd_, F_SETFL, flags | O_NONBLOCK) == -1) {
      state_ = TcpQPState::Err;
      return false;
    }
    state_ = TcpQPState::Rts;
    return true;
  }
  void postSend(const TcpSendWR* wr) {
    if (state_ != TcpQPState::Rts) throw std::runtime_error("");
    std::lock_guard<std::mutex> lock(send_mutex_);
    send_queue_.emplace(wr);
  }
  void postRecv(const TcpRecvWR* wr) {
    if (state_ != TcpQPState::Rtr) throw std::runtime_error("");
    std::lock_guard<std::mutex> lock(recv_mutex_);
    recv_queue_.emplace(wr);
  }
  void postWrite(const TcpWriteWR* wr) {
    if (state_ != TcpQPState::Rts) throw std::runtime_error("");
    std::lock_guard<std::mutex> lock(write_mutex_);
    write_queue_.emplace(wr);
  }
  void postRead(const TcpReadWR* wr) {
    if (state_ != TcpQPState::Rts) throw std::runtime_error("");
    std::lock_guard<std::mutex> lock(read_mutex_);
    read_queue_.emplace(wr);
  }
  int getSockFd() const { return sock_fd_; }
  TcpQPState getState() const { return state_; }
  void setState(TcpQPState state) { state_ = state; }
  std::queue<SendTaskContext>& getSendQueue() { return send_queue_; }
  std::mutex& getSendMutex() { return send_mutex_; }
  std::queue<RecvTaskContext>& getRecvQueue() { return recv_queue_; }
  std::mutex& getRecvMutex() { return recv_mutex_; }
  std::queue<WriteTaskContext>& getWriteQueue() { return write_queue_; }
  std::mutex& getWriteMutex() { return write_mutex_; }
  std::queue<ReadTaskContext>& getReadQueue() { return read_queue_; }
  std::mutex& getReadMutex() { return read_mutex_; }
  const Token& getToken() const { return token_; }
private:
  int sock_fd_;
  TcpQPState state_;
  Token token_;
  std::queue<SendTaskContext> send_queue_;
  std::mutex send_mutex_;
  std::queue<RecvTaskContext> recv_queue_;
  std::mutex recv_mutex_;
  std::queue<WriteTaskContext> write_queue_;
  std::mutex write_mutex_;
  std::queue<ReadTaskContext> read_queue_;
  std::mutex read_mutex_;
};

class TcpManager {
public:
  using ConnectionId = uint64_t;
  using QPId = uint64_t;
  struct ConnectionConfig { uint16_t local_port = 51113; int cq_size = 100; };
  struct QPConfig { int max_send_wr = 100; int max_recv_wr = 100; int max_send_sge = 1; int max_recv_sge = 1; };

  TcpManager() : epoll_fd_(-1), wakeup_pipe_{-1, -1}, listen_fd_(-1),
                 initialized_(false), running_(false), next_conn_id_(1), next_qp_id_(1) {}
  ~TcpManager() {
    running_ = false;
    if (wakeup_pipe_[1] != -1) write(wakeup_pipe_[1], "", 1);
    if (worker_thread_.joinable()) worker_thread_.join();
    if (epoll_fd_ != -1) close(epoll_fd_);
    if (wakeup_pipe_[0] != -1) close(wakeup_pipe_[0]);
    if (wakeup_pipe_[1] != -1) close(wakeup_pipe_[1]);
    if (listen_fd_ != -1) close(listen_fd_);
  }

  bool initialize(const std::string& local_ip = "", const Token& token = "default_token") {
    if (initialized_) return true;
    try {
      TcpDeviceList devices;
      local_ip_ = local_ip.empty() ? (devices.count() ? devices[0] : "") : local_ip;
      if (local_ip_.empty()) return false;
      context_ = std::make_shared<TcpContext>(local_ip_);
      pd_ = std::make_shared<TcpProtectionDomain>();
      token_ = token;
      epoll_fd_ = epoll_create1(EPOLL_CLOEXEC);
      if (epoll_fd_ == -1) throw std::runtime_error("");
      if (pipe2(wakeup_pipe_, O_NONBLOCK | O_CLOEXEC) == -1) throw std::runtime_error("");
      epoll_event wakeup_ev{.events = EPOLLIN, .data.fd = wakeup_pipe_[0]};
      if (epoll_ctl(epoll_fd_, EPOLL_CTL_ADD, wakeup_pipe_[0], &wakeup_ev) == -1) throw std::runtime_error("");
      listen_fd_ = socket(AF_INET, SOCK_STREAM, 0);
      if (listen_fd_ == -1) throw std::runtime_error("");
      int opt = 1;
      setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
      struct sockaddr_in addr{};
      addr.sin_family = AF_INET;
      addr.sin_port = htons(ConnectionConfig{}.local_port);
      inet_pton(AF_INET, local_ip_.c_str(), &addr.sin_addr);
      if (bind(listen_fd_, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) == -1) throw std::runtime_error("");
      if (listen(listen_fd_, SOMAXCONN) == -1) throw std::runtime_error("");
      epoll_event listen_ev{.events = EPOLLIN, .data.fd = listen_fd_};
      if (epoll_ctl(epoll_fd_, EPOLL_CTL_ADD, listen_fd_, &listen_ev) == -1) throw std::runtime_error("");
      running_ = true;
      worker_thread_ = std::thread(&TcpManager::workerLoop, this);
      initialized_ = true;
      return true;
    } catch (...) {
      cleanup();
      return false;
    }
  }

  void registerGID(const GID& gid, const std::string& ip, uint16_t port) {
    gid_map_.registerGID(gid, ip, port);
  }

  ConnectionId createConnection(const ConnectionConfig& config = ConnectionConfig{}) {
    if (!initialized_) return 0;
    ConnectionId id = next_conn_id_++;
    std::lock_guard<std::mutex> lock(conn_mutex_);
    connections_[id] = {config, std::make_shared<TcpCompletionQueue>(), {}};
    return id;
  }

  QPId createQP(ConnectionId conn_id, const QPConfig& config = QPConfig{}) {
    if (!initialized_) return 0;
    std::lock_guard<std::mutex> lock(conn_mutex_);
    auto& conn = connections_[conn_id];
    QPId id = next_qp_id_++;
    conn.qps[id] = std::make_shared<TcpQueuePair>(token_);
    return id;
  }

  bool modifyQPToInit(ConnectionId conn_id, QPId qp_id) {
    auto qp = getQP(conn_id, qp_id);
    return qp && qp->modifyToInit();
  }

  bool modifyQPToRTR(ConnectionId conn_id, QPId qp_id, const GID& gid) {
    auto qp = getQP(conn_id, qp_id);
    return qp && qp->modifyToRTR(gid, gid_map_);
  }

  bool modifyQPToRTS(ConnectionId conn_id, QPId qp_id) {
    auto qp = getQP(conn_id, qp_id);
    if (!qp || !qp->modifyToRTS()) return false;
    int fd = qp->getSockFd();
    epoll_event ev{.events = EPOLLIN | EPOLLOUT | EPOLLET, .data.fd = fd};
    if (epoll_ctl(epoll_fd_, EPOLL_CTL_ADD, fd, &ev) == -1) return false;
    std::lock_guard<std::mutex> lock(fd_map_mutex_);
    fd_map_[fd] = {conn_id, qp_id};
    return true;
  }

  bool postSend(ConnectionId conn_id, QPId qp_id, const TcpSendWR* wr) {
    auto qp = getQP(conn_id, qp_id);
    if (!qp) return false;
    qp->postSend(wr);
    wakeupWorker();
    return true;
  }

  bool postRecv(ConnectionId conn_id, QPId qp_id, const TcpRecvWR* wr) {
    auto qp = getQP(conn_id, qp_id);
    if (!qp) return false;
    qp->postRecv(wr);
    wakeupWorker();
    return true;
  }

  bool postWrite(ConnectionId conn_id, QPId qp_id, const TcpWriteWR* wr) {
    auto qp = getQP(conn_id, qp_id);
    if (!qp) return false;
    qp->postWrite(wr);
    wakeupWorker();
    return true;
  }

  bool postRead(ConnectionId conn_id, QPId qp_id, const TcpReadWR* wr) {
    auto qp = getQP(conn_id, qp_id);
    if (!qp) return false;
    qp->postRead(wr);
    wakeupWorker();
    return true;
  }

  int pollCQ(ConnectionId conn_id, int num_entries, TcpWC* wc) {
    std::lock_guard<std::mutex> lock(conn_mutex_);
    auto it = connections_.find(conn_id);
    return it != connections_.end() ? it->second.cq->poll(wc, num_entries) : -1;
  }

private:
  struct ConnectionInfo {
    ConnectionConfig config;
    std::shared_ptr<TcpCompletionQueue> cq;
    std::unordered_map<QPId, std::shared_ptr<TcpQueuePair>> qps;
  };

  struct TcpCompletionQueue {
    void add(const TcpWC& wc) {
      std::lock_guard<std::mutex> lock(mutex_);
      queue_.push(wc);
    }
    int poll(TcpWC* wc, int num) {
      std::lock_guard<std::mutex> lock(mutex_);
      int count = 0;
      while (count < num && !queue_.empty()) {
        *wc++ = queue_.front();
        queue_.pop();
        count++;
      }
      return count;
    }
    std::queue<TcpWC> queue_;
    std::mutex mutex_;
  };

  void workerLoop() {
    constexpr int MAX_EVENTS = 1024;
    epoll_event events[MAX_EVENTS];
    while (running_) {
      int n = epoll_wait(epoll_fd_, events, MAX_EVENTS, -1);
      if (n == -1) {
        if (errno == EINTR) continue;
        break;
      }
      for (int i = 0; i < n; ++i) {
        if (events[i].data.fd == wakeup_pipe_[0]) {
          char buf[16];
          while (read(wakeup_pipe_[0], buf, sizeof(buf)) > 0);
          continue;
        }
        if (events[i].data.fd == listen_fd_) {
          handleNewConnection();
          continue;
        }
        handleSocketEvent(events[i].data.fd, events[i].events);
      }
    }
  }

  void handleNewConnection() {
    struct sockaddr_in client_addr{};
    socklen_t addr_len = sizeof(client_addr);
    int client_fd = accept4(listen_fd_, reinterpret_cast<struct sockaddr*>(&client_addr), &addr_len, SOCK_NONBLOCK | SOCK_CLOEXEC);
    if (client_fd == -1) return;
    char client_ip[INET_ADDRSTRLEN];
    inet_ntop(AF_INET, &client_addr.sin_addr, client_ip, sizeof(client_ip));
    TcpCtrlHeader header;
    std::vector<char> buf(sizeof(header));
    ssize_t recvd = recv(client_fd, buf.data(), buf.size(), MSG_WAITALL);
    if (recvd != sizeof(header)) {
      close(client_fd);
      return;
    }
    memcpy(&header, buf.data(), sizeof(header));
    if (header.op != TcpCtrlHeader::Op::Token || header.token != token_) {
      close(client_fd);
      return;
    }
    auto qp = std::make_shared<TcpQueuePair>(token_);
    if (!qp->modifyToInit() || !qp->modifyToRTR(header.gid, gid_map_) || !qp->modifyToRTS()) {
      close(client_fd);
      return;
    }
    ConnectionId conn_id = next_conn_id_++;
    QPId qp_id = next_qp_id_++;
    std::lock_guard<std::mutex> lock(conn_mutex_);
    connections_[conn_id] = {ConnectionConfig{}, std::make_shared<TcpCompletionQueue>(), {{qp_id, qp}}};
    epoll_event ev{.events = EPOLLIN | EPOLLOUT | EPOLLET, .data.fd = client_fd};
    epoll_ctl(epoll_fd_, EPOLL_CTL_ADD, client_fd, &ev);
    std::lock_guard<std::mutex> fd_lock(fd_map_mutex_);
    fd_map_[client_fd] = {conn_id, qp_id};
    TcpCtrlHeader resp{.op = TcpCtrlHeader::Op::Token, .token = token_, .gid = header.gid};
    std::vector<char> resp_buf(sizeof(resp));
    memcpy(resp_buf.data(), &resp, sizeof(resp));
    send(client_fd, resp_buf.data(), resp_buf.size(), MSG_NOSIGNAL);
  }

  void handleSocketEvent(int fd, uint32_t events) {
    std::pair<ConnectionId, QPId> ids;
    {
      std::lock_guard<std::mutex> lock(fd_map_mutex_);
      auto it = fd_map_.find(fd);
      if (it == fd_map_.end()) return;
      ids = it->second;
    }
    auto [conn_id, qp_id] = ids;
    auto qp = getQP(conn_id, qp_id);
    auto cq = getCQ(conn_id);
    if (!qp || !cq) return;
    if (events & (EPOLLERR | EPOLLHUP)) {
      cq->add({TcpWCStatus::ConnectionReset, TcpWCOpcode::Recv, 0});
      close(fd);
      std::lock_guard<std::mutex> lock(fd_map_mutex_);
      fd_map_.erase(fd);
      return;
    }
    if (events & EPOLLOUT) handleWriteEvent(fd, qp, cq);
    if (events & EPOLLIN) handleReadEvent(fd, qp, cq);
  }

  void handleWriteEvent(int fd, std::shared_ptr<TcpQueuePair> qp, std::shared_ptr<TcpCompletionQueue> cq) {
    std::lock_guard<std::mutex> send_lock(qp->getSendMutex());
    auto& send_queue = qp->getSendQueue();
    while (!send_queue.empty()) {
      auto& task = send_queue.front();
      if (task.is_completed()) { send_queue.pop(); continue; }
      auto [data, len] = task.get_current_data();
      ssize_t sent = send(fd, data, len, MSG_NOSIGNAL);
      if (sent <= 0) break;
      task.update_sent(sent);
      if (task.is_completed()) {
        cq->add({TcpWCStatus::Success, TcpWCOpcode::Send, task.total_bytes});
        send_queue.pop();
      }
    }
    std::lock_guard<std::mutex> write_lock(qp->getWriteMutex());
    auto& write_queue = qp->getWriteQueue();
    while (!write_queue.empty()) {
      auto& task = write_queue.front();
      if (task.is_completed()) { write_queue.pop(); continue; }
      if (!task.ctrl_sent) {
        auto ctrl = task.get_ctrl_header(token_);
        if (send(fd, ctrl.data(), ctrl.size(), MSG_NOSIGNAL) != ctrl.size()) break;
        task.ctrl_sent = true;
      }
      auto [data, len] = task.get_current_data();
      ssize_t sent = send(fd, data, len, MSG_NOSIGNAL);
      if (sent <= 0) break;
      task.update_sent(sent);
      if (task.is_completed()) {
        cq->add({TcpWCStatus::Success, TcpWCOpcode::Write, task.total_bytes});
        write_queue.pop();
      }
    }
    std::lock_guard<std::mutex> read_lock(qp->getReadMutex());
    auto& read_queue = qp->getReadQueue();
    while (!read_queue.empty()) {
      auto& task = read_queue.front();
      if (task.is_completed() || task.ctrl_sent) continue;
      auto ctrl = task.get_ctrl_header(token_);
      if (send(fd, ctrl.data(), ctrl.size(), MSG_NOSIGNAL) != ctrl.size()) break;
      task.ctrl_sent = true;
    }
  }

  void handleReadEvent(int fd, std::shared_ptr<TcpQueuePair> qp, std::shared_ptr<TcpCompletionQueue> cq) {
    std::lock_guard<std::mutex> recv_lock(qp->getRecvMutex());
    auto& recv_queue = qp->getRecvQueue();
    while (!recv_queue.empty()) {
      auto& task = recv_queue.front();
      if (task.is_completed()) { recv_queue.pop(); continue; }
      auto [buf, len] = task.get_current_buffer();
      ssize_t recvd = recv(fd, buf, len, MSG_NOSIGNAL);
      if (recvd <= 0) break;
      task.update_received(recvd);
      if (task.is_completed()) {
        cq->add({TcpWCStatus::Success, TcpWCOpcode::Recv, task.total_bytes});
        recv_queue.pop();
      }
    }
    TcpCtrlHeader header;
    std::vector<char> ctrl_buf(sizeof(header));
    ssize_t recvd = recv(fd, ctrl_buf.data(), ctrl_buf.size(), MSG_PEEK);
    if (recvd != sizeof(header)) return;
    memcpy(&header, ctrl_buf.data(), sizeof(header));
    if (header.op == TcpCtrlHeader::Op::Write) {
      handleRemoteWrite(fd, qp, cq, header);
    } else if (header.op == TcpCtrlHeader::Op::Read) {
      handleRemoteRead(fd, qp, cq, header);
    }
  }

  void handleRemoteWrite(int fd, std::shared_ptr<TcpQueuePair> qp, std::shared_ptr<TcpCompletionQueue> cq, const TcpCtrlHeader& header) {
    auto mr = pd_->lookUpMR(header.remote_rkey);
    if (!mr) {
      cq->add({TcpWCStatus::Failure, TcpWCOpcode::Write, 0});
      return;
    }
    uint64_t mr_addr = reinterpret_cast<uint64_t>(mr->getAddr());
    if (header.remote_addr < mr_addr || header.remote_addr + header.length > mr_addr + mr->getLength()) {
      cq->add({TcpWCStatus::Failure, TcpWCOpcode::Write, 0});
      return;
    }
    std::vector<char> buf(sizeof(header));
    recv(fd, buf.data(), buf.size(), MSG_WAITALL);
    void* data = reinterpret_cast<void*>(header.remote_addr);
    size_t remaining = header.length;
    while (remaining > 0) {
      ssize_t recvd = recv(fd, data, remaining, MSG_NOSIGNAL);
      if (recvd <= 0) break;
      data = static_cast<char*>(data) + recvd;
      remaining -= recvd;
    }
    if (remaining == 0) {
      cq->add({TcpWCStatus::Success, TcpWCOpcode::Write, header.length});
    }
  }

  void handleRemoteRead(int fd, std::shared_ptr<TcpQueuePair> qp, std::shared_ptr<TcpCompletionQueue> cq, const TcpCtrlHeader& header) {
    auto mr = pd_->lookUpMR(header.remote_rkey);
    if (!mr) {
      cq->add({TcpWCStatus::Failure, TcpWCOpcode::Read, 0});
      return;
    }
    uint64_t mr_addr = reinterpret_cast<uint64_t>(mr->getAddr());
    if (header.remote_addr < mr_addr || header.remote_addr + header.length > mr_addr + mr->getLength()) {
      cq->add({TcpWCStatus::Failure, TcpWCOpcode::Read, 0});
      return;
    }
    std::vector<char> buf(sizeof(header));
    recv(fd, buf.data(), buf.size(), MSG_WAITALL);
    void* data = reinterpret_cast<void*>(header.remote_addr);
    size_t remaining = header.length;
    while (remaining > 0) {
      ssize_t sent = send(fd, data, remaining, MSG_NOSIGNAL);
      if (sent <= 0) break;
      data = static_cast<char*>(data) + sent;
      remaining -= sent;
    }
    if (remaining == 0) {
      cq->add({TcpWCStatus::Success, TcpWCOpcode::Read, header.length});
    }
  }

  std::shared_ptr<TcpQueuePair> getQP(ConnectionId conn_id, QPId qp_id) {
    std::lock_guard<std::mutex> lock(conn_mutex_);
    auto it = connections_.find(conn_id);
    if (it == connections_.end()) return nullptr;
    auto qp_it = it->second.qps.find(qp_id);
    return qp_it != it->second.qps.end() ? qp_it->second : nullptr;
  }

  std::shared_ptr<TcpCompletionQueue> getCQ(ConnectionId conn_id) {
    std::lock_guard<std::mutex> lock(conn_mutex_);
    auto it = connections_.find(conn_id);
    return it != connections_.end() ? it->second.cq : nullptr;
  }

  void wakeupWorker() {
    char buf[] = "w";
    write(wakeup_pipe_[1], buf, sizeof(buf));
  }

  void cleanup() {
    if (epoll_fd_ != -1) close(epoll_fd_);
    if (wakeup_pipe_[0] != -1) close(wakeup_pipe_[0]);
    if (wakeup_pipe_[1] != -1) close(wakeup_pipe_[1]);
    if (listen_fd_ != -1) close(listen_fd_);
    initialized_ = false;
  }

private:
  std::string local_ip_;
  Token token_;
  std::shared_ptr<TcpContext> context_;
  std::shared_ptr<TcpProtectionDomain> pd_;
  TcpGIDMap gid_map_;
  std::unordered_map<ConnectionId, ConnectionInfo> connections_;
  std::mutex conn_mutex_;
  std::unordered_map<int, std::pair<ConnectionId, QPId>> fd_map_;
  std::mutex fd_map_mutex_;
  int epoll_fd_;
  int wakeup_pipe_[2];
  int listen_fd_;
  bool initialized_;
  std::atomic<bool> running_;
  std::thread worker_thread_;
  std::atomic<ConnectionId> next_conn_id_;
  std::atomic<QPId> next_qp_id_;
};

} // namespace pccl