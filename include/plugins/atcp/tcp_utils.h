#pragma once

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
#include <sys/epoll.h>
#include <fcntl.h>
#include <optional>

namespace pccl {

using GID = std::string;
using Token = std::string;

struct TcpRemoteConnectionMetadata {
  std::string ip;
  uint16_t port;
  TcpRemoteConnectionMetadata();
  TcpRemoteConnectionMetadata(const std::string& ip, uint16_t port);
  TcpRemoteConnectionMetadata(const TcpRemoteConnectionMetadata&) = default;
  TcpRemoteConnectionMetadata(TcpRemoteConnectionMetadata&&) noexcept = default;
  TcpRemoteConnectionMetadata& operator=(const TcpRemoteConnectionMetadata&) = default;
  TcpRemoteConnectionMetadata& operator=(TcpRemoteConnectionMetadata&&) noexcept = default;
};

class TcpGIDMap {
public:
  void registerGID(const GID& gid, const std::string& ip, uint16_t port);
  std::optional<std::pair<std::string, uint16_t>> lookupGID(const GID& gid) const;  // 添加const
private:
  std::unordered_map<GID, std::pair<std::string, uint16_t>> map_;
  std::mutex mutex_;
};

class TcpDeviceList {
public:
  TcpDeviceList();
  int count() const;
  std::string operator[](int index) const;
private:
  std::vector<std::string> devices_;
};

class TcpContext {
public:
  explicit TcpContext(const std::string& local_ip);
  const std::string& getLocalIp() const;
private:
  std::string local_ip_;
};

class TcpMemoryRegion {
public:
  TcpMemoryRegion(void* addr, size_t length, uint32_t lkey, uint32_t rkey);
  void* getAddr() const;
  size_t getLength() const;
  uint32_t getLkey() const;
  uint32_t getRkey() const;
private:
  void* addr_;
  size_t length_;
  uint32_t lkey_;
  uint32_t rkey_;
};

class TcpProtectionDomain {
public:
  std::shared_ptr<TcpMemoryRegion> registerMR(void* addr, size_t length);
  void deregisterMR(const std::shared_ptr<TcpMemoryRegion>& mr);
  std::shared_ptr<TcpMemoryRegion> lookUpMR(uint32_t key);
private:
  std::unordered_map<uint32_t, std::shared_ptr<TcpMemoryRegion>> mrs_;
  std::mutex mutex_;
  std::atomic<uint32_t> next_key_{1};
};

enum class TcpWCStatus { Success, Failure, ConnectionReset };
enum class TcpWCOpcode { Send, Recv, Write, Read };
enum class TcpQPState { Reset, Init, Rtr, Rts, Err };

struct TcpSGE { 
  void* addr; 
  size_t length; 
  uint32_t lkey; 
};

struct TcpSendWR { 
  TcpSendWR* next; 
  TcpSGE* sg_list; 
  int num_sge; 
};

struct TcpRecvWR { 
  TcpRecvWR* next; 
  TcpSGE* sg_list; 
  int num_sge; 
};

struct TcpWC { 
  TcpWCStatus status; 
  TcpWCOpcode opcode; 
  size_t byte_len; 
};

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
  
  // 添加默认构造函数和拷贝构造函数
  TcpCtrlHeader() = default;
  TcpCtrlHeader(const TcpCtrlHeader&) = default;
  TcpCtrlHeader& operator=(const TcpCtrlHeader&) = default;
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
  explicit SendTaskContext(const TcpSendWR* wr);
  bool is_completed() const;
  std::pair<void*, size_t> get_current_data() const;
  bool update_sent(size_t bytes);
};

struct RecvTaskContext {
  std::vector<TcpSGE> sg_list;
  size_t curr_sge_idx = 0;
  size_t curr_sge_offset = 0;
  size_t total_bytes = 0;
  explicit RecvTaskContext(const TcpRecvWR* wr);
  bool is_completed() const;
  std::pair<void*, size_t> get_current_buffer() const;
  bool update_received(size_t bytes);
};

struct WriteTaskContext {
  std::vector<TcpSGE> sg_list;
  TcpRemoteMRMetadata remote_mr;
  size_t curr_sge_idx = 0;
  size_t curr_sge_offset = 0;
  size_t total_bytes = 0;
  bool ctrl_sent = false;
  explicit WriteTaskContext(const TcpWriteWR* wr);
  bool is_completed() const;
  std::vector<char> get_ctrl_header(const Token& token) const;
  std::pair<void*, size_t> get_current_data() const;
  bool update_sent(size_t bytes);
};

struct ReadTaskContext {
  std::vector<TcpSGE> sg_list;
  TcpRemoteMRMetadata remote_mr;
  size_t curr_sge_idx = 0;
  size_t curr_sge_offset = 0;
  size_t total_bytes = 0;
  bool ctrl_sent = false;
  bool data_received = false;
  explicit ReadTaskContext(const TcpReadWR* wr);
  bool is_completed() const;
  std::vector<char> get_ctrl_header(const Token& token) const;
  std::pair<void*, size_t> get_current_buffer() const;
  bool update_received(size_t bytes);
};

class TcpQueuePair {
public:
  TcpQueuePair(const Token& token);
  ~TcpQueuePair();
  bool modifyToInit();
  bool modifyToRTR(const GID& gid, const TcpGIDMap& gid_map);
  bool modifyToRTS();
  void postSend(const TcpSendWR* wr);
  void postRecv(const TcpRecvWR* wr);
  void postWrite(const TcpWriteWR* wr);
  void postRead(const TcpReadWR* wr);
  int getSockFd() const;
  TcpQPState getState() const;
  void setState(TcpQPState state);
  std::queue<SendTaskContext>& getSendQueue();
  std::mutex& getSendMutex();
  std::queue<RecvTaskContext>& getRecvQueue();
  std::mutex& getRecvMutex();
  std::queue<WriteTaskContext>& getWriteQueue();
  std::mutex& getWriteMutex();
  std::queue<ReadTaskContext>& getReadQueue();
  std::mutex& getReadMutex();
  const Token& getToken() const;
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
  class TcpCompletionQueue {  // 移动到这里定义
  public:
    void add(const TcpWC& wc);
    int poll(TcpWC* wc, int num);
  private:
    std::queue<TcpWC> queue_;
    std::mutex mutex_;
  };

  using ConnectionId = uint64_t;
  using QPId = uint64_t;
  
  struct ConnectionConfig {
    ConnectionConfig();
    uint16_t local_port = 51113; 
    int cq_size = 100; 
  };
  
  struct QPConfig { 
    QPConfig();
    int max_send_wr = 100; 
    int max_recv_wr = 100; 
    int max_send_sge = 1; 
    int max_recv_sge = 1; 
  };

  struct ConnectionInfo {
    ConnectionConfig config;
    std::shared_ptr<TcpCompletionQueue> cq;
    std::unordered_map<QPId, std::shared_ptr<TcpQueuePair>> qps;
    
    // 添加构造函数
    ConnectionInfo() = default;
    ConnectionInfo(const ConnectionConfig& cfg, std::shared_ptr<TcpCompletionQueue> c, 
                  std::unordered_map<QPId, std::shared_ptr<TcpQueuePair>> q)
      : config(cfg), cq(c), qps(q) {}
  };

  TcpManager();
  ~TcpManager();
  bool initialize(const std::string& local_ip = "", const Token& token = "default_token");
  void registerGID(const GID& gid, const std::string& ip, uint16_t port);
  ConnectionId createConnection(const ConnectionConfig& config = ConnectionConfig());
  QPId createQP(ConnectionId conn_id, const QPConfig& config = QPConfig());
  bool modifyQPToInit(ConnectionId conn_id, QPId qp_id);
  bool modifyQPToRTR(ConnectionId conn_id, QPId qp_id, const GID& gid);
  bool modifyQPToRTS(ConnectionId conn_id, QPId qp_id);
  bool postSend(ConnectionId conn_id, QPId qp_id, const TcpSendWR* wr);
  bool postRecv(ConnectionId conn_id, QPId qp_id, const TcpRecvWR* wr);
  bool postWrite(ConnectionId conn_id, QPId qp_id, const TcpWriteWR* wr);
  bool postRead(ConnectionId conn_id, QPId qp_id, const TcpReadWR* wr);
  int pollCQ(ConnectionId conn_id, int num_entries, TcpWC* wc);

private:
  void workerLoop();
  void handleNewConnection();
  void handleSocketEvent(int fd, uint32_t events);
  void handleWriteEvent(int fd, std::shared_ptr<TcpQueuePair> qp, std::shared_ptr<TcpCompletionQueue> cq);
  void handleReadEvent(int fd, std::shared_ptr<TcpQueuePair> qp, std::shared_ptr<TcpCompletionQueue> cq);
  void handleRemoteWrite(int fd, std::shared_ptr<TcpQueuePair> qp, std::shared_ptr<TcpCompletionQueue> cq, const TcpCtrlHeader& header);
  void handleRemoteRead(int fd, std::shared_ptr<TcpQueuePair> qp, std::shared_ptr<TcpCompletionQueue> cq, const TcpCtrlHeader& header);
  std::shared_ptr<TcpQueuePair> getQP(ConnectionId conn_id, QPId qp_id);
  std::shared_ptr<TcpCompletionQueue> getCQ(ConnectionId conn_id);
  void wakeupWorker();
  void cleanup();

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
