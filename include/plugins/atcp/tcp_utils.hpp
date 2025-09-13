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

namespace pccl {

struct TcpRemoteConnectionMetadata {
    std::string ip; uint16_t port;
    TcpRemoteConnectionMetadata() : port(0) {}
    TcpRemoteConnectionMetadata(const std::string& ip, uint16_t port) : ip(ip), port(port) {}
    TcpRemoteConnectionMetadata(const TcpRemoteConnectionMetadata&) = default;
    TcpRemoteConnectionMetadata(TcpRemoteConnectionMetadata&&) noexcept = default;
    TcpRemoteConnectionMetadata& operator=(const TcpRemoteConnectionMetadata&) = default;
    TcpRemoteConnectionMetadata& operator=(TcpRemoteConnectionMetadata&&) noexcept = default;
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
    ~TcpDeviceList() = default;
    TcpDeviceList(const TcpDeviceList&) = delete;
    TcpDeviceList& operator=(const TcpDeviceList&) = delete;
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
    ~TcpContext() = default;
    TcpContext(const TcpContext&) = delete;
    TcpContext& operator=(const TcpContext&) = delete;
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
    TcpProtectionDomain() = default;
    ~TcpProtectionDomain() = default;
    TcpProtectionDomain(const TcpProtectionDomain&) = delete;
    TcpProtectionDomain& operator=(const TcpProtectionDomain&) = delete;
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
    std::string remote_ip;
    uint16_t remote_port;
    uint64_t remote_addr;
    uint32_t remote_rkey;
};

struct TcpCtrlHeader {
    enum class Op { Write, Read, Data };
    Op op;
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

class TcpCompletionQueue {
public:
    TcpCompletionQueue() = default;
    ~TcpCompletionQueue() = default;
    TcpCompletionQueue(const TcpCompletionQueue&) = delete;
    TcpCompletionQueue& operator=(const TcpCompletionQueue&) = delete;
    void addCompletion(const TcpWC& wc) { 
        std::lock_guard<std::mutex> lock(mutex_); 
        completion_queue_.push(wc); 
    }
    int poll(TcpWC* wc, int num_entries) {
        std::lock_guard<std::mutex> lock(mutex_);
        int count = 0;
        while (count < num_entries && !completion_queue_.empty()) {
            *wc = completion_queue_.front(); 
            completion_queue_.pop(); 
            wc++; 
            count++;
        }
        return count;
    }
private:
    std::queue<TcpWC> completion_queue_;
    std::mutex mutex_;
};

struct SendTaskContext {
    std::vector<TcpSGE> sg_list;
    size_t curr_sge_idx = 0;
    size_t curr_sge_offset = 0;
    size_t total_bytes = 0;
    explicit SendTaskContext(const TcpSendWR* wr) {
        if (!wr) return;
        for (const auto* curr_wr = wr; curr_wr; curr_wr = curr_wr->next) {
            for (int i = 0; i < curr_wr->num_sge; ++i) {
                sg_list.push_back(curr_wr->sg_list[i]);
                total_bytes += curr_wr->sg_list[i].length;
            }
        }
    }
    bool is_completed() const {
        return curr_sge_idx >= sg_list.size();
    }
    std::pair<void*, size_t> get_current_data() const {
        if (is_completed()) return {nullptr, 0};
        const auto& sge = sg_list[curr_sge_idx];
        void* addr = static_cast<char*>(sge.addr) + curr_sge_offset;
        size_t remaining = sge.length - curr_sge_offset;
        return {addr, remaining};
    }
    bool update_sent(size_t bytes) {
        if (is_completed()) return false;
        const auto& sge = sg_list[curr_sge_idx];
        curr_sge_offset += bytes;
        if (curr_sge_offset >= sge.length) {
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
            for (int i = 0; i < curr_wr->num_sge; ++i) {
                sg_list.push_back(curr_wr->sg_list[i]);
                total_bytes += curr_wr->sg_list[i].length;
            }
        }
    }
    bool is_completed() const {
        return curr_sge_idx >= sg_list.size();
    }
    std::pair<void*, size_t> get_current_buffer() const {
        if (is_completed()) return {nullptr, 0};
        const auto& sge = sg_list[curr_sge_idx];
        void* addr = static_cast<char*>(sge.addr) + curr_sge_offset;
        size_t remaining = sge.length - curr_sge_offset;
        return {addr, remaining};
    }
    bool update_received(size_t bytes) {
        if (is_completed()) return false;
        const auto& sge = sg_list[curr_sge_idx];
        curr_sge_offset += bytes;
        if (curr_sge_offset >= sge.length) {
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
            for (int i = 0; i < curr_wr->num_sge; ++i) {
                sg_list.push_back(curr_wr->sg_list[i]);
                total_bytes += curr_wr->sg_list[i].length;
            }
        }
    }
    bool is_completed() const {
        return curr_sge_idx >= sg_list.size() && ctrl_sent;
    }
    std::vector<char> get_ctrl_header() const {
        TcpCtrlHeader header;
        header.op = TcpCtrlHeader::Op::Write;
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
        void* addr = static_cast<char*>(sge.addr) + curr_sge_offset;
        size_t remaining = sge.length - curr_sge_offset;
        return {addr, remaining};
    }
    bool update_sent(size_t bytes) {
        if (is_completed()) return false;
        const auto& sge = sg_list[curr_sge_idx];
        curr_sge_offset += bytes;
        if (curr_sge_offset >= sge.length) {
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
            for (int i = 0; i < curr_wr->num_sge; ++i) {
                sg_list.push_back(curr_wr->sg_list[i]);
                total_bytes += curr_wr->sg_list[i].length;
            }
        }
    }
    bool is_completed() const {
        return curr_sge_idx >= sg_list.size() && ctrl_sent && data_received;
    }
    std::vector<char> get_ctrl_header() const {
        TcpCtrlHeader header;
        header.op = TcpCtrlHeader::Op::Read;
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
        void* addr = static_cast<char*>(sge.addr) + curr_sge_offset;
        size_t remaining = sge.length - curr_sge_offset;
        return {addr, remaining};
    }
    bool update_received(size_t bytes) {
        if (is_completed()) return false;
        const auto& sge = sg_list[curr_sge_idx];
        curr_sge_offset += bytes;
        if (curr_sge_offset >= sge.length) {
            curr_sge_idx++;
            curr_sge_offset = 0;
            return true;
        }
        return false;
    }
};

class TcpQueuePair {
public:
    TcpQueuePair() : sock_fd_(-1), state_(TcpQPState::Reset) {}
    ~TcpQueuePair() { if (sock_fd_ != -1) close(sock_fd_); }
    TcpQueuePair(const TcpQueuePair&) = delete;
    TcpQueuePair& operator=(const TcpQueuePair&) = delete;
    bool modifyToInit() {
        if (state_ != TcpQPState::Reset) return false;
        sock_fd_ = socket(AF_INET, SOCK_STREAM, 0);
        if (sock_fd_ == -1) return false;
        state_ = TcpQPState::Init;
        return true;
    }
    bool modifyToRTR(const TcpRemoteConnectionMetadata& remote) {
        if (state_ != TcpQPState::Init) return false;
        struct sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(remote.port);
        if (inet_pton(AF_INET, remote.ip.c_str(), &addr.sin_addr) != 1) return false;
        if (connect(sock_fd_, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) == -1) {
            if (errno != EINPROGRESS) return false;
        }
        state_ = TcpQPState::Rtr;
        return true;
    }
    bool modifyToRTS() { 
        if (state_ != TcpQPState::Rtr) return false; 
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
private:
    int sock_fd_;
    TcpQPState state_;
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
    struct ConnectionConfig { uint16_t local_port = 0; int cq_size = 100; };
    struct QPConfig { int max_send_wr = 100; int max_recv_wr = 100; int max_send_sge = 1; int max_recv_sge = 1; };
    using ConnectionId = uint64_t; 
    using QPId = uint64_t;

    TcpManager() : epoll_fd_(-1), wakeup_pipe_{-1, -1}, listen_socket_(-1),
                   initialized_(false), running_(false), next_connection_id_(1), next_qp_id_(1) {}
    ~TcpManager() {
        running_ = false;
        if (wakeup_pipe_[1] != -1) write(wakeup_pipe_[1], "", 1);
        if (worker_thread_.joinable()) worker_thread_.join();
        if (epoll_fd_ != -1) close(epoll_fd_);
        if (wakeup_pipe_[0] != -1) close(wakeup_pipe_[0]);
        if (wakeup_pipe_[1] != -1) close(wakeup_pipe_[1]);
        if (listen_socket_ != -1) close(listen_socket_);
    }
    TcpManager(const TcpManager&) = delete; 
    TcpManager& operator=(const TcpManager&) = delete;

    bool initialize(const std::string& local_ip = "") {
        if (initialized_) return true;
        try {
            if (local_ip.empty()) {
                TcpDeviceList devices;
                if (devices.count() == 0) return false;
                local_ip_ = devices[0];
            } else local_ip_ = local_ip;
            context_ = std::make_shared<TcpContext>(local_ip_);
            pd_ = std::make_shared<TcpProtectionDomain>();
            epoll_fd_ = epoll_create1(EPOLL_CLOEXEC);
            if (epoll_fd_ == -1) throw std::runtime_error("");
            if (pipe2(wakeup_pipe_, O_NONBLOCK | O_CLOEXEC) == -1) 
                throw std::runtime_error("");
            epoll_event wakeup_ev{};
            wakeup_ev.events = EPOLLIN;
            wakeup_ev.data.fd = wakeup_pipe_[0];
            if (epoll_ctl(epoll_fd_, EPOLL_CTL_ADD, wakeup_pipe_[0], &wakeup_ev) == -1)
                throw std::runtime_error("");
            listen_socket_ = socket(AF_INET, SOCK_STREAM, 0);
            if (listen_socket_ == -1) throw std::runtime_error("");
            int opt = 1;
            if (setsockopt(listen_socket_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) == -1) throw std::runtime_error("");
            struct sockaddr_in listen_addr{};
            listen_addr.sin_family = AF_INET;
            listen_addr.sin_port = htons(51113);
            if (inet_pton(AF_INET, local_ip_.c_str(), &listen_addr.sin_addr) != 1) throw std::runtime_error("");
            if (bind(listen_socket_, reinterpret_cast<struct sockaddr*>(&listen_addr), sizeof(listen_addr)) == -1) throw std::runtime_error("");
            if (listen(listen_socket_, SOMAXCONN) == -1) throw std::runtime_error("");
            epoll_event listen_ev{};
            listen_ev.events = EPOLLIN;
            listen_ev.data.fd = listen_socket_;
            if (epoll_ctl(epoll_fd_, EPOLL_CTL_ADD, listen_socket_, &listen_ev) == -1) throw std::runtime_error("");
            running_ = true;
            worker_thread_ = std::thread(&TcpManager::workerThread, this);
            initialized_ = true;
            return true;
        } catch (...) {
            if (epoll_fd_ != -1) close(epoll_fd_);
            if (wakeup_pipe_[0] != -1) close(wakeup_pipe_[0]);
            if (wakeup_pipe_[1] != -1) close(wakeup_pipe_[1]);
            if (listen_socket_ != -1) close(listen_socket_);
            return false;
        }
    }

    ConnectionId createConnection(const ConnectionConfig& config) {
        if (!initialized_) return 0;
        ConnectionId conn_id = next_connection_id_++;
        ConnectionInfo conn_info;
        conn_info.config = config;
        conn_info.cq = std::make_shared<TcpCompletionQueue>();
        std::lock_guard<std::mutex> lock(connections_mutex_);
        connections_[conn_id] = std::move(conn_info);
        return conn_id;
    }

    bool destroyConnection(ConnectionId conn_id) {
        std::lock_guard<std::mutex> lock(connections_mutex_);
        auto it = connections_.find(conn_id);
        if (it == connections_.end()) return false;
        for (auto& [qp_id, qp] : it->second.qps) {
            int sock_fd = qp->getSockFd();
            if (sock_fd != -1) {
                epoll_ctl(epoll_fd_, EPOLL_CTL_DEL, sock_fd, nullptr);
                std::lock_guard<std::mutex> fd_lock(fd_to_qp_info_mutex_);
                fd_to_qp_info_.erase(sock_fd);
            }
        }
        connections_.erase(it);
        return true;
    }

    QPId createQP(ConnectionId conn_id, const QPConfig& config) {
        std::lock_guard<std::mutex> lock(connections_mutex_);
        auto conn_it = connections_.find(conn_id);
        if (conn_it == connections_.end()) return 0;
        QPId qp_id = next_qp_id_++;
        conn_it->second.qps[qp_id] = std::make_shared<TcpQueuePair>();
        return qp_id;
    }

    bool destroyQP(ConnectionId conn_id, QPId qp_id) {
        std::lock_guard<std::mutex> lock(connections_mutex_);
        auto conn_it = connections_.find(conn_id);
        if (conn_it == connections_.end()) return false;
        auto qp_it = conn_it->second.qps.find(qp_id);
        if (qp_it == conn_it->second.qps.end()) return false;
        auto qp = qp_it->second;
        int sock_fd = qp->getSockFd();
        if (sock_fd != -1) {
            epoll_ctl(epoll_fd_, EPOLL_CTL_DEL, sock_fd, nullptr);
            std::lock_guard<std::mutex> fd_lock(fd_to_qp_info_mutex_);
            fd_to_qp_info_.erase(sock_fd);
        }
        conn_it->second.qps.erase(qp_it);
        return true;
    }

    bool modifyQPToInit(ConnectionId conn_id, QPId qp_id) {
        std::lock_guard<std::mutex> lock(connections_mutex_);
        auto conn_it = connections_.find(conn_id);
        if (conn_it == connections_.end()) return false;
        auto qp_it = conn_it->second.qps.find(qp_id);
        if (qp_it == conn_it->second.qps.end()) return false;
        return qp_it->second->modifyToInit();
    }

    bool modifyQPToRTR(ConnectionId conn_id, QPId qp_id, const TcpRemoteConnectionMetadata& remote) {
        std::lock_guard<std::mutex> lock(connections_mutex_);
        auto conn_it = connections_.find(conn_id);
        if (conn_it == connections_.end()) return false;
        auto qp_it = conn_it->second.qps.find(qp_id);
        if (qp_it == conn_it->second.qps.end()) return false;
        return qp_it->second->modifyToRTR(remote);
    }

    bool modifyQPToRTS(ConnectionId conn_id, QPId qp_id) {
        std::lock_guard<std::mutex> conn_lock(connections_mutex_);
        auto conn_it = connections_.find(conn_id);
        if (conn_it == connections_.end()) return false;
        auto qp_it = conn_it->second.qps.find(qp_id);
        if (qp_it == conn_it->second.qps.end()) return false;
        auto qp = qp_it->second;
        if (!qp->modifyToRTS()) return false;
        int sock_fd = qp->getSockFd();
        if (sock_fd == -1) {
            qp->setState(TcpQPState::Err);
            return false;
        }
        int flags = fcntl(sock_fd, F_GETFL, 0);
        if (flags == -1 || fcntl(sock_fd, F_SETFL, flags | O_NONBLOCK) == -1) {
            qp->setState(TcpQPState::Err);
            return false;
        }
        epoll_event ev{};
        ev.events = EPOLLIN | EPOLLOUT | EPOLLET;
        ev.data.fd = sock_fd;
        if (epoll_ctl(epoll_fd_, EPOLL_CTL_ADD, sock_fd, &ev) == -1) {
            qp->setState(TcpQPState::Err);
            return false;
        }
        std::lock_guard<std::mutex> fd_lock(fd_to_qp_info_mutex_);
        fd_to_qp_info_[sock_fd] = {conn_id, qp_id};
        return true;
    }

    bool postSend(ConnectionId conn_id, QPId qp_id, const TcpSendWR* wr) {
        try {
            auto qp = getQP(conn_id, qp_id);
            if (!qp) return false;
            qp->postSend(wr);
            return true;
        } catch (...) { return false; }
    }

    bool postRecv(ConnectionId conn_id, QPId qp_id, const TcpRecvWR* wr) {
        try {
            auto qp = getQP(conn_id, qp_id);
            if (!qp) return false;
            qp->postRecv(wr);
            return true;
        } catch (...) { return false; }
    }

    bool postWrite(ConnectionId conn_id, QPId qp_id, const TcpWriteWR* wr) {
        try {
            auto qp = getQP(conn_id, qp_id);
            if (!qp) return false;
            qp->postWrite(wr);
            return true;
        } catch (...) { return false; }
    }

    bool postRead(ConnectionId conn_id, QPId qp_id, const TcpReadWR* wr) {
        try {
            auto qp = getQP(conn_id, qp_id);
            if (!qp) return false;
            qp->postRead(wr);
            return true;
        } catch (...) { return false; }
    }

    int pollCQ(ConnectionId conn_id, int num_entries, TcpWC* wc) {
        auto cq = getCQ(conn_id);
        return cq ? cq->poll(wc, num_entries) : -1;
    }

    std::shared_ptr<TcpProtectionDomain> getPD() const { return pd_; }

private:
    struct ConnectionInfo {
        ConnectionConfig config;
        std::shared_ptr<TcpCompletionQueue> cq;
        std::unordered_map<QPId, std::shared_ptr<TcpQueuePair>> qps;
    };

    void workerThread() {
        constexpr int MAX_EVENTS = 1024;
        epoll_event events[MAX_EVENTS];
        while (running_) {
            int n = epoll_wait(epoll_fd_, events, MAX_EVENTS, -1);
            if (n == -1) {
                if (errno == EINTR) continue;
                break;
            }
            for (int i = 0; i < n; ++i) {
                int sock_fd = events[i].data.fd;
                uint32_t event_mask = events[i].events;
                if (sock_fd == wakeup_pipe_[0]) {
                    char buf[16];
                    while (read(sock_fd, buf, sizeof(buf)) > 0);
                    continue;
                }
                if (sock_fd == listen_socket_) {
                    handleNewConnection();
                    continue;
                }
                handleConnectedSocketEvent(sock_fd, event_mask);
            }
        }
    }

    void handleNewConnection() {
        struct sockaddr_in client_addr{};
        socklen_t client_addr_len = sizeof(client_addr);
        int client_sock = accept4(listen_socket_, reinterpret_cast<struct sockaddr*>(&client_addr),
                                 &client_addr_len, SOCK_NONBLOCK | SOCK_CLOEXEC);
        if (client_sock == -1) return;
        char client_ip[INET_ADDRSTRLEN];
        if (!inet_ntop(AF_INET, &client_addr.sin_addr, client_ip, sizeof(client_ip))) {
            close(client_sock);
            return;
        }
        auto qp = std::make_shared<TcpQueuePair>();
        if (!qp->modifyToInit() || !qp->modifyToRTR(TcpRemoteConnectionMetadata(client_ip, ntohs(client_addr.sin_port))) || !qp->modifyToRTS()) {
            close(client_sock);
            return;
        }
        epoll_event ev{};
        ev.events = EPOLLIN | EPOLLOUT | EPOLLET;
        ev.data.fd = client_sock;
        if (epoll_ctl(epoll_fd_, EPOLL_CTL_ADD, client_sock, &ev) == -1) {
            close(client_sock);
            return;
        }
        ConnectionId conn_id = next_connection_id_++;
        QPId qp_id = next_qp_id_++;
        ConnectionInfo conn_info;
        conn_info.config = ConnectionConfig{};
        conn_info.cq = std::make_shared<TcpCompletionQueue>();
        conn_info.qps[qp_id] = qp;
        std::lock_guard<std::mutex> conn_lock(connections_mutex_);
        connections_[conn_id] = std::move(conn_info);
        std::lock_guard<std::mutex> fd_lock(fd_to_qp_info_mutex_);
        fd_to_qp_info_[client_sock] = {conn_id, qp_id};
    }

    void handleConnectedSocketEvent(int sock_fd, uint32_t event_mask) {
        std::pair<ConnectionId, QPId> qp_info;
        {
            std::lock_guard<std::mutex> fd_lock(fd_to_qp_info_mutex_);
            auto it = fd_to_qp_info_.find(sock_fd);
            if (it == fd_to_qp_info_.end()) return;
            qp_info = it->second;
        }
        ConnectionId conn_id = qp_info.first;
        QPId qp_id = qp_info.second;
        std::shared_ptr<TcpCompletionQueue> cq;
        std::shared_ptr<TcpQueuePair> qp;
        {
            std::lock_guard<std::mutex> conn_lock(connections_mutex_);
            auto conn_it = connections_.find(conn_id);
            if (conn_it == connections_.end()) return;
            auto qp_it = conn_it->second.qps.find(qp_id);
            if (qp_it == conn_it->second.qps.end()) return;
            cq = conn_it->second.cq;
            qp = qp_it->second;
        }
        if (event_mask & (EPOLLERR | EPOLLHUP)) {
            handleErrorEvent(sock_fd, conn_id, qp_id, cq, qp);
            return;
        }
        if (event_mask & EPOLLOUT) {
            handleWriteEvent(sock_fd, cq, qp);
        }
        if (event_mask & EPOLLIN) {
            handleReadEvent(sock_fd, cq, qp);
        }
    }

    void handleErrorEvent(int sock_fd, ConnectionId conn_id, QPId qp_id, 
                        std::shared_ptr<TcpCompletionQueue> cq, 
                        std::shared_ptr<TcpQueuePair> qp) {
        qp->setState(TcpQPState::Err);
        if (cq) {
            TcpWC wc{};
            wc.status = TcpWCStatus::ConnectionReset;
            wc.opcode = TcpWCOpcode::Recv;
            cq->addCompletion(wc);
        }
        close(sock_fd);
        std::lock_guard<std::mutex> fd_lock(fd_to_qp_info_mutex_);
        fd_to_qp_info_.erase(sock_fd);
        epoll_ctl(epoll_fd_, EPOLL_CTL_DEL, sock_fd, nullptr);
    }

    void handleWriteEvent(int sock_fd, std::shared_ptr<TcpCompletionQueue> cq, std::shared_ptr<TcpQueuePair> qp) {
        std::lock_guard<std::mutex> send_lock(qp->getSendMutex());
        auto& send_queue = qp->getSendQueue();
        while (!send_queue.empty()) {
            auto& task = send_queue.front();
            if (task.is_completed()) { send_queue.pop(); continue; }
            auto [data, len] = task.get_current_data();
            ssize_t sent = send(sock_fd, data, len, MSG_NOSIGNAL);
            if (sent <= 0) break;
            task.update_sent(sent);
            if (task.is_completed()) {
                TcpWC wc{.status = TcpWCStatus::Success, .opcode = TcpWCOpcode::Send, .byte_len = task.total_bytes};
                cq->addCompletion(wc);
                send_queue.pop();
            }
        }
        std::lock_guard<std::mutex> write_lock(qp->getWriteMutex());
        auto& write_queue = qp->getWriteQueue();
        while (!write_queue.empty()) {
            auto& task = write_queue.front();
            if (task.is_completed()) { write_queue.pop(); continue; }
            if (!task.ctrl_sent) {
                auto ctrl_header = task.get_ctrl_header();
                ssize_t sent = send(sock_fd, ctrl_header.data(), ctrl_header.size(), MSG_NOSIGNAL);
                if (sent != (ssize_t)ctrl_header.size()) break;
                task.ctrl_sent = true;
            }
            auto [data, len] = task.get_current_data();
            ssize_t sent = send(sock_fd, data, len, MSG_NOSIGNAL);
            if (sent <= 0) break;
            task.update_sent(sent);
            if (task.is_completed()) {
                TcpWC wc{.status = TcpWCStatus::Success, .opcode = TcpWCOpcode::Write, .byte_len = task.total_bytes};
                cq->addCompletion(wc);
                write_queue.pop();
            }
        }
        std::lock_guard<std::mutex> read_lock(qp->getReadMutex());
        auto& read_queue = qp->getReadQueue();
        while (!read_queue.empty()) {
            auto& task = read_queue.front();
            if (task.is_completed()) { read_queue.pop(); continue; }
            if (!task.ctrl_sent) {
                auto ctrl_header = task.get_ctrl_header();
                ssize_t sent = send(sock_fd, ctrl_header.data(), ctrl_header.size(), MSG_NOSIGNAL);
                if (sent != (ssize_t)ctrl_header.size()) break;
                task.ctrl_sent = true;
            }
        }
    }

    void handleReadEvent(int sock_fd, std::shared_ptr<TcpCompletionQueue> cq, std::shared_ptr<TcpQueuePair> qp) {
        static std::vector<char> ctrl_buf(sizeof(TcpCtrlHeader));
        static size_t ctrl_received = 0;
        while (ctrl_received < sizeof(TcpCtrlHeader)) {
            ssize_t recvd = recv(sock_fd, ctrl_buf.data() + ctrl_received, sizeof(TcpCtrlHeader) - ctrl_received, MSG_DONTWAIT);
            if (recvd <= 0) return;
            ctrl_received += recvd;
        }
        TcpCtrlHeader header;
        memcpy(&header, ctrl_buf.data(), sizeof(TcpCtrlHeader));
        ctrl_received = 0;
        switch (header.op) {
            case TcpCtrlHeader::Op::Write: handleRemoteWrite(sock_fd, cq, qp, header); break;
            case TcpCtrlHeader::Op::Read:  handleRemoteRead(sock_fd, cq, qp, header);  break;
            case TcpCtrlHeader::Op::Data:  handleRemoteReadData(sock_fd, cq, qp, header); break;
            default: handleErrorEvent(sock_fd, 0, 0, cq, qp); break;
        }
    }

    void handleRemoteWrite(int sock_fd, std::shared_ptr<TcpCompletionQueue> cq, std::shared_ptr<TcpQueuePair> qp, const TcpCtrlHeader& header) {
        auto mr = pd_->lookUpMR(header.remote_rkey);
        if (!mr) {
            TcpWC wc{.status = TcpWCStatus::Failure, .opcode = TcpWCOpcode::Write, .byte_len = 0};
            cq->addCompletion(wc);
            return;
        }
        uint64_t mr_start = reinterpret_cast<uint64_t>(mr->getAddr());
        uint64_t mr_end = mr_start + mr->getLength();
        if (header.remote_addr < mr_start || header.remote_addr + header.length > mr_end) {
            TcpWC wc{.status = TcpWCStatus::Failure, .opcode = TcpWCOpcode::Write, .byte_len = 0};
            cq->addCompletion(wc);
            return;
        }
        void* buf = reinterpret_cast<void*>(header.remote_addr);
        size_t remaining = header.length;
        while (remaining > 0) {
            ssize_t recvd = recv(sock_fd, buf, remaining, MSG_DONTWAIT);
            if (recvd <= 0) break;
            buf = reinterpret_cast<char*>(buf) + recvd;
            remaining -= recvd;
        }
        if (remaining == 0) {
            TcpWC wc{.status = TcpWCStatus::Success, .opcode = TcpWCOpcode::Write, .byte_len = header.length};
            cq->addCompletion(wc);
        }
    }

    void handleRemoteRead(int sock_fd, std::shared_ptr<TcpCompletionQueue> cq, std::shared_ptr<TcpQueuePair> qp, const TcpCtrlHeader& header) {
        auto mr = pd_->lookUpMR(header.remote_rkey);
        if (!mr) {
            TcpWC wc{.status = TcpWCStatus::Failure, .opcode = TcpWCOpcode::Read, .byte_len = 0};
            cq->addCompletion(wc);
            return;
        }
        uint64_t mr_start = reinterpret_cast<uint64_t>(mr->getAddr());
        uint64_t mr_end = mr_start + mr->getLength();
        if (header.remote_addr < mr_start || header.remote_addr + header.length > mr_end) {
            TcpWC wc{.status = TcpWCStatus::Failure, .opcode = TcpWCOpcode::Read, .byte_len = 0};
            cq->addCompletion(wc);
            return;
        }
        TcpCtrlHeader data_header{.op = TcpCtrlHeader::Op::Data, .remote_addr = header.remote_addr, .remote_rkey = header.remote_rkey, .length = header.length};
        ssize_t sent = send(sock_fd, &data_header, sizeof(data_header), MSG_NOSIGNAL);
        if (sent != sizeof(data_header)) {
            TcpWC wc{.status = TcpWCStatus::Failure, .opcode = TcpWCOpcode::Read, .byte_len = 0};
            cq->addCompletion(wc);
            return;
        }
        void* data = reinterpret_cast<void*>(header.remote_addr);
        size_t remaining = header.length;
        while (remaining > 0) {
            ssize_t sent_data = send(sock_fd, data, remaining, MSG_NOSIGNAL);
            if (sent_data <= 0) break;
            data = reinterpret_cast<char*>(data) + sent_data;
            remaining -= sent_data;
        }
        if (remaining == 0) {
            TcpWC wc{.status = TcpWCStatus::Success, .opcode = TcpWCOpcode::Read, .byte_len = header.length};
            cq->addCompletion(wc);
        }
    }

    void handleRemoteReadData(int sock_fd, std::shared_ptr<TcpCompletionQueue> cq, std::shared_ptr<TcpQueuePair> qp, const TcpCtrlHeader& header) {
        std::lock_guard<std::mutex> read_lock(qp->getReadMutex());
        auto& read_queue = qp->getReadQueue();
        if (read_queue.empty()) return;
        auto& task = read_queue.front();
        if (task.total_bytes != header.length) {
            TcpWC wc{.status = TcpWCStatus::Failure, .opcode = TcpWCOpcode::Read, .byte_len = 0};
            cq->addCompletion(wc);
            read_queue.pop();
            return;
        }
        size_t remaining = header.length;
        while (remaining > 0 && !task.is_completed()) {
            auto [buf, len] = task.get_current_buffer();
            ssize_t recvd = recv(sock_fd, buf, len, MSG_DONTWAIT);
            if (recvd <= 0) break;
            task.update_received(recvd);
            remaining -= recvd;
        }
        if (remaining == 0) {
            task.data_received = true;
            if (task.is_completed()) {
                TcpWC wc{.status = TcpWCStatus::Success, .opcode = TcpWCOpcode::Read, .byte_len = task.total_bytes};
                cq->addCompletion(wc);
                read_queue.pop();
            }
        }
    }

    std::shared_ptr<TcpQueuePair> getQP(ConnectionId conn_id, QPId qp_id) {
        std::lock_guard<std::mutex> lock(connections_mutex_);
        auto conn_it = connections_.find(conn_id);
        if (conn_it == connections_.end()) return nullptr;
        auto qp_it = conn_it->second.qps.find(qp_id);
        return qp_it != conn_it->second.qps.end() ? qp_it->second : nullptr;
    }

    std::shared_ptr<TcpCompletionQueue> getCQ(ConnectionId conn_id) {
        std::lock_guard<std::mutex> lock(connections_mutex_);
        auto it = connections_.find(conn_id);
        return it != connections_.end() ? it->second.cq : nullptr;
    }

private:
    std::string local_ip_;
    std::shared_ptr<TcpContext> context_;
    std::shared_ptr<TcpProtectionDomain> pd_;
    std::unordered_map<ConnectionId, ConnectionInfo> connections_;
    mutable std::mutex connections_mutex_;
    int epoll_fd_;
    int wakeup_pipe_[2];
    int listen_socket_;
    std::unordered_map<int, std::pair<ConnectionId, QPId>> fd_to_qp_info_;
    mutable std::mutex fd_to_qp_info_mutex_;
    bool initialized_;
    std::atomic<bool> running_;
    std::thread worker_thread_;
    std::atomic<ConnectionId> next_connection_id_;
    std::atomic<QPId> next_qp_id_;
};

}