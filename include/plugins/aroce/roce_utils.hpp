#pragma once

#include <unordered_map>
#include <memory>
#include <string>
#include <mutex>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <infiniband/verbs.h>
#include "utils/exception.hpp"

#define IBV_DEFAULT_QKEY 0x01234567

namespace pccl {

struct RemoteConnectionMetadata;

class IbverbsLib {
public:
    static IbverbsLib& getInstance() {
        static IbverbsLib instance;
        return instance;
    }
    
    void load(const std::string& library_path = "libibverbs.so") {
        std::lock_guard<std::mutex> lock(load_mutex_);
        if (handle_) {
            return;
        }
        
        handle_ = dlopen(library_path.c_str(), RTLD_LAZY);
        PCCL_THROW_IF(!handle_, RuntimeException, "Failed to open ", library_path);
        
        #define LOAD_VERBS_FUNC(name) \
            name = reinterpret_cast<name##_fn>(dlsym(handle_, #name)); \
            PCCL_THROW_IF(!name, RuntimeException, "Failed to load function ", #name);
        
        LOAD_VERBS_FUNC(ibv_get_device_list)
        LOAD_VERBS_FUNC(ibv_free_device_list)
        LOAD_VERBS_FUNC(ibv_get_device_name)
        LOAD_VERBS_FUNC(ibv_open_device)
        LOAD_VERBS_FUNC(ibv_close_device)
        LOAD_VERBS_FUNC(ibv_query_device)
        LOAD_VERBS_FUNC(ibv_query_port)
        LOAD_VERBS_FUNC(ibv_alloc_pd)
        LOAD_VERBS_FUNC(ibv_dealloc_pd)
        LOAD_VERBS_FUNC(ibv_reg_mr)
        LOAD_VERBS_FUNC(ibv_dereg_mr)
        LOAD_VERBS_FUNC(ibv_create_cq)
        LOAD_VERBS_FUNC(ibv_destroy_cq)
        LOAD_VERBS_FUNC(ibv_poll_cq)
        LOAD_VERBS_FUNC(ibv_create_qp)
        LOAD_VERBS_FUNC(ibv_destroy_qp)
        LOAD_VERBS_FUNC(ibv_modify_qp)
        LOAD_VERBS_FUNC(ibv_post_send)
        LOAD_VERBS_FUNC(ibv_post_recv)
        LOAD_VERBS_FUNC(ibv_req_notify_cq)
        
        #undef LOAD_VERBS_FUNC
    }

    ~IbverbsLib() {
        std::lock_guard<std::mutex> lock(load_mutex_);
        if (handle_) {
            dlclose(handle_);
            handle_ = nullptr;
        }
    }

    IbverbsLib(const IbverbsLib&) = delete;
    IbverbsLib& operator=(const IbverbsLib&) = delete;

    ibv_device** get_device_list(int* num_devices) {
        return ibv_get_device_list(num_devices);
    }
    
    int free_device_list(ibv_device** list) {
        return ibv_free_device_list(list);
    }
    
    const char* get_device_name(ibv_device* device) {
        return ibv_get_device_name(device);
    }
    
    ibv_context* open_device(ibv_device* device) {
        return ibv_open_device(device);
    }
    
    int close_device(ibv_context* context) {
        return ibv_close_device(context);
    }
    
    int query_device(ibv_context* context, ibv_device_attr* device_attr) {
        return ibv_query_device(context, device_attr);
    }
    
    int query_port(ibv_context* context, uint8_t port_num, ibv_port_attr* port_attr) {
        return ibv_query_port(context, port_num, port_attr);
    }
    
    ibv_pd* alloc_pd(ibv_context* context) {
        return ibv_alloc_pd(context);
    }
    
    int dealloc_pd(ibv_pd* pd) {
        return ibv_dealloc_pd(pd);
    }
    
    ibv_mr* reg_mr(ibv_pd* pd, void* addr, size_t length, int access) {
        return ibv_reg_mr(pd, addr, length, access);
    }
    
    int dereg_mr(ibv_mr* mr) {
        return ibv_dereg_mr(mr);
    }
    
    ibv_cq* create_cq(ibv_context* context, int cqe, void* cq_context, 
            ibv_comp_channel* channel, int comp_vector) {
        return ibv_create_cq(context, cqe, cq_context, channel, comp_vector);
    }
    
    int destroy_cq(ibv_cq* cq) {
        return ibv_destroy_cq(cq);
    }
    
    int poll_cq(ibv_cq* cq, int num_entries, ibv_wc* wc) {
        return ibv_poll_cq(cq, num_entries, wc);
    }
    
    ibv_qp* create_qp(ibv_pd* pd, ibv_qp_init_attr* attr) {
        return ibv_create_qp(pd, attr);
    }
    
    int destroy_qp(ibv_qp* qp) {
        return ibv_destroy_qp(qp);
    }
    
    int modify_qp(ibv_qp* qp, ibv_qp_attr* attr, int attr_mask) {
        return ibv_modify_qp(qp, attr, attr_mask);
    }
    
    int post_send(ibv_qp* qp, ibv_send_wr* wr, ibv_send_wr** bad_wr) {
        return ibv_post_send(qp, wr, bad_wr);
    }
    
    int post_recv(ibv_qp* qp, ibv_recv_wr* wr, ibv_recv_wr** bad_wr) {
        return ibv_post_recv(qp, wr, bad_wr);
    }
    
    int req_notify_cq(ibv_cq* cq, int solicited_only) {
        return ibv_req_notify_cq(cq, solicited_only);
    }

private:
    typedef ibv_device** (*ibv_get_device_list_fn)(int*);
    typedef int (*ibv_free_device_list_fn)(ibv_device**);
    typedef const char* (*ibv_get_device_name_fn)(ibv_device*);
    typedef ibv_context* (*ibv_open_device_fn)(ibv_device*);
    typedef int (*ibv_close_device_fn)(ibv_context*);
    typedef int (*ibv_query_device_fn)(ibv_context*, ibv_device_attr*);
    typedef int (*ibv_query_port_fn)(ibv_context*, uint8_t, ibv_port_attr*);
    typedef ibv_pd* (*ibv_alloc_pd_fn)(ibv_context*);
    typedef int (*ibv_dealloc_pd_fn)(ibv_pd*);
    typedef ibv_mr* (*ibv_reg_mr_fn)(ibv_pd*, void*, size_t, int);
    typedef int (*ibv_dereg_mr_fn)(ibv_mr*);
    typedef ibv_cq* (*ibv_create_cq_fn)(ibv_context*, int, void*, ibv_comp_channel*, int);
    typedef int (*ibv_destroy_cq_fn)(ibv_cq*);
    typedef int (*ibv_poll_cq_fn)(ibv_cq*, int, ibv_wc*);
    typedef ibv_qp* (*ibv_create_qp_fn)(ibv_pd*, ibv_qp_init_attr*);
    typedef int (*ibv_destroy_qp_fn)(ibv_qp*);
    typedef int (*ibv_modify_qp_fn)(ibv_qp*, ibv_qp_attr*, int);
    typedef int (*ibv_post_send_fn)(ibv_qp*, ibv_send_wr*, ibv_send_wr**);
    typedef int (*ibv_post_recv_fn)(ibv_qp*, ibv_recv_wr*, ibv_recv_wr**);
    typedef int (*ibv_req_notify_cq_fn)(ibv_cq*, int);

    ibv_get_device_list_fn ibv_get_device_list = nullptr;
    ibv_free_device_list_fn ibv_free_device_list = nullptr;
    ibv_get_device_name_fn ibv_get_device_name = nullptr;
    ibv_open_device_fn ibv_open_device = nullptr;
    ibv_close_device_fn ibv_close_device = nullptr;
    ibv_query_device_fn ibv_query_device = nullptr;
    ibv_query_port_fn ibv_query_port = nullptr;
    ibv_alloc_pd_fn ibv_alloc_pd = nullptr;
    ibv_dealloc_pd_fn ibv_dealloc_pd = nullptr;
    ibv_reg_mr_fn ibv_reg_mr = nullptr;
    ibv_dereg_mr_fn ibv_dereg_mr = nullptr;
    ibv_create_cq_fn ibv_create_cq = nullptr;
    ibv_destroy_cq_fn ibv_destroy_cq = nullptr;
    ibv_poll_cq_fn ibv_poll_cq = nullptr;
    ibv_create_qp_fn ibv_create_qp = nullptr;
    ibv_destroy_qp_fn ibv_destroy_qp = nullptr;
    ibv_modify_qp_fn ibv_modify_qp = nullptr;
    ibv_post_send_fn ibv_post_send = nullptr;
    ibv_post_recv_fn ibv_post_recv = nullptr;
    ibv_req_notify_cq_fn ibv_req_notify_cq = nullptr;

    IbverbsLib() = default;
    void* handle_ = nullptr;
    std::mutex load_mutex_;
};

class RdmaDeviceList {
public:
    RdmaDeviceList() {
        auto& lib = IbverbsLib::getInstance();
        devices_ = lib.get_device_list(&num_devices_);
        PCCL_THROW_IF(!devices_, RuntimeException, "Failed to get RDMA device list");
    }
    
    ~RdmaDeviceList() {
        if (devices_) {
            auto& lib = IbverbsLib::getInstance();
            lib.free_device_list(devices_);
            devices_ = nullptr;
            num_devices_ = 0;
        }
    }
    
    RdmaDeviceList(const RdmaDeviceList&) = delete;
    RdmaDeviceList& operator=(const RdmaDeviceList&) = delete;
    
    RdmaDeviceList(RdmaDeviceList&& other) noexcept 
        : devices_(other.devices_), num_devices_(other.num_devices_) {
        other.devices_ = nullptr;
        other.num_devices_ = 0;
    }
    
    RdmaDeviceList& operator=(RdmaDeviceList&& other) noexcept {
        if (this != &other) {
            if (devices_) {
                auto& lib = IbverbsLib::getInstance();
                lib.free_device_list(devices_);
            }
            
            devices_ = other.devices_;
            num_devices_ = other.num_devices_;
            
            other.devices_ = nullptr;
            other.num_devices_ = 0;
        }
        return *this;
    }
    
    int count() const { return num_devices_; }
    
    ibv_device* operator[](int index) const {
        PCCL_THROW_IF(index < 0 || index >= num_devices_, 
               RuntimeException, "Device index out of range");
        return devices_[index];
    }
    
    std::string get_device_name(int index) const {
        auto& lib = IbverbsLib::getInstance();
        return lib.get_device_name(devices_[index]);
    }
    
private:
    ibv_device** devices_ = nullptr;
    int num_devices_ = 0;
};

class RdmaContext {
public:
    explicit RdmaContext(ibv_device* device) {
        auto& lib = IbverbsLib::getInstance();
        context_ = lib.open_device(device);
        PCCL_THROW_IF(!context_, RuntimeException, "Failed to open RDMA device");
    }
    
    ~RdmaContext() {
        if (context_) {
            auto& lib = IbverbsLib::getInstance();
            lib.close_device(context_);
            context_ = nullptr;
        }
    }
    
    RdmaContext(const RdmaContext&) = delete;
    RdmaContext& operator=(const RdmaContext&) = delete;
    
    RdmaContext(RdmaContext&& other) noexcept : context_(other.context_) {
        other.context_ = nullptr;
    }
    
    RdmaContext& operator=(RdmaContext&& other) noexcept {
        if (this != &other) {
            if (context_) {
                auto& lib = IbverbsLib::getInstance();
                lib.close_device(context_);
            }
            
            context_ = other.context_;
            other.context_ = nullptr;
        }
        return *this;
    }
    
    ibv_context* get() const { return context_; }
    explicit operator bool() const { return context_ != nullptr; }
    
    ibv_device_attr query_device() const {
        auto& lib = IbverbsLib::getInstance();
        ibv_device_attr device_attr;
        PCCL_THROW_IF(lib.query_device(context_, &device_attr), 
               RuntimeException, "Failed to query device attributes");
        return device_attr;
    }
    
    ibv_port_attr query_port(uint8_t port_num) const {
        auto& lib = IbverbsLib::getInstance();
        ibv_port_attr port_attr;
        PCCL_THROW_IF(lib.query_port(context_, port_num, &port_attr), 
               RuntimeException, "Failed to query port attributes");
        return port_attr;
    }
    
private:
    ibv_context* context_ = nullptr;
};

class ProtectionDomain {
public:
    explicit ProtectionDomain(ibv_context* context) {
        auto& lib = IbverbsLib::getInstance();
        pd_ = lib.alloc_pd(context);
        PCCL_THROW_IF(!pd_, RuntimeException, "Failed to allocate protection domain");
    }
    
    ~ProtectionDomain() {
        if (pd_) {
            auto& lib = IbverbsLib::getInstance();
            lib.dealloc_pd(pd_);
            pd_ = nullptr;
        }
    }
    
    ProtectionDomain(const ProtectionDomain&) = delete;
    ProtectionDomain& operator=(const ProtectionDomain&) = delete;
    
    ProtectionDomain(ProtectionDomain&& other) noexcept : pd_(other.pd_) {
        other.pd_ = nullptr;
    }
    
    ProtectionDomain& operator=(ProtectionDomain&& other) noexcept {
        if (this != &other) {
            if (pd_) {
                auto& lib = IbverbsLib::getInstance();
                lib.dealloc_pd(pd_);
            }
            
            pd_ = other.pd_;
            other.pd_ = nullptr;
        }
        return *this;
    }
    
    ibv_pd* get() const { return pd_; }
    explicit operator bool() const { return pd_ != nullptr; }
    
private:
    ibv_pd* pd_ = nullptr;
};

class CompletionQueue {
public:
    CompletionQueue(ibv_context* context, int cqe, void* cq_context = nullptr) {
        auto& lib = IbverbsLib::getInstance();
        cq_ = lib.create_cq(context, cqe, cq_context, nullptr, 0);
        PCCL_THROW_IF(!cq_, RuntimeException, "Failed to create completion queue");
    }
    
    ~CompletionQueue() {
        if (cq_) {
            auto& lib = IbverbsLib::getInstance();
            lib.destroy_cq(cq_);
            cq_ = nullptr;
        }
    }
    
    CompletionQueue(const CompletionQueue&) = delete;
    CompletionQueue& operator=(const CompletionQueue&) = delete;
    
    CompletionQueue(CompletionQueue&& other) noexcept : cq_(other.cq_) {
        other.cq_ = nullptr;
    }
    
    CompletionQueue& operator=(CompletionQueue&& other) noexcept {
        if (this != &other) {
            if (cq_) {
                auto& lib = IbverbsLib::getInstance();
                lib.destroy_cq(cq_);
            }
            
            cq_ = other.cq_;
            other.cq_ = nullptr;
        }
        return *this;
    }
    
    ibv_cq* get() const { return cq_; }
    explicit operator bool() const { return cq_ != nullptr; }
    
    int poll(ibv_wc* wc, int num_entries) const {
        auto& lib = IbverbsLib::getInstance();
        return lib.poll_cq(cq_, num_entries, wc);
    }
    
    void request_notification(bool solicited_only = false) const {
        auto& lib = IbverbsLib::getInstance();
        PCCL_THROW_IF(lib.req_notify_cq(cq_, solicited_only ? 1 : 0), 
               RuntimeException, "Failed to request CQ notification");
    }
    
private:
    ibv_cq* cq_ = nullptr;
};

class MemoryRegion {
public:
    MemoryRegion(ibv_pd* pd, void* addr, size_t length, int access_flags) 
        : addr_(addr), length_(length) {
        auto& lib = IbverbsLib::getInstance();
        mr_ = lib.reg_mr(pd, addr, length, access_flags);
        PCCL_THROW_IF(!mr_, RuntimeException, "Failed to register memory region");
    }
    
    ~MemoryRegion() {
        if (mr_) {
            auto& lib = IbverbsLib::getInstance();
            lib.dereg_mr(mr_);
            mr_ = nullptr;
        }
    }
    
    MemoryRegion(const MemoryRegion&) = delete;
    MemoryRegion& operator=(const MemoryRegion&) = delete;
    
    MemoryRegion(MemoryRegion&& other) noexcept 
        : mr_(other.mr_), addr_(other.addr_), length_(other.length_) {
        other.mr_ = nullptr;
        other.addr_ = nullptr;
        other.length_ = 0;
    }
    
    MemoryRegion& operator=(MemoryRegion&& other) noexcept {
        if (this != &other) {
            if (mr_) {
                auto& lib = IbverbsLib::getInstance();
                lib.dereg_mr(mr_);
            }
            
            mr_ = other.mr_;
            addr_ = other.addr_;
            length_ = other.length_;
            
            other.mr_ = nullptr;
            other.addr_ = nullptr;
            other.length_ = 0;
        }
        return *this;
    }
    
    ibv_mr* get() const { return mr_; }
    void* addr() const { return addr_; }
    size_t length() const { return length_; }
    uint32_t lkey() const { return mr_ ? mr_->lkey : 0; }
    uint32_t rkey() const { return mr_ ? mr_->rkey : 0; }
    explicit operator bool() const { return mr_ != nullptr; }
    
private:
    ibv_mr* mr_ = nullptr;
    void* addr_ = nullptr;
    size_t length_ = 0;
};

class QueuePair {
public:
    QueuePair(ibv_pd* pd, ibv_qp_init_attr* init_attr) {
        auto& lib = IbverbsLib::getInstance();
        qp_ = lib.create_qp(pd, init_attr);
        PCCL_THROW_IF(!qp_, RuntimeException, "Failed to create queue pair");
    }
    
    ~QueuePair() {
        if (qp_) {
            auto& lib = IbverbsLib::getInstance();
            lib.destroy_qp(qp_);
            qp_ = nullptr;
        }
    }
    
    QueuePair(const QueuePair&) = delete;
    QueuePair& operator=(const QueuePair&) = delete;
    
    QueuePair(QueuePair&& other) noexcept : qp_(other.qp_) {
        other.qp_ = nullptr;
    }
    
    QueuePair& operator=(QueuePair&& other) noexcept {
        if (this != &other) {
            if (qp_) {
                auto& lib = IbverbsLib::getInstance();
                lib.destroy_qp(qp_);
            }
            
            qp_ = other.qp_;
            other.qp_ = nullptr;
        }
        return *this;
    }
    
    ibv_qp* get() const { return qp_; }
    explicit operator bool() const { return qp_ != nullptr; }
    
    void modify(ibv_qp_attr* attr, int attr_mask) {
        auto& lib = IbverbsLib::getInstance();
        PCCL_THROW_IF(lib.modify_qp(qp_, attr, attr_mask), 
               RuntimeException, "Failed to modify queue pair");
    }
    
    void post_send(ibv_send_wr* wr, ibv_send_wr** bad_wr = nullptr) {
        auto& lib = IbverbsLib::getInstance();
        PCCL_THROW_IF(lib.post_send(qp_, wr, bad_wr), 
               RuntimeException, "Failed to post send work request");
    }
    
    void post_recv(ibv_recv_wr* wr, ibv_recv_wr** bad_wr = nullptr) {
        auto& lib = IbverbsLib::getInstance();
        PCCL_THROW_IF(lib.post_recv(qp_, wr, bad_wr), 
               RuntimeException, "Failed to post receive work request");
    }
    
private:
    ibv_qp* qp_ = nullptr;
};

struct RemoteConnectionMetadata {
    uint32_t qkey;
    uint16_t lid;
    union ibv_gid gid;
    
    RemoteConnectionMetadata() : qkey(IBV_DEFAULT_QKEY), lid(0) {
        memset(&gid, 0, sizeof(gid));
    }
    
    RemoteConnectionMetadata(const RemoteConnectionMetadata& other)
        : qkey(other.qkey), lid(other.lid) {
        memcpy(&gid, &other.gid, sizeof(gid));
    }
    
    RemoteConnectionMetadata(RemoteConnectionMetadata&& other) noexcept
        : qkey(other.qkey), lid(other.lid) {
        memcpy(&gid, &other.gid, sizeof(gid));
        other.qkey = IBV_DEFAULT_QKEY;
        other.lid = 0;
        memset(&other.gid, 0, sizeof(gid));
    }
    
    RemoteConnectionMetadata& operator=(const RemoteConnectionMetadata& other) {
        if (this != &other) {
            qkey = other.qkey;
            lid = other.lid;
            memcpy(&gid, &other.gid, sizeof(gid));
        }
        return *this;
    }
    
    RemoteConnectionMetadata& operator=(RemoteConnectionMetadata&& other) noexcept {
        if (this != &other) {
            qkey = other.qkey;
            lid = other.lid;
            memcpy(&gid, &other.gid, sizeof(gid));
            
            other.qkey = IBV_DEFAULT_QKEY;
            other.lid = 0;
            memset(&other.gid, 0, sizeof(gid));
        }
        return *this;
    }
};

class RdmaManager {
public:
    struct ConnectionConfig {
        uint8_t port_num;
        int gid_index;
        int max_qp_per_connection;
        int cq_size;
        
        ConnectionConfig() : port_num(1), gid_index(0), max_qp_per_connection(1), cq_size(100) {}
        
        ConnectionConfig(const ConnectionConfig&) = default;
        
        ConnectionConfig(ConnectionConfig&&) = default;
        
        ConnectionConfig& operator=(const ConnectionConfig&) = default;
        
        ConnectionConfig& operator=(ConnectionConfig&&) = default;
    };

    struct QPConfig {
        ibv_qp_type qp_type;
        int max_send_wr;
        int max_recv_wr;
        int max_send_sge;
        int max_recv_sge;
        int max_inline_data;
        
        QPConfig() 
            : qp_type(IBV_QPT_RC), 
              max_send_wr(100), 
              max_recv_wr(100), 
              max_send_sge(1), 
              max_recv_sge(1), 
              max_inline_data(0) {}
        
        QPConfig(const QPConfig&) = default;
        
        QPConfig(QPConfig&&) = default;
        
        QPConfig& operator=(const QPConfig&) = default;
        
        QPConfig& operator=(QPConfig&&) = default;
    };

    using ConnectionId = uint64_t;
    using QPId = uint64_t;

    RdmaManager() {
        IbverbsLib::getInstance().load();
    }
    
    ~RdmaManager() {
        std::lock_guard<std::mutex> lock(connections_mutex_);
        for (auto& pair : connections_) {
            if (pair.second.connected) {
                disconnect(pair.first);
            }
            destroyConnection(pair.first);
        }
    }
    
    RdmaManager(const RdmaManager&) = delete;
    RdmaManager& operator=(const RdmaManager&) = delete;
    
    RdmaManager(RdmaManager&& other) noexcept 
        : device_list_(std::move(other.device_list_)),
          context_(std::move(other.context_)),
          pd_(std::move(other.pd_)),
          connections_(std::move(other.connections_)),
          next_connection_id_(other.next_connection_id_.load()),
          next_qp_id_(other.next_qp_id_.load()),
          initialized_(other.initialized_) {
        other.next_connection_id_ = 1;
        other.next_qp_id_ = 1;
        other.initialized_ = false;
    }
    
    RdmaManager& operator=(RdmaManager&& other) noexcept {
        if (this != &other) {
            std::lock_guard<std::mutex> lock(connections_mutex_);
            for (auto& pair : connections_) {
                if (pair.second.connected) {
                    disconnect(pair.first);
                }
                destroyConnection(pair.first);
            }
            
            device_list_ = std::move(other.device_list_);
            context_ = std::move(other.context_);
            pd_ = std::move(other.pd_);
            connections_ = std::move(other.connections_);
            next_connection_id_ = other.next_connection_id_.load();
            next_qp_id_ = other.next_qp_id_.load();
            initialized_ = other.initialized_;
            
            other.next_connection_id_ = 1;
            other.next_qp_id_ = 1;
            other.initialized_ = false;
        }
        return *this;
    }

    bool initialize(const std::string& device_name = "", uint8_t port_num = 1) {
        if (initialized_) {
            return true;
        }
        
        try {
            device_list_ = std::make_shared<RdmaDeviceList>();
            
            if (device_list_->count() == 0) {
                return false;
            }
            
            ibv_device* device = nullptr;
            if (device_name.empty()) {
                device = (*device_list_)[0];
            } else {
                for (int i = 0; i < device_list_->count(); ++i) {
                    if (device_list_->get_device_name(i) == device_name) {
                        device = (*device_list_)[i];
                        break;
                    }
                }
            }
            
            if (!device) {
                return false;
            }
            
            context_ = std::make_shared<RdmaContext>(device);
            pd_ = std::make_shared<ProtectionDomain>(context_->get());
            
            initialized_ = true;
            return true;
        } catch (const RuntimeException& e) {
            return false;
        }
    }

    ConnectionId createConnection(const ConnectionConfig& config) {
        if (!initialized_) {
            return 0;
        }
        
        ConnectionId conn_id = generateConnectionId();
        
        try {
            ConnectionInfo conn_info;
            conn_info.config = config;
            
            conn_info.cq = std::make_shared<CompletionQueue>(
                context_->get(), config.cq_size);
            
            conn_info.pd = pd_;
            conn_info.context = context_;
            
            std::lock_guard<std::mutex> lock(connections_mutex_);
            connections_[conn_id] = std::move(conn_info);
            
            return conn_id;
        } catch (const RuntimeException& e) {
            return 0;
        }
    }

    bool destroyConnection(ConnectionId conn_id) {
        std::lock_guard<std::mutex> lock(connections_mutex_);
        
        auto it = connections_.find(conn_id);
        if (it == connections_.end()) {
            return false;
        }
        
        if (it->second.connected) {
            disconnect(conn_id);
        }
        
        it->second.qps.clear();
        connections_.erase(it);
        return true;
    }

    bool connect(ConnectionId conn_id, const RemoteConnectionMetadata& remote_metadata) {
        std::lock_guard<std::mutex> lock(connections_mutex_);
        
        auto it = connections_.find(conn_id);
        if (it == connections_.end() || it->second.connected) {
            return false;
        }
        
        try {
            auto& conn_info = it->second;
            
            for (auto& qp_pair : conn_info.qps) {
                if (!modifyQPToInit(conn_id, qp_pair.first) ||
                    !modifyQPToRTR(conn_id, qp_pair.first, 
                          remote_metadata.qkey, 
                          remote_metadata.lid,
                          0,
                          conn_info.config.port_num,
                          0,
                          &remote_metadata.gid) ||
                    !modifyQPToRTS(conn_id, qp_pair.first)) {
                    disconnect(conn_id);
                    return false;
                }
            }
            
            conn_info.connected = true;
            return true;
        } catch (const RuntimeException& e) {
            return false;
        }
    }

    bool disconnect(ConnectionId conn_id) {
        std::lock_guard<std::mutex> lock(connections_mutex_);
        
        auto it = connections_.find(conn_id);
        if (it == connections_.end() || !it->second.connected) {
            return false;
        }
        
        it->second.connected = false;
        return true;
    }

    QPId createQP(ConnectionId conn_id, const QPConfig& config) {
        std::lock_guard<std::mutex> lock(connections_mutex_);
        
        auto it = connections_.find(conn_id);
        if (it == connections_.end()) {
            return 0;
        }
        
        if (it->second.qps.size() >= (size_t)it->second.config.max_qp_per_connection) {
            return 0;
        }
        
        try {
            ibv_qp_init_attr init_attr;
            memset(&init_attr, 0, sizeof(init_attr));
            init_attr.qp_type = config.qp_type;
            init_attr.sq_sig_all = 0;
            init_attr.send_cq = it->second.cq->get();
            init_attr.recv_cq = it->second.cq->get();
            init_attr.cap.max_send_wr = config.max_send_wr;
            init_attr.cap.max_recv_wr = config.max_recv_wr;
            init_attr.cap.max_send_sge = config.max_send_sge;
            init_attr.cap.max_recv_sge = config.max_recv_sge;
            init_attr.cap.max_inline_data = config.max_inline_data;
            
            QPId qp_id = generateQPId();
            auto qp = std::make_shared<QueuePair>(it->second.pd->get(), &init_attr);
            it->second.qps[qp_id] = qp;
            
            return qp_id;
        } catch (const RuntimeException& e) {
            return 0;
        }
    }

    bool destroyQP(ConnectionId conn_id, QPId qp_id) {
        std::lock_guard<std::mutex> lock(connections_mutex_);
        
        auto conn_it = connections_.find(conn_id);
        if (conn_it == connections_.end()) {
            return false;
        }
        
        auto qp_it = conn_it->second.qps.find(qp_id);
        if (qp_it == conn_it->second.qps.end()) {
            return false;
        }
        
        conn_it->second.qps.erase(qp_it);
        return true;
    }

    bool modifyQPToInit(ConnectionId conn_id, QPId qp_id) {
        auto qp = getQP(conn_id, qp_id);
        if (!qp) {
            return false;
        }
        
        ibv_qp_attr attr;
        memset(&attr, 0, sizeof(attr));
        attr.qp_state = IBV_QPS_INIT;
        attr.pkey_index = 0;
        attr.port_num = connections_[conn_id].config.port_num;
        attr.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | 
                   IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC;
        
        try {
            qp->modify(&attr, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS);
            return true;
        } catch (const RuntimeException& e) {
            return false;
        }
    }

    bool modifyQPToRTR(ConnectionId conn_id, QPId qp_id, 
               uint32_t remote_qpn, uint16_t dlid, uint8_t sl, 
               uint8_t port_num, uint16_t pkey_index, 
               const ibv_gid* sgid) {
        auto qp = getQP(conn_id, qp_id);
        if (!qp) {
            return false;
        }
        
        ibv_qp_attr attr;
        memset(&attr, 0, sizeof(attr));
        attr.qp_state = IBV_QPS_RTR;
        attr.path_mtu = IBV_MTU_1024;
        attr.dest_qp_num = remote_qpn;
        attr.rq_psn = 0;
        attr.max_dest_rd_atomic = 1;
        attr.min_rnr_timer = 12;
        attr.ah_attr.dlid = dlid;
        attr.ah_attr.sl = sl;
        attr.ah_attr.src_path_bits = 0;
        attr.ah_attr.port_num = port_num;
        attr.ah_attr.is_global = (sgid != nullptr) ? 1 : 0;
        
        if (sgid) {
            memcpy(&attr.ah_attr.grh.dgid, sgid, sizeof(ibv_gid));
            attr.ah_attr.grh.flow_label = 0;
            attr.ah_attr.grh.hop_limit = 1;
            attr.ah_attr.grh.sgid_index = connections_[conn_id].config.gid_index;
        }
        
        try {
            qp->modify(&attr, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | 
                   IBV_QP_DEST_QPN | IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | 
                   IBV_QP_MIN_RNR_TIMER);
            return true;
        } catch (const RuntimeException& e) {
            return false;
        }
    }

    bool modifyQPToRTS(ConnectionId conn_id, QPId qp_id) {
        auto qp = getQP(conn_id, qp_id);
        if (!qp) {
            return false;
        }
        
        ibv_qp_attr attr;
        memset(&attr, 0, sizeof(attr));
        attr.qp_state = IBV_QPS_RTS;
        attr.timeout = 14;
        attr.retry_cnt = 7;
        attr.rnr_retry = 7;
        attr.sq_psn = 0;
        attr.max_rd_atomic = 1;
        
        try {
            qp->modify(&attr, IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | 
                   IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC);
            return true;
        } catch (const RuntimeException& e) {
            return false;
        }
    }

    bool postSend(ConnectionId conn_id, QPId qp_id, ibv_send_wr* wr, ibv_send_wr** bad_wr) {
        auto qp = getQP(conn_id, qp_id);
        if (!qp) {
            return false;
        }
        
        try {
            qp->post_send(wr, bad_wr);
            return true;
        } catch (const RuntimeException& e) {
            return false;
        }
    }

    bool postRecv(ConnectionId conn_id, QPId qp_id, ibv_recv_wr* wr, ibv_recv_wr** bad_wr) {
        auto qp = getQP(conn_id, qp_id);
        if (!qp) {
            return false;
        }
        
        try {
            qp->post_recv(wr, bad_wr);
            return true;
        } catch (const RuntimeException& e) {
            return false;
        }
    }

    int pollCQ(ConnectionId conn_id, int num_entries, ibv_wc* wc) {
        auto cq = getCQ(conn_id);
        if (!cq) {
            return -1;
        }
        
        return cq->poll(wc, num_entries);
    }

    std::shared_ptr<QueuePair> getQP(ConnectionId conn_id, QPId qp_id) {
        std::lock_guard<std::mutex> lock(connections_mutex_);
        
        auto conn_it = connections_.find(conn_id);
        if (conn_it == connections_.end()) {
            return nullptr;
        }
        
        auto qp_it = conn_it->second.qps.find(qp_id);
        if (qp_it == conn_it->second.qps.end()) {
            return nullptr;
        }
        
        return qp_it->second;
    }

    std::shared_ptr<CompletionQueue> getCQ(ConnectionId conn_id) {
        std::lock_guard<std::mutex> lock(connections_mutex_);
        
        auto it = connections_.find(conn_id);
        if (it == connections_.end()) {
            return nullptr;
        }
        
        return it->second.cq;
    }

    std::shared_ptr<ProtectionDomain> getPD() {
        return pd_;
    }

    bool isConnected(ConnectionId conn_id) const {
        std::lock_guard<std::mutex> lock(connections_mutex_);
        
        auto it = connections_.find(conn_id);
        if (it == connections_.end()) {
            return false;
        }
        
        return it->second.connected;
    }

    size_t getConnectionCount() const {
        std::lock_guard<std::mutex> lock(connections_mutex_);
        return connections_.size();
    }

    size_t getQPCount(ConnectionId conn_id) const {
        std::lock_guard<std::mutex> lock(connections_mutex_);
        
        auto it = connections_.find(conn_id);
        if (it == connections_.end()) {
            return 0;
        }
        
        return it->second.qps.size();
    }

    RemoteConnectionMetadata getLocalMetadata(ConnectionId conn_id) {
        std::lock_guard<std::mutex> lock(connections_mutex_);
        
        auto it = connections_.find(conn_id);
        if (it == connections_.end()) {
            return RemoteConnectionMetadata{};
        }
        
        RemoteConnectionMetadata metadata;
        metadata.qkey = IBV_DEFAULT_QKEY;
        
        try {
            ibv_port_attr port_attr = context_->query_port(it->second.config.port_num);
            metadata.lid = port_attr.lid;
            
            if (ibv_query_gid(context_->get(), it->second.config.port_num, 
                     it->second.config.gid_index, &metadata.gid) != 0) {
                memset(&metadata.gid, 0, sizeof(ibv_gid));
            }
        } catch (const RuntimeException& e) {
            memset(&metadata.gid, 0, sizeof(ibv_gid));
        }
        
        return metadata;
    }

private:
    struct ConnectionInfo {
        ConnectionConfig config;
        std::shared_ptr<RdmaContext> context;
        std::shared_ptr<ProtectionDomain> pd;
        std::shared_ptr<CompletionQueue> cq;
        std::unordered_map<QPId, std::shared_ptr<QueuePair>> qps;
        std::atomic<bool> connected{false};
        
        ConnectionInfo() = default;
        ConnectionInfo(const ConnectionInfo& other)
            : config(other.config),
              context(other.context),
              pd(other.pd),
              cq(other.cq),
              connected(other.connected.load()) {
            for (const auto& pair : other.qps) {
                qps[pair.first] = pair.second;
            }
        }
        
        ConnectionInfo(ConnectionInfo&& other) noexcept
            : config(std::move(other.config)),
              context(std::move(other.context)),
              pd(std::move(other.pd)),
              cq(std::move(other.cq)),
              qps(std::move(other.qps)),
              connected(other.connected.load()) {
            other.connected = false;
        }
        
        ConnectionInfo& operator=(const ConnectionInfo& other) {
            if (this != &other) {
                config = other.config;
                context = other.context;
                pd = other.pd;
                cq = other.cq;
                connected = other.connected.load();
                
                qps.clear();
                for (const auto& pair : other.qps) {
                    qps[pair.first] = pair.second;
                }
            }
            return *this;
        }
        
        ConnectionInfo& operator=(ConnectionInfo&& other) noexcept {
            if (this != &other) {
                config = std::move(other.config);
                context = std::move(other.context);
                pd = std::move(other.pd);
                cq = std::move(other.cq);
                qps = std::move(other.qps);
                connected = other.connected.load();
                
                other.connected = false;
            }
            return *this;
        }
    };

    ConnectionId generateConnectionId() {
        return next_connection_id_++;
    }

    QPId generateQPId() {
        return next_qp_id_++;
    }

    std::shared_ptr<RdmaDeviceList> device_list_;
    std::shared_ptr<RdmaContext> context_;
    std::shared_ptr<ProtectionDomain> pd_;
    
    std::unordered_map<ConnectionId, ConnectionInfo> connections_;
    mutable std::mutex connections_mutex_;
    
    std::atomic<ConnectionId> next_connection_id_{1};
    std::atomic<QPId> next_qp_id_{1};
    
    bool initialized_{false};
};

} // namespace pccl