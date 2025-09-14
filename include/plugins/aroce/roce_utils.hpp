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
#include <format>
#include "configs/verbs_config.h"
#include "utils/exception.hpp"
#include "utils/logging.h"

namespace pccl {

class VerbsLib {
public:
  static VerbsLib& getInstance() {
    static VerbsLib instance;
    return instance;
  }

  void load(const std::string& library_path) {
    std::lock_guard<std::mutex> lock(load_mutex_);
    if (handle_) {
      PCCL_DLOG_DEBUG("libibverbs already loaded. Skipping.");
      return;
    }
    PCCL_DLOG_DEBUG(std::format("Attempting to load {}", library_path));
    handle_ = dlopen(library_path.c_str(), RTLD_LAZY);
    PCCL_DLOG_DEBUG(std::format("dlopen handle: {}", handle_));

    PCCL_THROW_IF(!handle_, RuntimeException, std::format("Failed to open {}", library_path));

    #define LOAD_VERBS_FUNC(name) \
      PCCL_DLOG_DEBUG(std::format("Loading function {}...", #name)); \
      name##_func = reinterpret_cast<name##_fn>(dlsym(handle_, #name)); \
      PCCL_DLOG_DEBUG(std::format("{} address: {}", #name, (void*)name##_func)); \
      PCCL_THROW_IF(!name##_func, RuntimeException, std::format("Failed to load function {}", #name));

    LOAD_VERBS_FUNC(ibv_get_device_list)
    LOAD_VERBS_FUNC(ibv_free_device_list)
    LOAD_VERBS_FUNC(ibv_get_device_name)
    LOAD_VERBS_FUNC(ibv_open_device)
    LOAD_VERBS_FUNC(ibv_close_device)
    LOAD_VERBS_FUNC(ibv_query_device)
    LOAD_VERBS_FUNC(ibv_query_port)
    LOAD_VERBS_FUNC(ibv_query_gid)
    LOAD_VERBS_FUNC(ibv_alloc_pd)
    LOAD_VERBS_FUNC(ibv_dealloc_pd)
    LOAD_VERBS_FUNC(ibv_reg_mr)
    LOAD_VERBS_FUNC(ibv_dereg_mr)
    LOAD_VERBS_FUNC(ibv_create_comp_channel)
    LOAD_VERBS_FUNC(ibv_destroy_comp_channel)
    LOAD_VERBS_FUNC(ibv_create_cq)
    LOAD_VERBS_FUNC(ibv_destroy_cq)
    LOAD_VERBS_FUNC(ibv_get_cq_event)
    LOAD_VERBS_FUNC(ibv_ack_cq_events)
    LOAD_VERBS_FUNC(ibv_create_qp)
    LOAD_VERBS_FUNC(ibv_destroy_qp)
    LOAD_VERBS_FUNC(ibv_modify_qp)
    LOAD_VERBS_FUNC(ibv_wc_status_str)
    #undef LOAD_VERBS_FUNC
    PCCL_DLOG_DEBUG("Successfully loaded all required ibverbs functions.");
  }

  ~VerbsLib() {
    std::lock_guard<std::mutex> lock(load_mutex_);
    if (handle_) {
      PCCL_DLOG_DEBUG(std::format("Unloading libibverbs.so, handle: {}", handle_));
      dlclose(handle_);
      handle_ = nullptr;
    }
  }

  VerbsLib(const VerbsLib&) = delete;
  VerbsLib& operator=(const VerbsLib&) = delete;

  ibv_device** getDeviceList(int* num_devices) { return ibv_get_device_list_func(num_devices); }
  int freeDeviceList(ibv_device** list) { return ibv_free_device_list_func(list); }
  const char* getDeviceName(ibv_device* device) { return ibv_get_device_name_func(device); }
  ibv_context* openDevice(ibv_device* device) { return ibv_open_device_func(device); }
  int closeDevice(ibv_context* context) { return ibv_close_device_func(context); }
  int queryDevice(ibv_context* context, ibv_device_attr* device_attr) { return ibv_query_device_func(context, device_attr); }
  int queryPort(ibv_context* context, uint8_t port_num, ibv_port_attr* port_attr) { return ibv_query_port_func(context, port_num, port_attr); }
  int queryGid(ibv_context* context, uint8_t port_num, int index, ibv_gid* gid) { return ibv_query_gid_func(context, port_num, index, gid); }
  ibv_pd* allocPd(ibv_context* context) { return ibv_alloc_pd_func(context); }
  int deallocPd(ibv_pd* pd) { return ibv_dealloc_pd_func(pd); }
  ibv_mr* regMr(ibv_pd* pd, void* addr, size_t length, int access) { return ibv_reg_mr_func(pd, addr, length, access); }
  int deregMr(ibv_mr* mr) { return ibv_dereg_mr_func(mr); }
  ibv_comp_channel* createCompChannel(ibv_context* context) { return ibv_create_comp_channel_func(context); }
  int destroyCompChannel(ibv_comp_channel* channel) { return ibv_destroy_comp_channel_func(channel); }
  int getCqEvent(ibv_comp_channel* channel, ibv_cq** cq, void** cq_context) { return ibv_get_cq_event_func(channel, cq, cq_context); }
  void ackCqEvents(ibv_cq* cq, unsigned int nevents) { ibv_ack_cq_events_func(cq, nevents); }
  ibv_cq* createCq(ibv_context* context, int cqe, void* cq_context, ibv_comp_channel* channel, int comp_vector) { return ibv_create_cq_func(context, cqe, cq_context, channel, comp_vector); }
  int destroyCq(ibv_cq* cq) { return ibv_destroy_cq_func(cq); }
  int pollCq(ibv_cq* cq, int num_entries, ibv_wc* wc) { return ibv_poll_cq(cq, num_entries, wc); }
  ibv_qp* createQp(ibv_pd* pd, ibv_qp_init_attr* attr) { return ibv_create_qp_func(pd, attr); }
  int destroyQp(ibv_qp* qp) { return ibv_destroy_qp_func(qp); }
  int modifyQp(ibv_qp* qp, ibv_qp_attr* attr, int attr_mask) { return ibv_modify_qp_func(qp, attr, attr_mask); }
  int postSend(ibv_qp* qp, ibv_send_wr* wr, ibv_send_wr** bad_wr) { return ibv_post_send(qp, wr, bad_wr); }
  int postRecv(ibv_qp* qp, ibv_recv_wr* wr, ibv_recv_wr** bad_wr) { return ibv_post_recv(qp, wr, bad_wr); }
  int reqNotifyCq(ibv_cq* cq, int solicited_only) { return ibv_req_notify_cq(cq, solicited_only); }
  const char* wcStatusStr(enum ibv_wc_status status) { return ibv_wc_status_str_func(status); }

private:
  typedef ibv_device** (*ibv_get_device_list_fn)(int*);
  typedef int (*ibv_free_device_list_fn)(ibv_device**);
  typedef const char* (*ibv_get_device_name_fn)(ibv_device*);
  typedef ibv_context* (*ibv_open_device_fn)(ibv_device*);
  typedef int (*ibv_close_device_fn)(ibv_context*);
  typedef int (*ibv_query_device_fn)(ibv_context*, ibv_device_attr*);
  typedef int (*ibv_query_port_fn)(ibv_context*, uint8_t, ibv_port_attr*);
  typedef int (*ibv_query_gid_fn)(ibv_context*, uint8_t, int, ibv_gid*);
  typedef ibv_pd* (*ibv_alloc_pd_fn)(ibv_context*);
  typedef int (*ibv_dealloc_pd_fn)(ibv_pd*);
  typedef ibv_mr* (*ibv_reg_mr_fn)(ibv_pd*, void*, size_t, int);
  typedef int (*ibv_dereg_mr_fn)(ibv_mr*);
  typedef ibv_comp_channel* (*ibv_create_comp_channel_fn)(ibv_context*);
  typedef int (*ibv_destroy_comp_channel_fn)(ibv_comp_channel*);
  typedef ibv_cq* (*ibv_create_cq_fn)(ibv_context*, int, void*, ibv_comp_channel*, int);
  typedef int (*ibv_destroy_cq_fn)(ibv_cq*);
  typedef int (*ibv_get_cq_event_fn)(ibv_comp_channel*, ibv_cq**, void**);
  typedef void (*ibv_ack_cq_events_fn)(ibv_cq*, unsigned int);
  typedef ibv_qp* (*ibv_create_qp_fn)(ibv_pd*, ibv_qp_init_attr*);
  typedef int (*ibv_destroy_qp_fn)(ibv_qp*);
  typedef int (*ibv_modify_qp_fn)(ibv_qp*, ibv_qp_attr*, int);
  typedef const char* (*ibv_wc_status_str_fn)(enum ibv_wc_status);

  ibv_get_device_list_fn ibv_get_device_list_func = nullptr;
  ibv_free_device_list_fn ibv_free_device_list_func = nullptr;
  ibv_get_device_name_fn ibv_get_device_name_func = nullptr;
  ibv_open_device_fn ibv_open_device_func = nullptr;
  ibv_close_device_fn ibv_close_device_func = nullptr;
  ibv_query_device_fn ibv_query_device_func = nullptr;
  ibv_query_port_fn ibv_query_port_func = nullptr;
  ibv_query_gid_fn ibv_query_gid_func = nullptr;
  ibv_alloc_pd_fn ibv_alloc_pd_func = nullptr;
  ibv_dealloc_pd_fn ibv_dealloc_pd_func = nullptr;
  ibv_reg_mr_fn ibv_reg_mr_func = nullptr;
  ibv_dereg_mr_fn ibv_dereg_mr_func = nullptr;
  ibv_create_comp_channel_fn ibv_create_comp_channel_func = nullptr;
  ibv_destroy_comp_channel_fn ibv_destroy_comp_channel_func = nullptr;
  ibv_create_cq_fn ibv_create_cq_func = nullptr;
  ibv_destroy_cq_fn ibv_destroy_cq_func = nullptr;
  ibv_get_cq_event_fn ibv_get_cq_event_func = nullptr;
  ibv_ack_cq_events_fn ibv_ack_cq_events_func = nullptr;
  ibv_create_qp_fn ibv_create_qp_func = nullptr;
  ibv_destroy_qp_fn ibv_destroy_qp_func = nullptr;
  ibv_modify_qp_fn ibv_modify_qp_func = nullptr;
  ibv_wc_status_str_fn ibv_wc_status_str_func = nullptr;

  VerbsLib() = default;
  void* handle_ = nullptr;
  std::mutex load_mutex_;
};

class VerbsDevList {
public:
  VerbsDevList() {
    auto& verbs_lib = VerbsLib::getInstance();
    devices_ = verbs_lib.getDeviceList(&num_devices_);
    PCCL_THROW_IF(!devices_, RuntimeException, "Failed to get IB devices");
  }

  ~VerbsDevList() {
    if (devices_) {
      VerbsLib::getInstance().freeDeviceList(devices_);
    }
  }

  VerbsDevList(const VerbsDevList&) = delete;
  VerbsDevList& operator=(const VerbsDevList&) = delete;
  VerbsDevList(VerbsDevList&& other) noexcept 
    : devices_(other.devices_), num_devices_(other.num_devices_) {
    other.devices_ = nullptr;
    other.num_devices_ = 0;
  }

  VerbsDevList& operator=(VerbsDevList&& other) noexcept {
    if (this != &other) {
      if (devices_) VerbsLib::getInstance().freeDeviceList(devices_);
      devices_ = other.devices_;
      num_devices_ = other.num_devices_;
      other.devices_ = nullptr;
      other.num_devices_ = 0;
    }
    return *this;
  }

  ibv_device* getDevice(int index) const {
    PCCL_THROW_IF(index < 0 || index >= num_devices_, 
      RuntimeException, std::format("Device index out of range: {}", index));
    return devices_[index];
  }

  std::string getDeviceName(int index) {
    ibv_device* dev = getDevice(index);
    return VerbsLib::getInstance().getDeviceName(dev);
  }

  int getNumDevices() const { return num_devices_; }

private:
  ibv_device** devices_ = nullptr;
  int num_devices_ = 0;
};

class VerbsContext {
public:
  explicit VerbsContext(ibv_device* device) {
    auto& verbs_lib = VerbsLib::getInstance();
    context_ = verbs_lib.openDevice(device);
    PCCL_THROW_IF(!context_, RuntimeException, 
      std::format("Failed to open device: {}", verbs_lib.getDeviceName(device)));
  }

  ~VerbsContext() {
    if (context_) VerbsLib::getInstance().closeDevice(context_);
  }

  VerbsContext(const VerbsContext&) = delete;
  VerbsContext& operator=(const VerbsContext&) = delete;
  VerbsContext(VerbsContext&& other) noexcept : context_(other.context_) {
    other.context_ = nullptr;
  }
  VerbsContext& operator=(VerbsContext&& other) noexcept {
    if (this != &other) {
      if (context_) VerbsLib::getInstance().closeDevice(context_);
      context_ = other.context_;
      other.context_ = nullptr;
    }
    return *this;
  }

  ibv_device_attr queryDevice() const {
    ibv_device_attr attr;
    PCCL_THROW_IF(VerbsLib::getInstance().queryDevice(context_, &attr) != 0,
      RuntimeException, "Failed to query device attributes");
    return attr;
  }

  ibv_port_attr queryPort(uint8_t port_num) const {
    ibv_port_attr attr;
    PCCL_THROW_IF(VerbsLib::getInstance().queryPort(context_, port_num, &attr) != 0,
      RuntimeException, std::format("Failed to query port {} attributes", port_num));
    return attr;
  }

  ibv_gid queryGid(uint8_t port_num, int index) const {
    ibv_gid gid;
    PCCL_THROW_IF(VerbsLib::getInstance().queryGid(context_, port_num, index, &gid) != 0,
      RuntimeException, std::format("Failed to query GID (port {}, index {})", port_num, index));
    return gid;
  }

  ibv_context* get() const { return context_; }

private:
  ibv_context* context_ = nullptr;
};

class VerbsProtectionDomain {
public:
  explicit VerbsProtectionDomain(const VerbsContext& ctx) {
    pd_ = VerbsLib::getInstance().allocPd(ctx.get());
    PCCL_THROW_IF(!pd_, RuntimeException, "Failed to allocate PD");
  }

  ~VerbsProtectionDomain() {
    if (pd_) VerbsLib::getInstance().deallocPd(pd_);
  }

  VerbsProtectionDomain(const VerbsProtectionDomain&) = delete;
  VerbsProtectionDomain& operator=(const VerbsProtectionDomain&) = delete;
  VerbsProtectionDomain(VerbsProtectionDomain&& other) noexcept : pd_(other.pd_) {
    other.pd_ = nullptr;
  }
  VerbsProtectionDomain& operator=(VerbsProtectionDomain&& other) noexcept {
    if (this != &other) {
      if (pd_) VerbsLib::getInstance().deallocPd(pd_);
      pd_ = other.pd_;
      other.pd_ = nullptr;
    }
    return *this;
  }

  ibv_pd* get() const { return pd_; }

private:
  ibv_pd* pd_ = nullptr;
};

class VerbsCompChannel {
public:
  explicit VerbsCompChannel(const VerbsContext& ctx) {
    channel_ = VerbsLib::getInstance().createCompChannel(ctx.get());
    PCCL_THROW_IF(!channel_, RuntimeException, "Failed to create comp channel");
  }

  ~VerbsCompChannel() {
    if (channel_) VerbsLib::getInstance().destroyCompChannel(channel_);
  }

  VerbsCompChannel(const VerbsCompChannel&) = delete;
  VerbsCompChannel& operator=(const VerbsCompChannel&) = delete;
  VerbsCompChannel(VerbsCompChannel&& other) noexcept : channel_(other.channel_) {
    other.channel_ = nullptr;
  }
  VerbsCompChannel& operator=(VerbsCompChannel&& other) noexcept {
    if (this != &other) {
      if (channel_) VerbsLib::getInstance().destroyCompChannel(channel_);
      channel_ = other.channel_;
      other.channel_ = nullptr;
    }
    return *this;
  }

  ibv_comp_channel* get() const { return channel_; }

private:
  ibv_comp_channel* channel_ = nullptr;
};

class VerbsCompletionQueue {
public:
  VerbsCompletionQueue(const VerbsContext& ctx, int cqe, void* cq_ctx,
                        const VerbsCompChannel* channel = nullptr, int comp_vec = 0) {
    cq_ = VerbsLib::getInstance().createCq(
      ctx.get(), cqe, cq_ctx, channel ? channel->get() : nullptr, comp_vec
    );
    PCCL_THROW_IF(!cq_, RuntimeException, "Failed to create CQ");
  }

  ~VerbsCompletionQueue() {
    if (cq_) VerbsLib::getInstance().destroyCq(cq_);
  }

  VerbsCompletionQueue(const VerbsCompletionQueue&) = delete;
  VerbsCompletionQueue& operator=(const VerbsCompletionQueue&) = delete;
  VerbsCompletionQueue(VerbsCompletionQueue&& other) noexcept : cq_(other.cq_) {
    other.cq_ = nullptr;
  }
  VerbsCompletionQueue& operator=(VerbsCompletionQueue&& other) noexcept {
    if (this != &other) {
      if (cq_) VerbsLib::getInstance().destroyCq(cq_);
      cq_ = other.cq_;
      other.cq_ = nullptr;
    }
    return *this;
  }

  void getEvent(const VerbsCompChannel& channel, VerbsCompletionQueue*& out_cq, void*& out_ctx) {
    ibv_cq* cq = nullptr;
    PCCL_THROW_IF(VerbsLib::getInstance().getCqEvent(channel.get(), &cq, &out_ctx) != 0,
      RuntimeException, "Failed to get CQ event");
    out_cq = static_cast<VerbsCompletionQueue*>(out_ctx); // 需将cq_ctx设为this指针
  }

  int poll(int num_entries, ibv_wc* wc) const {
    return VerbsLib::getInstance().pollCq(cq_, num_entries, wc);
  }

  void ackEvents(unsigned int nevents) const {
    VerbsLib::getInstance().ackCqEvents(cq_, nevents);
  }

  ibv_cq* get() const { return cq_; }

private:
  ibv_cq* cq_ = nullptr;
};

class VerbsMemoryRegion {
public:
  VerbsMemoryRegion(const VerbsProtectionDomain& pd, void* addr, size_t len, int access)
    : addr_(addr), len_(len) {
    mr_ = VerbsLib::getInstance().regMr(pd.get(), addr, len, access);
    PCCL_THROW_IF(!mr_, RuntimeException, "Failed to register MR");
    lkey_ = mr_->lkey;
    rkey_ = mr_->rkey;
  }

  ~VerbsMemoryRegion() {
    if (mr_) VerbsLib::getInstance().deregMr(mr_);
  }

  VerbsMemoryRegion(const VerbsMemoryRegion&) = delete;
  VerbsMemoryRegion& operator=(const VerbsMemoryRegion&) = delete;
  VerbsMemoryRegion(VerbsMemoryRegion&& other) noexcept 
    : mr_(other.mr_), addr_(other.addr_), len_(other.len_), lkey_(other.lkey_), rkey_(other.rkey_) {
    other.mr_ = nullptr;
  }
  VerbsMemoryRegion& operator=(VerbsMemoryRegion&& other) noexcept {
    if (this != &other) {
      if (mr_) VerbsLib::getInstance().deregMr(mr_);
      mr_ = other.mr_;
      addr_ = other.addr_;
      len_ = other.len_;
      lkey_ = other.lkey_;
      rkey_ = other.rkey_;
      other.mr_ = nullptr;
    }
    return *this;
  }

  void* getAddr() const { return addr_; }
  size_t getLength() const { return len_; }
  uint32_t getLKey() const { return lkey_; }
  uint32_t getRKey() const { return rkey_; }
  ibv_mr* get() const { return mr_; }

private:
  ibv_mr* mr_ = nullptr;
  void* addr_ = nullptr;
  size_t len_ = 0;
  uint32_t lkey_ = 0;
  uint32_t rkey_ = 0;
};

class VerbsQueuePair {
public:
  VerbsQueuePair(const VerbsProtectionDomain& pd, const ibv_qp_init_attr& init_attr) {
    qp_ = VerbsLib::getInstance().createQp(pd.get(), const_cast<ibv_qp_init_attr*>(&init_attr));
    PCCL_THROW_IF(!qp_, RuntimeException, "Failed to create QP");
    qp_num_ = qp_->qp_num;
  }

  ~VerbsQueuePair() {
    if (qp_) VerbsLib::getInstance().destroyQp(qp_);
  }

  VerbsQueuePair(const VerbsQueuePair&) = delete;
  VerbsQueuePair& operator=(const VerbsQueuePair&) = delete;
  VerbsQueuePair(VerbsQueuePair&& other) noexcept 
    : qp_(other.qp_), qp_num_(other.qp_num_) {
    other.qp_ = nullptr;
  }

  VerbsQueuePair& operator=(VerbsQueuePair&& other) noexcept {
    if (this != &other) {
      if (qp_) VerbsLib::getInstance().destroyQp(qp_);
      qp_ = other.qp_;
      qp_num_ = other.qp_num_;
      other.qp_ = nullptr;
    }
    return *this;
  }
  
  int modify(const ibv_qp_attr& attr, int attr_mask) const {
    return VerbsLib::getInstance().modifyQp(qp_, const_cast<ibv_qp_attr*>(&attr), attr_mask);
  }

  int postSend(ibv_send_wr* wr, ibv_send_wr** bad_wr) const {
    return VerbsLib::getInstance().postSend(qp_, wr, bad_wr);
  }

  int postRecv(ibv_recv_wr* wr, ibv_recv_wr** bad_wr) const {
    return VerbsLib::getInstance().postRecv(qp_, wr, bad_wr);
  }

  uint32_t getQpNum() const { return qp_num_; }
  ibv_qp* get() const { return qp_; }

private:
  ibv_qp* qp_ = nullptr;
  uint32_t qp_num_ = 0;
};

struct VerbsRemotePeerInfo {
  uint32_t qp_num;
  uint16_t lid;
  union ibv_gid gid;
    
  VerbsRemotePeerInfo() : qp_num(0), lid(0) {
    std::memset(&gid, 0, sizeof(gid));
  }
    
  VerbsRemotePeerInfo(const VerbsRemotePeerInfo& other)
    : qp_num(other.qp_num), lid(other.lid) {
    std::memcpy(&gid, &other.gid, sizeof(gid));
  }
    
  VerbsRemotePeerInfo(VerbsRemotePeerInfo&& other) noexcept
    : qp_num(other.qp_num), lid(other.lid) {
    std::memcpy(&gid, &other.gid, sizeof(gid));
    other.qp_num = 0;
    other.lid = 0;
    std::memset(&other.gid, 0, sizeof(gid));
  }
    
  VerbsRemotePeerInfo& operator=(const VerbsRemotePeerInfo& other) {
    if (this != &other) {
      qp_num = other.qp_num;
      lid = other.lid;
      std::memcpy(&gid, &other.gid, sizeof(gid));
    }
    return *this;
  }
    
  VerbsRemotePeerInfo& operator=(VerbsRemotePeerInfo&& other) noexcept {
    if (this != &other) {
      qp_num = other.qp_num;
      lid = other.lid;
      std::memcpy(&gid, &other.gid, sizeof(gid));
      other.qp_num = 0;
      other.lid = 0;
      std::memset(&other.gid, 0, sizeof(gid));
    }
    return *this;
  }
};

class VerbsManager {
public:
  using ConnectionId = uint64_t;
  using QPId = uint64_t;

  struct QPConfig {
    ibv_qp_type qp_type;
    int max_send_wr;
    int max_recv_wr;
    int max_send_sge;
    int max_recv_sge;
    int max_inline_data;

    QPConfig() 
      : qp_type(IBV_QPT_RC), 
        max_send_wr(VerbsConfig::getInstance().max_send_wr), 
        max_recv_wr(VerbsConfig::getInstance().max_recv_wr), 
        max_send_sge(VerbsConfig::getInstance().max_send_sge), 
        max_recv_sge(VerbsConfig::getInstance().max_recv_sge), 
        max_inline_data(VerbsConfig::getInstance().max_inline_data) {}
    QPConfig(const QPConfig&) = default;
    QPConfig(QPConfig&&) = default;
    QPConfig& operator=(const QPConfig&) = default;
    QPConfig& operator=(QPConfig&&) = default;
  };

  struct ConnectionConfig {
    int port_num;
    int gid_index;
    int max_qp_per_connection;
    int cq_size;
  };

  VerbsManager() {
    PCCL_DLOG_INFO("VerbsManager creating...");
    VerbsLib::getInstance().load(VerbsConfig::getInstance().lib_path);
    PCCL_DLOG_INFO("VerbsManager created");
  }

  ~VerbsManager() {
    PCCL_DLOG_INFO("Destroying VerbsManager...");
    std::lock_guard<std::mutex> lock(connections_mutex_);
    for (auto& pair : connections_) {
      if (pair.second.connected) {
        disconnect(pair.first);
      }
    }
    connections_.clear();
    PCCL_DLOG_INFO("All connections cleared.");
  }

  VerbsManager(const VerbsManager&) = delete;
  VerbsManager& operator=(const VerbsManager&) = delete;

  VerbsManager(VerbsManager&& other) noexcept 
    : device_list_(std::move(other.device_list_)),
      context_(std::move(other.context_)),
      pd_(std::move(other.pd_)),
      connections_(std::move(other.connections_)),
      next_connection_id_(other.next_connection_id_.load()),
      next_qp_id_(other.next_qp_id_.load()),
      initialized_(other.initialized_.load()) {
    PCCL_DLOG_INFO(std::format("Moving VerbsManager from {} to {}", (void*)&other, (void*)this));
    other.next_connection_id_ = 1;
    other.next_qp_id_ = 1;
    other.initialized_ = false;
  }

  VerbsManager& operator=(VerbsManager&& other) noexcept {
    PCCL_DLOG_DEBUG(std::format("Move-assigning VerbsManager from {} to {}", (void*)&other, (void*)this));
    if (this != &other) {
      std::lock_guard<std::mutex> lock(connections_mutex_);
      for (auto& pair : connections_) {
        if (pair.second.connected) {
          disconnect(pair.first);
        }
      }
      connections_.clear();
      
      device_list_ = std::move(other.device_list_);
      context_ = std::move(other.context_);
      pd_ = std::move(other.pd_);
      connections_ = std::move(other.connections_);
      next_connection_id_ = other.next_connection_id_.load();
      next_qp_id_ = other.next_qp_id_.load();
      initialized_ = other.initialized_.load();
      
      other.next_connection_id_ = 1;
      other.next_qp_id_ = 1;
      other.initialized_ = false;
    }
    return *this;
  }

  bool initialize(const std::string& device_name = "", uint8_t port_num = 1) {
    PCCL_DLOG_INFO("Initializing VerbsManager...");
    device_list_ = std::make_unique<VerbsDevList>();
    if (device_list_->getNumDevices() == 0) {
      PCCL_DLOG_INFO("No RDMA devices found. Initialization failed.");
      return false;
    }
    PCCL_DLOG_DEBUG(std::format("Device list count: {}", device_list_->getNumDevices()));
    ibv_device* device = nullptr;
    if (device_name.empty()) {
      device = device_list_->getDevice(0);
      PCCL_DLOG_DEBUG(std::format("No device name provided, selecting first device: {}", device_list_->getDeviceName(0)));
    } else {
      PCCL_DLOG_DEBUG(std::format("Searching for device: '{}'", device_name));
      for (int i = 0; i < device_list_->getNumDevices(); ++i) {
        if (device_list_->getDeviceName(i) == device_name) {
          device = device_list_->getDevice(i);
          PCCL_DLOG_DEBUG(std::format("Found and selected device: {}", device_list_->getDeviceName(i)));
          break;
        }
      }
    }
      
    if (!device) {
      PCCL_DLOG_DEBUG(std::format("Device '{}' not found. Initialization failed.", device_name));
      return false;
    }
      
    context_ = std::make_unique<VerbsContext>(device);
    pd_ = std::make_unique<VerbsProtectionDomain>(*context_);
      
    initialized_ = true;
    PCCL_DLOG_DEBUG("VerbsManager initialized successfully.");
    return true;
  }

  ConnectionId createConnection(const ConnectionConfig& config) {
    PCCL_DLOG_DEBUG("Attempting to create a new connection...");
    if (!initialized_.load(std::memory_order_relaxed)) {
      PCCL_DLOG_DEBUG("Manager not initialized. Failed.");
      return 0;
    }
        
    ConnectionId conn_id = generateConnectionId();
    PCCL_DLOG_DEBUG(std::format("Generated new ConnectionId: {}", conn_id));
        
    ConnectionInfo conn_info;
    conn_info.config = config;
    conn_info.cq = std::make_shared<VerbsCompletionQueue>(*context_, config.cq_size, nullptr);
    
    std::lock_guard<std::mutex> lock(connections_mutex_);
    connections_[conn_id] = std::move(conn_info);
    
    PCCL_DLOG_DEBUG(std::format("Connection {} created successfully.", conn_id));
    return conn_id;
  }

  bool destroyConnection(ConnectionId conn_id) {
    PCCL_DLOG_DEBUG(std::format("Destroying connection {}", conn_id));
    std::lock_guard<std::mutex> lock(connections_mutex_);
    auto it = connections_.find(conn_id);
    if (it == connections_.end()) {
      PCCL_DLOG_DEBUG(std::format("Connection {} not found", conn_id));
      return false;
    }
    
    if (it->second.connected) {
      PCCL_DLOG_DEBUG(std::format("Connection {} is active, disconnecting first.", conn_id));
      disconnect(conn_id);
    }
    
    it->second.qps.clear();
    connections_.erase(it);
    PCCL_DLOG_DEBUG(std::format("Connection {} destroyed.", conn_id));
    return true;
  }

  bool connect(ConnectionId conn_id, const VerbsRemotePeerInfo& remote_peer_info) {
    PCCL_DLOG_DEBUG(std::format("Attempting to connect conn_id={} with remote_qp_num={}, remote_lid={}", conn_id, remote_peer_info.qp_num, remote_peer_info.lid));
    std::lock_guard<std::mutex> lock(connections_mutex_);
    auto it = connections_.find(conn_id);
    if (it == connections_.end()) {
      PCCL_DLOG_DEBUG(std::format("Connection {} not found. Failed.", conn_id));
      return false;
    }
    if (it->second.connected) {
      PCCL_DLOG_DEBUG(std::format("Connection {} already connected. Failed.", conn_id));
      return false;
    }

    try {
      auto& conn_info = it->second;
      if (conn_info.qps.empty()) {
        PCCL_DLOG_DEBUG(std::format("No QPs created for connection {}. Cannot connect.", conn_id));
        return false;
      }

      for (auto& qp_pair : conn_info.qps) {
        QPId qp_id = qp_pair.first;
        PCCL_DLOG_DEBUG(std::format("Processing QP {} for connection {}", qp_id, conn_id));
        
        if (!modifyQPToInit(conn_id, qp_id)) {
          PCCL_DLOG_DEBUG(std::format("Failed to modify QP {} to INIT state.", qp_id));
          return false;
        }
        
        if (!modifyQPToRTR(conn_id, qp_id, 
                  remote_peer_info.qp_num,
                  remote_peer_info.lid,
                  0,
                  conn_info.config.port_num,
                  0,
                  &remote_peer_info.gid)) {
          PCCL_DLOG_DEBUG(std::format("Failed to modify QP {} to RTR state.", qp_id));
          return false;
        }

        if (!modifyQPToRTS(conn_id, qp_id)) {
          PCCL_DLOG_DEBUG(std::format("Failed to modify QP {} to RTS state.", qp_id));
          return false;
        }
      }
      conn_info.connected = true;
      PCCL_DLOG_DEBUG(std::format("Connection {} successfully connected.", conn_id));
      return true;
    } catch (const std::exception& e) {
      PCCL_DLOG_DEBUG(std::format("Exception during connect: {}", e.what()));
      return false;
    }
  }

  bool disconnect(ConnectionId conn_id) {
    PCCL_DLOG_DEBUG(std::format("Disconnecting connection {}", conn_id));
    std::lock_guard<std::mutex> lock(connections_mutex_);
    auto it = connections_.find(conn_id);
    if (it == connections_.end()) {
      PCCL_DLOG_DEBUG(std::format("Connection {} not found.", conn_id));
      return false;
    }
    if (!it->second.connected) {
      PCCL_DLOG_DEBUG(std::format("Connection {} already disconnected.", conn_id));
      return false;
    }
    
    it->second.connected = false;
    PCCL_DLOG_DEBUG(std::format("Connection {} marked as disconnected.", conn_id));
    // Note: This doesn't move the QPs back to a disconnected state (e.g., RESET).
    // This is often acceptable as the connection object is usually destroyed.
    return true;
  }

  QPId createQP(ConnectionId conn_id, const QPConfig& config) {
    PCCL_DLOG_DEBUG(std::format("Creating QP for connection {}", conn_id));
    std::lock_guard<std::mutex> lock(connections_mutex_);
    auto it = connections_.find(conn_id);
    if (it == connections_.end()) {
      PCCL_DLOG_DEBUG(std::format("Connection {} not found. Failed.", conn_id));
      return 0;
    }
    if (it->second.qps.size() >= (size_t)it->second.config.max_qp_per_connection) {
      PCCL_DLOG_DEBUG(std::format("Max QPs per connection reached. Failed."));
      return 0;
    }
    
    try {
      ibv_qp_init_attr init_attr{};
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
      auto qp = std::make_shared<VerbsQueuePair>(*pd_, init_attr);
      it->second.qps[qp_id] = qp;
      
      PCCL_DLOG_DEBUG(std::format("Created QP with id={} and qp_num={} for connection {}", qp_id, qp->getQpNum(), conn_id));
      return qp_id;
    } catch (const RuntimeException& e) {
      PCCL_DLOG_DEBUG(std::format("Exception during QP creation: {}", e.what()));
      return 0;
    }
  }

  bool modifyQPToInit(ConnectionId conn_id, QPId qp_id) {
    PCCL_DLOG_DEBUG(std::format("Modifying QP {} on conn {} to INIT state.", qp_id, conn_id));
    auto qp = getQP(conn_id, qp_id);
    if (!qp) {
      PCCL_DLOG_DEBUG(std::format("QP not found. Failed."));
      return false;
    }
    
    ibv_qp_attr attr{};
    attr.qp_state = IBV_QPS_INIT;
    attr.pkey_index = 0;
    auto conn_it = connections_.find(conn_id);
    if (conn_it == connections_.end()) {
      PCCL_DLOG_DEBUG(std::format("Connection {} not found. Failed.", conn_id));
      return false;
    }
    attr.port_num = conn_it->second.config.port_num;
    attr.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | 
                           IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC;
    
    try {
      qp->modify(attr, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS);
      PCCL_DLOG_DEBUG(std::format("QP {} successfully moved to INIT state.", qp_id));
      return true;
    } catch (const RuntimeException& e) {
      PCCL_DLOG_DEBUG(std::format("Exception while modifying QP to INIT: {}", e.what()));
      return false;
    }
  }

  bool modifyQPToRTR(ConnectionId conn_id, QPId qp_id, 
              uint32_t remote_qpn, uint16_t dlid, uint8_t sl, 
              uint8_t port_num, uint16_t pkey_index, 
              const ibv_gid* sgid) {
    PCCL_DLOG_DEBUG(std::format("Modifying QP {} on conn {} to RTR state. Remote QPN: {}, Remote LID: {}", qp_id, conn_id, remote_qpn, dlid));
    auto qp = getQP(conn_id, qp_id);
    if (!qp) {
      PCCL_DLOG_DEBUG(std::format("QP not found. Failed."));
      return false;
    }
    
    auto conn_it = connections_.find(conn_id);
    if (conn_it == connections_.end()) {
      PCCL_DLOG_DEBUG(std::format("Connection {} not found. Failed.", conn_id));
      return false;
    }
    
    ibv_qp_attr attr{};
    attr.qp_state = IBV_QPS_RTR;
    attr.path_mtu = IBV_MTU_4096;
    attr.dest_qp_num = remote_qpn;
    attr.rq_psn = 0;
    attr.max_dest_rd_atomic = 1;
    attr.min_rnr_timer = 12;
    attr.ah_attr.dlid = dlid;
    attr.ah_attr.sl = sl;
    attr.ah_attr.src_path_bits = 0;
    attr.ah_attr.port_num = port_num;
    attr.ah_attr.is_global = (sgid != nullptr && sgid->global.interface_id != 0) ? 1 : 0;
    
    if (attr.ah_attr.is_global) {
      PCCL_DLOG_DEBUG(std::format("Using RoCE (is_global=1), GID index={}", conn_it->second.config.gid_index));
      memcpy(&attr.ah_attr.grh.dgid, sgid, sizeof(ibv_gid));
      attr.ah_attr.grh.flow_label = 0;
      attr.ah_attr.grh.hop_limit = 1;
      attr.ah_attr.grh.sgid_index = conn_it->second.config.gid_index;
    } else {
      PCCL_DLOG_DEBUG("Using InfiniBand (is_global=0)");
    }
    
    try {
      qp->modify(attr, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | 
                        IBV_QP_DEST_QPN | IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | 
                        IBV_QP_MIN_RNR_TIMER);
      PCCL_DLOG_DEBUG(std::format("QP {} successfully moved to RTR state.", qp_id));
      return true;
    } catch (const RuntimeException& e) {
      PCCL_DLOG_DEBUG(std::format("Exception while modifying QP to RTR: {}", e.what()));
      return false;
    }
  }

  bool modifyQPToRTS(ConnectionId conn_id, QPId qp_id) {
    PCCL_DLOG_DEBUG(std::format("Modifying QP {} on conn {} to RTS state.", qp_id, conn_id));
    auto qp = getQP(conn_id, qp_id);
    if (!qp) {
      PCCL_DLOG_DEBUG(std::format("QP not found. Failed."));
      return false;
    }
    
    ibv_qp_attr attr{};

    attr.qp_state = IBV_QPS_RTS;
    attr.timeout = 15;
    attr.retry_cnt = 5;
    attr.rnr_retry = 5;
    attr.sq_psn = 0;
    attr.max_rd_atomic = 1;
    
    try {
      qp->modify(attr, IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | 
                        IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC);
      PCCL_DLOG_DEBUG(std::format("QP {} successfully moved to RTS state.", qp_id));
      return true;
    } catch (const RuntimeException& e) {
      PCCL_DLOG_DEBUG(std::format("Exception while modifying QP to RTS: {}", e.what()));
      return false;
    }
  }

  bool postSend(ConnectionId conn_id, QPId qp_id, ibv_send_wr* wr, ibv_send_wr** bad_wr) {
    auto qp = getQP(conn_id, qp_id);
    if (!qp) {
      PCCL_DLOG_DEBUG(std::format("QP {} not found. Failed.", qp_id));
      return false;
    }
    int status = qp->postSend(wr, bad_wr);
    if (status) PCCL_LOG_ERROR(std::format("QP post send error {}", status));
    return true;
  }

  bool postRecv(ConnectionId conn_id, QPId qp_id, ibv_recv_wr* wr, ibv_recv_wr** bad_wr) {
    auto qp = getQP(conn_id, qp_id);
    if (!qp) {
      PCCL_DLOG_DEBUG(std::format("QP {} not found. Failed.", qp_id));
      return false;
    }
    int status = qp->postRecv(wr, bad_wr);
    if (status) PCCL_LOG_ERROR(std::format("QP post send error {}", status));
    return true;
  }

  int pollCQ(ConnectionId conn_id, int num_entries, ibv_wc* wc) {
    auto cq = getCQ(conn_id);
    if (!cq) {
      PCCL_DLOG_DEBUG(std::format("CQ for connection {} not found. Failed.", conn_id));
      return -1;
    }
    int num_completions = cq->poll(num_entries, wc);
    if (num_completions > 0) {
      PCCL_DLOG_DEBUG(std::format("Polled CQ for conn {}, got {} completions.", conn_id, num_completions));
      for (int i = 0; i < num_completions; ++i) {
        PCCL_DLOG_DEBUG(std::format(" WC[{}]: wr_id={}, status={}", 
          i, wc[i].wr_id, VerbsLib::getInstance().wcStatusStr(wc[i].status)));
      }
    } else if (num_completions < 0) {
      PCCL_DLOG_DEBUG(std::format("Error polling CQ for conn {}, pollCq returned {}", conn_id, num_completions));
    }
    return num_completions;
  }

  std::shared_ptr<VerbsQueuePair> getQP(ConnectionId conn_id, QPId qp_id) {
    std::lock_guard<std::mutex> lock(connections_mutex_);
    auto conn_it = connections_.find(conn_id);
    if (conn_it == connections_.end()) return nullptr;
    
    auto qp_it = conn_it->second.qps.find(qp_id);
    if (qp_it == conn_it->second.qps.end()) return nullptr;
    
    return qp_it->second;
  }

  std::shared_ptr<VerbsCompletionQueue> getCQ(ConnectionId conn_id) {
    std::lock_guard<std::mutex> lock(connections_mutex_);
    auto it = connections_.find(conn_id);
    if (it == connections_.end()) return nullptr;
    return it->second.cq;
  }

  std::shared_ptr<VerbsProtectionDomain> getPD() {
    return pd_;
  }

  bool isConnected(ConnectionId conn_id) const {
    std::lock_guard<std::mutex> lock(connections_mutex_);
    auto it = connections_.find(conn_id);
    if (it == connections_.end()) return false;
    return it->second.connected;
  }

  size_t getConnectionCount() const {
    std::lock_guard<std::mutex> lock(connections_mutex_);
    return connections_.size();
  }

  size_t getQPCount(ConnectionId conn_id) const {
    std::lock_guard<std::mutex> lock(connections_mutex_);
    auto it = connections_.find(conn_id);
    if (it == connections_.end()) return 0;
    return it->second.qps.size();
  }

  VerbsRemotePeerInfo getLocalMetadata(ConnectionId conn_id, QPId qp_id) {
    PCCL_DLOG_DEBUG(std::format("Getting local metadata for conn={}, qp={}", conn_id, qp_id));
    std::lock_guard<std::mutex> lock(connections_mutex_);
    auto conn_it = connections_.find(conn_id);
    if (conn_it == connections_.end()) {
      PCCL_DLOG_DEBUG(std::format("Connection {} not found.", conn_id));
      return VerbsRemotePeerInfo{};
    }

    auto qp_it = conn_it->second.qps.find(qp_id);
    if (qp_it == conn_it->second.qps.end()) {
      PCCL_DLOG_DEBUG(std::format("QP {} not found.", qp_id));
      return VerbsRemotePeerInfo{};
    }
    
    VerbsRemotePeerInfo metadata;
    metadata.qp_num = qp_it->second->getQpNum();
    
    try {
      ibv_port_attr port_attr = context_->queryPort(conn_it->second.config.port_num);
      metadata.lid = port_attr.lid;
      
      auto& lib = VerbsLib::getInstance();
      if (lib.queryGid(context_->get(), conn_it->second.config.port_num, 
                       conn_it->second.config.gid_index, &metadata.gid) != 0) {
        PCCL_DLOG_DEBUG(std::format("Failed to query GID for port {} index {}, GID will be zero.", 
          conn_it->second.config.port_num, conn_it->second.config.gid_index));
        std::memset(&metadata.gid, 0, sizeof(ibv_gid));
      }
    } catch (const RuntimeException& e) {
      PCCL_DLOG_DEBUG(std::format("Exception while getting metadata: {}", e.what()));
      std::memset(&metadata.gid, 0, sizeof(ibv_gid));
    }
    
    PCCL_DLOG_DEBUG(std::format("Generated metadata: qp_num={}, lid={}", 
      metadata.qp_num, metadata.lid));
    return metadata;
  }

private:
  struct ConnectionInfo {
    ConnectionConfig config;
    std::shared_ptr<VerbsCompletionQueue> cq;
    std::unordered_map<QPId, std::shared_ptr<VerbsQueuePair>> qps;
    std::atomic<bool> connected{false};

    ConnectionInfo() = default;
    ConnectionInfo(const ConnectionInfo& other)
      : config(other.config), cq(other.cq), connected(other.connected.load()) {
      for (const auto& pair : other.qps) qps[pair.first] = pair.second;
    }
    ConnectionInfo(ConnectionInfo&& other) noexcept
      : config(std::move(other.config)), cq(std::move(other.cq)), qps(std::move(other.qps)),
        connected(other.connected.load()) {
      other.connected = false;
    }
    ConnectionInfo& operator=(const ConnectionInfo& other) {
      if (this != &other) {
        config = other.config;
        cq = other.cq;
        connected = other.connected.load();
        qps.clear();
        for (const auto& pair : other.qps) qps[pair.first] = pair.second;
      }
      return *this;
    }
    ConnectionInfo& operator=(ConnectionInfo&& other) noexcept {
      if (this != &other) {
        config = std::move(other.config);
        cq = std::move(other.cq);
        qps = std::move(other.qps);
        connected = other.connected.load();
        other.connected = false;
      }
      return *this;
    }
  };

  ConnectionId generateConnectionId() { return next_connection_id_.fetch_add(1, std::memory_order_relaxed); }
  QPId generateQPId() { return next_qp_id_.fetch_add(1, std::memory_order_relaxed); }

  std::shared_ptr<VerbsDevList> device_list_;
  std::shared_ptr<VerbsContext> context_;
  std::shared_ptr<VerbsProtectionDomain> pd_;

  std::unordered_map<ConnectionId, ConnectionInfo> connections_;
  mutable std::mutex connections_mutex_;

  std::atomic<ConnectionId> next_connection_id_{1};
  std::atomic<QPId> next_qp_id_{1};

  std::atomic<bool> initialized_{false};
};

} // namespace pccl
