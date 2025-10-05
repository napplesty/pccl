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

namespace pccl::communicator {

class VerbsLib {
public:
  static VerbsLib& getInstance();
  void load(const std::string& library_path);
  ~VerbsLib();
  VerbsLib(const VerbsLib&) = delete;
  VerbsLib& operator=(const VerbsLib&) = delete;

  ibv_device** getDeviceList(int* num_devices);
  int freeDeviceList(ibv_device** list);
  const char* getDeviceName(ibv_device* device);
  ibv_context* openDevice(ibv_device* device);
  int closeDevice(ibv_context* context);
  int queryDevice(ibv_context* context, ibv_device_attr* device_attr);
  int queryPort(ibv_context* context, uint8_t port_num, ibv_port_attr* port_attr);
  int queryGid(ibv_context* context, uint8_t port_num, int index, ibv_gid* gid);
  ibv_pd* allocPd(ibv_context* context);
  int deallocPd(ibv_pd* pd);
  ibv_mr* regMr(ibv_pd* pd, void* addr, size_t length, int access);
  int deregMr(ibv_mr* mr);
  ibv_comp_channel* createCompChannel(ibv_context* context);
  int destroyCompChannel(ibv_comp_channel* channel);
  int getCqEvent(ibv_comp_channel* channel, ibv_cq** cq, void** cq_context);
  void ackCqEvents(ibv_cq* cq, unsigned int nevents);
  ibv_cq* createCq(ibv_context* context, int cqe, void* cq_context, ibv_comp_channel* channel, int comp_vector);
  int destroyCq(ibv_cq* cq);
  int pollCq(ibv_cq* cq, int num_entries, ibv_wc* wc);
  ibv_qp* createQp(ibv_pd* pd, ibv_qp_init_attr* attr);
  int destroyQp(ibv_qp* qp);
  int modifyQp(ibv_qp* qp, ibv_qp_attr* attr, int attr_mask);
  int postSend(ibv_qp* qp, ibv_send_wr* wr, ibv_send_wr** bad_wr);
  int postRecv(ibv_qp* qp, ibv_recv_wr* wr, ibv_recv_wr** bad_wr);
  int reqNotifyCq(ibv_cq* cq, int solicited_only);
  const char* wcStatusStr(enum ibv_wc_status status);

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

  VerbsLib() = default;
  void* handle_ = nullptr;
  std::mutex load_mutex_;
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
};

class VerbsDevList {
public:
  VerbsDevList();
  ~VerbsDevList();
  VerbsDevList(const VerbsDevList&) = delete;
  VerbsDevList& operator=(const VerbsDevList&) = delete;
  VerbsDevList(VerbsDevList&& other) noexcept;
  VerbsDevList& operator=(VerbsDevList&& other) noexcept;
  ibv_device* getDevice(int index) const;
  std::string getDeviceName(int index);
  int getNumDevices() const;
private:
  ibv_device** devices_ = nullptr;
  int num_devices_ = 0;
};

class VerbsContext {
public:
  explicit VerbsContext(ibv_device* device);
  ~VerbsContext();
  VerbsContext(const VerbsContext&) = delete;
  VerbsContext& operator=(const VerbsContext&) = delete;
  VerbsContext(VerbsContext&& other) noexcept;
  VerbsContext& operator=(VerbsContext&& other) noexcept;
  ibv_device_attr queryDevice() const;
  ibv_port_attr queryPort(uint8_t port_num) const;
  ibv_gid queryGid(uint8_t port_num, int index) const;
  ibv_context* get() const;
private:
  ibv_context* context_ = nullptr;
};

class VerbsProtectionDomain {
public:
  explicit VerbsProtectionDomain(const VerbsContext& ctx);
  ~VerbsProtectionDomain();
  VerbsProtectionDomain(const VerbsProtectionDomain&) = delete;
  VerbsProtectionDomain& operator=(const VerbsProtectionDomain&) = delete;
  VerbsProtectionDomain(VerbsProtectionDomain&& other) noexcept;
  VerbsProtectionDomain& operator=(VerbsProtectionDomain&& other) noexcept;
  ibv_pd* get() const;
private:
  ibv_pd* pd_ = nullptr;
};

class VerbsCompChannel {
public:
  explicit VerbsCompChannel(const VerbsContext& ctx);
  ~VerbsCompChannel();
  VerbsCompChannel(const VerbsCompChannel&) = delete;
  VerbsCompChannel& operator=(const VerbsCompChannel&) = delete;
  VerbsCompChannel(VerbsCompChannel&& other) noexcept;
  VerbsCompChannel& operator=(VerbsCompChannel&& other) noexcept;
  ibv_comp_channel* get() const;
private:
  ibv_comp_channel* channel_ = nullptr;
};

class VerbsCompletionQueue {
public:
  VerbsCompletionQueue(const VerbsContext& ctx, int cqe, void* cq_ctx,
                        const VerbsCompChannel* channel = nullptr, int comp_vec = 0);
  ~VerbsCompletionQueue();
  VerbsCompletionQueue(const VerbsCompletionQueue&) = delete;
  VerbsCompletionQueue& operator=(const VerbsCompletionQueue&) = delete;
  VerbsCompletionQueue(VerbsCompletionQueue&& other) noexcept;
  VerbsCompletionQueue& operator=(VerbsCompletionQueue&& other) noexcept;
  void getEvent(const VerbsCompChannel& channel, VerbsCompletionQueue*& out_cq, void*& out_ctx);
  int poll(int num_entries, ibv_wc* wc) const;
  void ackEvents(unsigned int nevents) const;
  ibv_cq* get() const;
private:
  ibv_cq* cq_ = nullptr;
};

class VerbsMemoryRegion {
public:
  VerbsMemoryRegion(const VerbsProtectionDomain& pd, void* addr, size_t len, int access);
  ~VerbsMemoryRegion();
  VerbsMemoryRegion(const VerbsMemoryRegion&) = delete;
  VerbsMemoryRegion& operator=(const VerbsMemoryRegion&) = delete;
  VerbsMemoryRegion(VerbsMemoryRegion&& other) noexcept;
  VerbsMemoryRegion& operator=(VerbsMemoryRegion&& other) noexcept;
  void* getAddr() const;
  size_t getLength() const;
  uint32_t getLKey() const;
  uint32_t getRKey() const;
  ibv_mr* get() const;
private:
  ibv_mr* mr_ = nullptr;
  void* addr_ = nullptr;
  size_t len_ = 0;
  uint32_t lkey_ = 0;
  uint32_t rkey_ = 0;
};

class VerbsQueuePair {
public:
  VerbsQueuePair(const VerbsProtectionDomain& pd, const ibv_qp_init_attr& init_attr);
  ~VerbsQueuePair();
  VerbsQueuePair(const VerbsQueuePair&) = delete;
  VerbsQueuePair& operator=(const VerbsQueuePair&) = delete;
  VerbsQueuePair(VerbsQueuePair&& other) noexcept;
  VerbsQueuePair& operator=(VerbsQueuePair&& other) noexcept;
  int modify(const ibv_qp_attr& attr, int attr_mask) const;
  int postSend(ibv_send_wr* wr, ibv_send_wr** bad_wr) const;
  int postRecv(ibv_recv_wr* wr, ibv_recv_wr** bad_wr) const;
  uint32_t getQpNum() const;
  ibv_qp* get() const;
private:
  ibv_qp* qp_ = nullptr;
  uint32_t qp_num_ = 0;
};

struct VerbsRemotePeerInfo {
  uint32_t qp_num;
  uint16_t lid;
  union ibv_gid gid;
    
  VerbsRemotePeerInfo();
  VerbsRemotePeerInfo(const VerbsRemotePeerInfo& other);
  VerbsRemotePeerInfo(VerbsRemotePeerInfo&& other) noexcept;
  VerbsRemotePeerInfo& operator=(const VerbsRemotePeerInfo& other);
  VerbsRemotePeerInfo& operator=(VerbsRemotePeerInfo&& other) noexcept;
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

    QPConfig();
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

  VerbsManager();
  ~VerbsManager();
  VerbsManager(const VerbsManager&) = delete;
  VerbsManager& operator=(const VerbsManager&) = delete;
  VerbsManager(VerbsManager&& other) noexcept;
  VerbsManager& operator=(VerbsManager&& other) noexcept;

  bool initialize(const std::string& device_name = "", uint8_t port_num = 1);
  ConnectionId createConnection(const ConnectionConfig& config);
  bool destroyConnection(ConnectionId conn_id);
  bool connect(ConnectionId conn_id, const VerbsRemotePeerInfo& remote_peer_info);
  bool disconnect(ConnectionId conn_id);
  QPId createQP(ConnectionId conn_id, const QPConfig& config);
  bool modifyQPToInit(ConnectionId conn_id, QPId qp_id);
  bool modifyQPToRTR(ConnectionId conn_id, QPId qp_id, 
              uint32_t remote_qpn, uint16_t dlid, uint8_t sl, 
              uint8_t port_num, uint16_t pkey_index, 
              const ibv_gid* sgid);
  bool modifyQPToRTS(ConnectionId conn_id, QPId qp_id);
  bool postSend(ConnectionId conn_id, QPId qp_id, ibv_send_wr* wr, ibv_send_wr** bad_wr);
  bool postRecv(ConnectionId conn_id, QPId qp_id, ibv_recv_wr* wr, ibv_recv_wr** bad_wr);
  int pollCQ(ConnectionId conn_id, int num_entries, ibv_wc* wc);
  std::shared_ptr<VerbsQueuePair> getQP(ConnectionId conn_id, QPId qp_id);
  std::shared_ptr<VerbsCompletionQueue> getCQ(ConnectionId conn_id);
  std::shared_ptr<VerbsProtectionDomain> getPD();
  bool isConnected(ConnectionId conn_id) const;
  size_t getConnectionCount() const;
  size_t getQPCount(ConnectionId conn_id) const;
  VerbsRemotePeerInfo getLocalMetadata(ConnectionId conn_id, QPId qp_id);

private:
  struct ConnectionInfo {
    ConnectionConfig config;
    std::shared_ptr<VerbsCompletionQueue> cq;
    std::unordered_map<QPId, std::shared_ptr<VerbsQueuePair>> qps;
    std::atomic<bool> connected{false};

    ConnectionInfo();
    ConnectionInfo(const ConnectionInfo& other);
    ConnectionInfo(ConnectionInfo&& other) noexcept;
    ConnectionInfo& operator=(const ConnectionInfo& other);
    ConnectionInfo& operator=(ConnectionInfo&& other) noexcept;
  };

  ConnectionId generateConnectionId();
  QPId generateQPId();

  std::shared_ptr<VerbsDevList> device_list_;
  std::shared_ptr<VerbsContext> context_;
  std::shared_ptr<VerbsProtectionDomain> pd_;

  std::unordered_map<ConnectionId, ConnectionInfo> connections_;
  mutable std::mutex connections_mutex_;

  std::atomic<ConnectionId> next_connection_id_{1};
  std::atomic<QPId> next_qp_id_{1};

  std::atomic<bool> initialized_{false};
};

} // namespace pccl::communicator
