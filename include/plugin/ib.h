#pragma once
#if defined(USE_IBVERBS)
#include <dlfcn.h>
#include <infiniband/verbs.h>

#include <list>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace pccl {

struct IBVerbs {
public:
  static void initialize() {
    handle = dlopen("libibverbs.so", RTLD_NOW);
    if (!handle) {
      throw std::runtime_error("Failed to load libibverbs: " +
                               std::string(dlerror()));
    }

    ibv_get_device_list_lib =
        (ibv_get_device_list_t)dlsym(handle, "ibv_get_device_list");
    ibv_free_device_list_lib =
        (ibv_free_device_list_t)dlsym(handle, "ibv_free_device_list");
    ibv_alloc_pd_lib = (ibv_alloc_pd_t)dlsym(handle, "ibv_alloc_pd");
    ibv_dealloc_pd_lib = (ibv_dealloc_pd_t)dlsym(handle, "ibv_dealloc_pd");
    ibv_open_device_lib = (ibv_open_device_t)dlsym(handle, "ibv_open_device");
    ibv_close_device_lib =
        (ibv_close_device_t)dlsym(handle, "ibv_close_device");
    ibv_query_device_lib =
        (ibv_query_device_t)dlsym(handle, "ibv_query_device");
    ibv_create_cq_lib = (ibv_create_cq_t)dlsym(handle, "ibv_create_cq");
    ibv_create_qp_lib = (ibv_create_qp_t)dlsym(handle, "ibv_create_qp");
    ibv_destroy_cq_lib = (ibv_destroy_cq_t)dlsym(handle, "ibv_destroy_cq");
    ibv_reg_mr_lib = (ibv_reg_mr_t)dlsym(handle, "ibv_reg_mr");
    ibv_dereg_mr_lib = (ibv_dereg_mr_t)dlsym(handle, "ibv_dereg_mr");
    ibv_query_gid_lib = (ibv_query_gid_t)dlsym(handle, "ibv_query_gid");
    ibv_modify_qp_lib = (ibv_modify_qp_t)dlsym(handle, "ibv_modify_qp");
    ibv_destroy_qp_lib = (ibv_destroy_qp_t)dlsym(handle, "ibv_destroy_qp");
    ibv_query_port_lib = (ibv_query_port_t)dlsym(handle, "ibv_query_port");
    ibv_reg_mr_iova2_lib =
        (ibv_reg_mr_iova2_t)dlsym(handle, "ibv_reg_mr_iova2");
    ibv_create_comp_channel_lib =
        (ibv_create_comp_channel_t)dlsym(handle, "ibv_create_comp_channel");
    ibv_destroy_comp_channel_lib =
        (ibv_destroy_comp_channel_t)dlsym(handle, "ibv_destroy_comp_channel");
    ibv_get_cq_event_lib =
        (ibv_get_cq_event_t)dlsym(handle, "ibv_get_cq_event");
    ibv_ack_cq_events_lib =
        (ibv_ack_cq_events_t)dlsym(handle, "ibv_ack_cq_events");
    ibv_req_notify_cq_lib =
        (ibv_req_notify_cq_t)dlsym(handle, "ibv_req_notify_cq");
    ibv_post_recv_lib = (ibv_post_recv_t)dlsym(handle, "ibv_post_recv");
    ibv_create_srq_lib = (ibv_create_srq_t)dlsym(handle, "ibv_create_srq");
    ibv_destroy_srq_lib = (ibv_destroy_srq_t)dlsym(handle, "ibv_destroy_srq");
    ibv_post_srq_recv_lib =
        (ibv_post_srq_recv_t)dlsym(handle, "ibv_post_srq_recv");
    ibv_create_ah_lib = (ibv_create_ah_t)dlsym(handle, "ibv_create_ah");
    ibv_destroy_ah_lib = (ibv_destroy_ah_t)dlsym(handle, "ibv_destroy_ah");

    if (!ibv_get_device_list_lib || !ibv_free_device_list_lib ||
        !ibv_alloc_pd_lib || !ibv_dealloc_pd_lib || !ibv_open_device_lib ||
        !ibv_close_device_lib || !ibv_query_device_lib || !ibv_create_cq_lib ||
        !ibv_create_qp_lib || !ibv_destroy_cq_lib || !ibv_reg_mr_lib ||
        !ibv_dereg_mr_lib || !ibv_query_gid_lib || !ibv_reg_mr_iova2_lib ||
        !ibv_modify_qp_lib || !ibv_destroy_qp_lib || !ibv_query_port_lib ||
        !ibv_create_comp_channel_lib || !ibv_destroy_comp_channel_lib ||
        !ibv_get_cq_event_lib || !ibv_ack_cq_events_lib ||
        !ibv_req_notify_cq_lib || !ibv_post_recv_lib || !ibv_create_srq_lib ||
        !ibv_destroy_srq_lib || !ibv_post_srq_recv_lib || !ibv_create_ah_lib ||
        !ibv_destroy_ah_lib) {
      throw std::runtime_error(
          "Failed to load one or more function in the ibibverbs library: " +
          std::string(dlerror()));
      dlclose(handle);
    }
  }

public:
  static struct ibv_device **ibv_get_device_list(int *num_devices) {
    return ibv_get_device_list_lib(num_devices);
  }

  static void ibv_free_device_list(struct ibv_device **list) {
    ibv_free_device_list_lib(list);
  }

  static struct ibv_pd *ibv_alloc_pd(struct ibv_context *context) {
    return ibv_alloc_pd_lib(context);
  }

  static int ibv_dealloc_pd(struct ibv_pd *pd) {
    return ibv_dealloc_pd_lib(pd);
  }

  static struct ibv_context *ibv_open_device(struct ibv_device *device) {
    return ibv_open_device_lib(device);
  }

  static int ibv_close_device(struct ibv_context *context) {
    return ibv_close_device_lib(context);
  }

  static int ibv_query_device(struct ibv_context *context,
                              struct ibv_device_attr *device_attr) {
    return ibv_query_device_lib(context, device_attr);
  }

  static struct ibv_cq *ibv_create_cq(struct ibv_context *context, int cqe,
                                      void *cq_context,
                                      struct ibv_comp_channel *channel,
                                      int comp_vector) {
    return ibv_create_cq_lib(context, cqe, cq_context, channel, comp_vector);
  }

  static struct ibv_qp *ibv_create_qp(struct ibv_pd *pd,
                                      struct ibv_qp_init_attr *qp_init_attr) {
    return ibv_create_qp_lib(pd, qp_init_attr);
  }

  static int ibv_destroy_cq(struct ibv_cq *cq) {
    return ibv_destroy_cq_lib(cq);
  }

  static struct ibv_mr *ibv_reg_mr2(struct ibv_pd *pd, void *addr,
                                    size_t length, int access) {
    return ibv_reg_mr_lib(pd, addr, length, access);
  }

  static int ibv_dereg_mr(struct ibv_mr *mr) { return ibv_dereg_mr_lib(mr); }

  static int ibv_query_gid(struct ibv_context *context, uint8_t port_num,
                           int index, union ibv_gid *gid) {
    return ibv_query_gid_lib(context, port_num, index, gid);
  }

  static int ibv_modify_qp(struct ibv_qp *qp, struct ibv_qp_attr *attr,
                           int attr_mask) {
    return ibv_modify_qp_lib(qp, attr, attr_mask);
  }

  static int ibv_destroy_qp(struct ibv_qp *qp) {
    return ibv_destroy_qp_lib(qp);
  }

  static inline int ibv_post_send(struct ibv_qp *qp, struct ibv_send_wr *wr,
                                  struct ibv_send_wr **bad_wr) {
    return qp->context->ops.post_send(qp, wr, bad_wr);
  }

  static inline int ibv_poll_cq(struct ibv_cq *cq, int num_entries,
                                struct ibv_wc *wc) {
    return cq->context->ops.poll_cq(cq, num_entries, wc);
  }

  static int ibv_query_port_w(struct ibv_context *context, uint8_t port_num,
                              struct ibv_port_attr *port_attr) {
    return ibv_query_port_lib(context, port_num, port_attr);
  }

  static struct ibv_mr *ibv_reg_mr_iova2_w(struct ibv_pd *pd, void *addr,
                                           size_t length, uint64_t iova,
                                           unsigned int access) {
    return ibv_reg_mr_iova2_lib(pd, addr, length, iova, access);
  }

  static inline int ibv_post_recv(struct ibv_qp *qp, struct ibv_recv_wr *wr,
                                  struct ibv_recv_wr **bad_wr) {
    return qp->context->ops.post_recv(qp, wr, bad_wr);
  }

  static struct ibv_srq *
  ibv_create_srq(struct ibv_pd *pd, struct ibv_srq_init_attr *srq_init_attr) {
    return ibv_create_srq_lib(pd, srq_init_attr);
  }

  static int ibv_destroy_srq(struct ibv_srq *srq) {
    return ibv_destroy_srq_lib(srq);
  }

  static inline int ibv_post_srq_recv(struct ibv_srq *srq,
                                      struct ibv_recv_wr *wr,
                                      struct ibv_recv_wr **bad_wr) {
    return ibv_post_srq_recv_lib(srq, wr, bad_wr);
  }

  static struct ibv_ah *ibv_create_ah(struct ibv_pd *pd,
                                      struct ibv_ah_attr *attr) {
    return ibv_create_ah_lib(pd, attr);
  }

  static int ibv_destroy_ah(struct ibv_ah *ah) {
    return ibv_destroy_ah_lib(ah);
  }

  static struct ibv_comp_channel *
  ibv_create_comp_channel(struct ibv_context *context) {
    return ibv_create_comp_channel_lib(context);
  }

  static int ibv_destroy_comp_channel(struct ibv_comp_channel *channel) {
    return ibv_destroy_comp_channel_lib(channel);
  }

  static int ibv_get_cq_event(struct ibv_comp_channel *channel,
                              struct ibv_cq **cq, void **cq_context) {
    return ibv_get_cq_event_lib(channel, cq, cq_context);
  }

  static void ibv_ack_cq_events(struct ibv_cq *cq, unsigned int nevents) {
    ibv_ack_cq_events_lib(cq, nevents);
  }

  static int ibv_req_notify_cq(struct ibv_cq *cq, int solicited_only) {
    return ibv_req_notify_cq_lib(cq, solicited_only);
  }

  static void cleanup() {
    if (handle) {
      dlclose(handle);
      handle = nullptr;
    }
  }

private:
  static inline void *handle = nullptr;

  typedef struct ibv_device **(*ibv_get_device_list_t)(int *);
  typedef void (*ibv_free_device_list_t)(struct ibv_device **);
  typedef struct ibv_pd *(*ibv_alloc_pd_t)(struct ibv_context *);
  typedef int (*ibv_dealloc_pd_t)(struct ibv_pd *);
  typedef struct ibv_context *(*ibv_open_device_t)(struct ibv_device *);
  typedef int (*ibv_close_device_t)(struct ibv_context *);
  typedef int (*ibv_query_device_t)(struct ibv_context *,
                                    struct ibv_device_attr *);
  typedef struct ibv_cq *(*ibv_create_cq_t)(struct ibv_context *, int, void *,
                                            struct ibv_comp_channel *, int);
  typedef struct ibv_qp *(*ibv_create_qp_t)(struct ibv_pd *,
                                            struct ibv_qp_init_attr *);
  typedef int (*ibv_destroy_cq_t)(struct ibv_cq *);
  typedef int (*ibv_destroy_qp_t)(struct ibv_qp *);
  typedef struct ibv_mr *(*ibv_reg_mr_t)(struct ibv_pd *, void *, size_t, int);
  typedef int (*ibv_dereg_mr_t)(struct ibv_mr *);
  typedef int (*ibv_query_gid_t)(struct ibv_context *, uint8_t, int,
                                 union ibv_gid *);
  typedef int (*ibv_modify_qp_t)(struct ibv_qp *, struct ibv_qp_attr *, int);
  typedef int (*ibv_query_port_t)(struct ibv_context *, uint8_t,
                                  struct ibv_port_attr *);
  typedef struct ibv_mr *(*ibv_reg_mr_iova2_t)(struct ibv_pd *pd, void *addr,
                                               size_t length, uint64_t iova,
                                               unsigned int access);
  typedef struct ibv_comp_channel *(*ibv_create_comp_channel_t)(
      struct ibv_context *);
  typedef int (*ibv_destroy_comp_channel_t)(struct ibv_comp_channel *);
  typedef int (*ibv_get_cq_event_t)(struct ibv_comp_channel *, struct ibv_cq **,
                                    void **);
  typedef void (*ibv_ack_cq_events_t)(struct ibv_cq *, unsigned int);
  typedef int (*ibv_req_notify_cq_t)(struct ibv_cq *, int);
  typedef int (*ibv_post_recv_t)(struct ibv_qp *, struct ibv_recv_wr *,
                                 struct ibv_recv_wr **);
  typedef struct ibv_srq *(*ibv_create_srq_t)(struct ibv_pd *,
                                              struct ibv_srq_init_attr *);
  typedef int (*ibv_destroy_srq_t)(struct ibv_srq *);
  typedef int (*ibv_post_srq_recv_t)(struct ibv_srq *, struct ibv_recv_wr *,
                                     struct ibv_recv_wr **);
  typedef struct ibv_ah *(*ibv_create_ah_t)(struct ibv_pd *,
                                            struct ibv_ah_attr *);
  typedef int (*ibv_destroy_ah_t)(struct ibv_ah *);

  static inline ibv_get_device_list_t ibv_get_device_list_lib = nullptr;
  static inline ibv_free_device_list_t ibv_free_device_list_lib = nullptr;
  static inline ibv_alloc_pd_t ibv_alloc_pd_lib = nullptr;
  static inline ibv_dealloc_pd_t ibv_dealloc_pd_lib = nullptr;
  static inline ibv_open_device_t ibv_open_device_lib = nullptr;
  static inline ibv_close_device_t ibv_close_device_lib = nullptr;
  static inline ibv_query_device_t ibv_query_device_lib = nullptr;
  static inline ibv_create_cq_t ibv_create_cq_lib = nullptr;
  static inline ibv_create_qp_t ibv_create_qp_lib = nullptr;
  static inline ibv_destroy_cq_t ibv_destroy_cq_lib = nullptr;
  static inline ibv_reg_mr_t ibv_reg_mr_lib = nullptr;
  static inline ibv_dereg_mr_t ibv_dereg_mr_lib = nullptr;
  static inline ibv_query_gid_t ibv_query_gid_lib = nullptr;
  static inline ibv_modify_qp_t ibv_modify_qp_lib = nullptr;
  static inline ibv_destroy_qp_t ibv_destroy_qp_lib = nullptr;
  static inline ibv_query_port_t ibv_query_port_lib = nullptr;
  static inline ibv_reg_mr_iova2_t ibv_reg_mr_iova2_lib = nullptr;
  static inline ibv_create_comp_channel_t ibv_create_comp_channel_lib = nullptr;
  static inline ibv_destroy_comp_channel_t ibv_destroy_comp_channel_lib =
      nullptr;
  static inline ibv_get_cq_event_t ibv_get_cq_event_lib = nullptr;
  static inline ibv_ack_cq_events_t ibv_ack_cq_events_lib = nullptr;
  static inline ibv_req_notify_cq_t ibv_req_notify_cq_lib = nullptr;
  static inline ibv_post_recv_t ibv_post_recv_lib = nullptr;
  static inline ibv_create_srq_t ibv_create_srq_lib = nullptr;
  static inline ibv_destroy_srq_t ibv_destroy_srq_lib = nullptr;
  static inline ibv_post_srq_recv_t ibv_post_srq_recv_lib = nullptr;
  static inline ibv_create_ah_t ibv_create_ah_lib = nullptr;
  static inline ibv_destroy_ah_t ibv_destroy_ah_lib = nullptr;
};

struct IbMrInfo {
  uint64_t addr;
  uint32_t rkey;
};

class IbMr {
public:
  ~IbMr();

  IbMrInfo getInfo() const;
  const void *getBuff() const;
  uint32_t getLkey() const;

private:
  IbMr(ibv_pd *pd, void *buff, size_t size);

  ibv_mr *mr;
  void *buff;
  size_t size;

  friend class IbCtx;
};

struct IbQpInfo {
  uint16_t lid;
  uint8_t port;
  uint8_t linkLayer;
  uint32_t qpn;
  uint64_t spn;
  int mtu;
  uint64_t iid;
  bool is_grh;
};

enum class WsStatus {
  Success,
};

class IbQp {
public:
  ~IbQp();

  void rtr(const IbQpInfo &info);
  void rts();
  void stageLoad(const IbMr *mr, const IbMrInfo &info, size_t size,
                 uint64_t wrId, uint64_t srcOffset, uint64_t dstOffset,
                 bool signaled);
  void stageSend(const IbMr *mr, const IbMrInfo &info, uint32_t size,
                 uint64_t wrId, uint64_t srcOffset, uint64_t dstOffset,
                 bool signaled);
  void stageAtomicAdd(const IbMr *mr, const IbMrInfo &info, uint64_t wrId,
                      uint64_t dstOffset, uint64_t addVal, bool signaled);
  void stageSendWithImm(const IbMr *mr, const IbMrInfo &info, uint32_t size,
                        uint64_t wrId, uint64_t srcOffset, uint64_t dstOffset,
                        bool signaled, unsigned int immData);
  void postSend();
  int pollCq();

  IbQpInfo getInfo() { return this->info; }
  int getWcStatus(int idx) const;
  int getNumCqItems() const;

protected:
  struct WrInfo {
    ibv_send_wr *wr;
    ibv_sge *sge;
  };

  IbQp(ibv_context *ctx, ibv_pd *pd, int port, int maxCqSize, int maxCqPollNum,
       int maxSendWr, int maxRecvWr, int maxWrPerSend);
  WrInfo getNewWrInfo();

  IbQpInfo info;

  ibv_qp *qp;
  ibv_cq *cq;
  std::vector<ibv_wc> wcs;
  std::vector<ibv_send_wr> wrs;
  std::vector<ibv_sge> sges;
  int wrn;
  int numSignaledPostedItems;
  int numSignaledStagedItems;

  const int maxCqPollNum;
  const int maxWrPerSend;

  friend class IbCtx;
};

class IbCtx {
public:
  IbCtx(const std::string &devName);
  ~IbCtx();

  IbQp *createQp(int maxCqSize, int maxCqPollNum, int maxSendWr, int maxRecvWr,
                 int maxWrPerSend, int port = -1);
  const IbMr *registerMr(void *buff, size_t size);

private:
  bool isPortUsable(int port) const;
  int getAnyActivePort() const;

  const std::string devName;
  ibv_context *ctx;
  ibv_pd *pd;
  std::list<std::unique_ptr<IbQp>> qps;
  std::list<std::unique_ptr<IbMr>> mrs;
};

} // namespace pccl

#endif // USE_IBVERBS
