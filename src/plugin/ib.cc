#if defined(USE_IBVERBS)

#include "plugin/ib.h"

#include <unistd.h>

#include <fstream>
#include <sstream>
#include <string>

namespace pccl {

#if defined(USE_CUDA)

static bool checkNvPeerMemLoaded() {
  std::ifstream file("/proc/modules");
  std::string line;
  while (std::getline(file, line)) {
    if (line.find("nvidia_peermem") != std::string::npos) return true;
  }
  return false;
}

#endif

IbMr::IbMr(ibv_pd *pd, void *buff, size_t size) : buff(buff) {
  if (size == 0) {
    throw std::invalid_argument("invalid size: " + std::to_string(size));
  }
  static __thread uintptr_t pageSize = 0;
  if (pageSize == 0) {
    pageSize = sysconf(_SC_PAGESIZE);
  }
  uintptr_t addr = reinterpret_cast<uintptr_t>(buff) & -pageSize;
  std::size_t pages =
      (size + (reinterpret_cast<uintptr_t>(buff) - addr) + pageSize - 1) /
      pageSize;
  this->mr = IBVerbs::ibv_reg_mr2(
      pd, reinterpret_cast<void *>(addr), pages * pageSize,
      IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
          IBV_ACCESS_REMOTE_READ | IBV_ACCESS_RELAXED_ORDERING |
          IBV_ACCESS_REMOTE_ATOMIC);
  if (this->mr == nullptr) {
    std::stringstream err;
    err << "ibv_reg_mr failed (errno " << errno << ")";
    throw std::runtime_error(err.str());
  }
  this->size = pages * pageSize;
}

IbMr::~IbMr() { IBVerbs::ibv_dereg_mr(this->mr); }

IbMrInfo IbMr::getInfo() const {
  IbMrInfo info;
  info.addr = reinterpret_cast<uint64_t>(this->buff);
  info.rkey = this->mr->rkey;
  return info;
}

const void *IbMr::getBuff() const { return this->buff; }

uint32_t IbMr::getLkey() const { return this->mr->lkey; }

IbQp::IbQp(ibv_context *ctx, ibv_pd *pd, int port, int maxCqSize,
           int maxCqPollNum, int maxSendWr, int maxRecvWr, int maxWrPerSend)
    : numSignaledPostedItems(0),
      numSignaledStagedItems(0),
      maxCqPollNum(maxCqPollNum),
      maxWrPerSend(maxWrPerSend) {
  this->cq = IBVerbs::ibv_create_cq(ctx, maxCqSize, nullptr, nullptr, 0);
  if (this->cq == nullptr) {
    std::stringstream err;
    err << "ibv_create_cq failed (errno " << errno << ")";
    throw std::runtime_error(err.str());
  }

  struct ibv_qp_init_attr qpInitAttr;
  memset(&qpInitAttr, 0, sizeof(qpInitAttr));
  qpInitAttr.sq_sig_all = 0;
  qpInitAttr.send_cq = this->cq;
  qpInitAttr.recv_cq = this->cq;
  qpInitAttr.qp_type = IBV_QPT_RC;
  qpInitAttr.cap.max_send_wr = maxSendWr;
  qpInitAttr.cap.max_recv_wr = maxRecvWr;
  qpInitAttr.cap.max_send_sge = 1;
  qpInitAttr.cap.max_recv_sge = 1;
  qpInitAttr.cap.max_inline_data = 0;

  struct ibv_qp *_qp = IBVerbs::ibv_create_qp(pd, &qpInitAttr);
  if (_qp == nullptr) {
    std::stringstream err;
    err << "ibv_create_qp failed (errno " << errno << ")";
    throw std::runtime_error(err.str());
  }

  struct ibv_port_attr portAttr;
  if (IBVerbs::ibv_query_port_w(ctx, port, &portAttr) != 0) {
    std::stringstream err;
    err << "ibv_query_port failed (errno " << errno << ")";
    throw std::runtime_error(err.str());
  }
  this->info.lid = portAttr.lid;
  this->info.port = port;
  this->info.linkLayer = portAttr.link_layer;
  this->info.qpn = _qp->qp_num;
  this->info.mtu = portAttr.active_mtu;
  this->info.is_grh = (portAttr.flags & IBV_QPF_GRH_REQUIRED);

  if (portAttr.link_layer != IBV_LINK_LAYER_INFINIBAND || this->info.is_grh) {
    union ibv_gid gid;
    if (IBVerbs::ibv_query_gid(ctx, port, 0, &gid) != 0) {
      std::stringstream err;
      err << "ibv_query_gid failed (errno " << errno << ")";
      throw std::runtime_error(err.str());
    }
    this->info.spn = gid.global.subnet_prefix;
    this->info.iid = gid.global.interface_id;
  }

  struct ibv_qp_attr qpAttr;
  memset(&qpAttr, 0, sizeof(qpAttr));
  qpAttr.qp_state = IBV_QPS_INIT;
  qpAttr.pkey_index = 0;
  qpAttr.port_num = port;
  qpAttr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ |
                           IBV_ACCESS_REMOTE_ATOMIC;
  if (IBVerbs::ibv_modify_qp(_qp, &qpAttr,
                             IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT |
                                 IBV_QP_ACCESS_FLAGS) != 0) {
    std::stringstream err;
    err << "ibv_modify_qp failed (errno " << errno << ")";
    throw std::runtime_error(err.str());
  }
  this->qp = _qp;
  this->wrn = 0;
  this->wrs = std::make_shared<std::vector<ibv_send_wr>>(maxWrPerSend);
  this->sges = std::make_shared<std::vector<ibv_sge>>(maxWrPerSend);
  this->wcs = std::make_shared<std::vector<ibv_wc>>(maxCqPollNum);
}

}  // namespace pccl

#endif  // USE_IBVERBS
