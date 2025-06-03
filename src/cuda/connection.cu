#include "config.h" // For pccl::Config::DefaultMaxCqSize etc.
#include "cuda/connection.h"
#include "plugin/ib.h"   // For pccl::IbCtx, pccl::IbQp, etc.
#include "plugin/sock.h" // For pccl::SockCtx, pccl::SockQp, etc.
#include "runtime.h" // For pccl::Transport, pccl::RegisteredMemory, pccl::TransportInfo (constructor param)

// Assuming cuda/registered_memory.h is included via cuda/connection.h for the
// complex RegisteredMemory::Impl #include "cuda/registered_memory.h" // Should
// be via cuda/connection.h

#include <cstdarg> // For va_list, va_start, va_end, vsnprintf
#include <cstdio>  // For fprintf, stderr
#include <cuda_runtime.h>
#include <sstream>   // For std::ostringstream
#include <stdexcept> // For std::runtime_error
#include <string>    // For std::string
#include <vector>    // For std::vector in helper functions

// --- Utility Functions (replacing macros) ---

// CUDA Error Checking Function
inline void pccl_cuda_check_impl_conn(cudaError_t err, const char *file,
                                      int line, const char *call_str) {
  if (err != cudaSuccess) {
    std::ostringstream oss;
    oss << "CUDA error in " << file << " at line " << line << " (" << call_str
        << "): " << cudaGetErrorString(err) << " (" << static_cast<int>(err)
        << ")";
    throw std::runtime_error(oss.str());
  }
}
#define PCCL_CUDA_CHECK(call)                                                  \
  pccl_cuda_check_impl_conn(call, __FILE__, __LINE__, #call)

// Logging Functions
inline void pccl_log_impl_conn(FILE *stream, const char *level,
                               const char *file, int line, const char *fmt,
                               ...) {
  fprintf(stream, "[PCCL %s Conn] (%s:%d) ", level, file, line);
  va_list args;
  va_start(args, fmt);
  vfprintf(stream, fmt, args);
  va_end(args);
  fprintf(stream, "\n");
}

#define PCCL_LOG_INFO(fmt, ...)                                                \
  pccl_log_impl_conn(stderr, "INFO", __FILE__, __LINE__, fmt, ##__VA_ARGS__)
#define PCCL_LOG_WARN(fmt, ...)                                                \
  pccl_log_impl_conn(stderr, "WARN", __FILE__, __LINE__, fmt, ##__VA_ARGS__)
#define PCCL_LOG_ERROR(fmt, ...)                                               \
  pccl_log_impl_conn(stderr, "ERROR", __FILE__, __LINE__, fmt, ##__VA_ARGS__)

// --- End Utility Functions ---

namespace pccl {

// Forward declaration of a potential helper function if needed, or define it
// static void validatePcclTransport(RegisteredMemory mem, Transport
// expectedTransport, uint64_t offset, uint64_t size);

// --- CudaIpcConnection Implementation ---

CudaIpcConnection::CudaIpcConnection(Endpoint remote, Endpoint local)
    : remoteRank_(remote.rank()) {
  if (local.transport() != Transport::CudaIpc) {
    throw std::runtime_error(
        "CudaIpcConnection: Local endpoint must be CudaIpc.");
  }
  if (remote.transport() != Transport::CudaIpc) {
    throw std::runtime_error(
        "CudaIpcConnection: Remote endpoint must be CudaIpc.");
  }
  // TODO: Add host hash check if pccl::Endpoint supports it.
  PCCL_LOG_INFO("CudaIpcConnection created for remoteRank %d", remoteRank_);
}

CudaIpcConnection::~CudaIpcConnection() {
  PCCL_LOG_INFO("CudaIpcConnection destroyed for remoteRank %d", remoteRank_);
}

void CudaIpcConnection::switchBackend(void * /*backend*/) {
  PCCL_LOG_WARN(
      "CudaIpcConnection::switchBackend is not implemented for remoteRank %d.",
      remoteRank_);
}

static void validate_pccl_ipc_memory(const RegisteredMemory &mem,
                                     uint64_t offset, uint64_t access_size,
                                     const char *mem_name) {
  if (!mem.devicePtr()) { // Assuming RegisteredMemory from runtime.h has
                          // devicePtr()
    std::ostringstream ss;
    ss << "CudaIpcConnection: " << mem_name << " memory has null devicePtr.";
    PCCL_LOG_WARN("%s", ss.str().c_str());
  }
  if (offset + access_size >
      mem.size()) { // Assuming RegisteredMemory from runtime.h has size()
    std::ostringstream ss;
    ss << "CudaIpcConnection: " << mem_name
       << " memory access out of bounds. Offset: " << offset
       << ", Access Size: " << access_size << ", Memory Size: " << mem.size();
    throw std::runtime_error(ss.str());
  }
}

void CudaIpcConnection::write(RegisteredMemory dst, uint64_t dstOffset,
                              RegisteredMemory src, uint64_t srcOffset,
                              uint64_t size) {
  validate_pccl_ipc_memory(dst, dstOffset, size, "Destination");
  validate_pccl_ipc_memory(src, srcOffset, size, "Source");

  char *dstDevPtr = static_cast<char *>(dst.devicePtr());
  char *srcDevPtr = static_cast<char *>(src.devicePtr());

  if (!dstDevPtr || !srcDevPtr) {
    PCCL_LOG_WARN("CudaIpcConnection::write: null device pointer(s) (dst: %p, "
                  "src: %p). Aborting write.",
                  static_cast<void *>(dstDevPtr),
                  static_cast<void *>(srcDevPtr));
    return;
  }

  PCCL_LOG_INFO("CudaIpcConnection write: from %p (offset %lu) to %p (offset "
                "%lu), size %lu, remoteRank %d",
                static_cast<void *>(srcDevPtr), srcOffset,
                static_cast<void *>(dstDevPtr), dstOffset, size, remoteRank_);

  PCCL_CUDA_CHECK(cudaMemcpyAsync(dstDevPtr + dstOffset, srcDevPtr + srcOffset,
                                  size, cudaMemcpyDeviceToDevice,
                                  cudaStreamDefault));
}

void CudaIpcConnection::updateAndSync(RegisteredMemory dst, uint64_t dstOffset,
                                      RegisteredMemory src, uint64_t srcOffset,
                                      uint64_t size) {
  validate_pccl_ipc_memory(dst, dstOffset, size, "Destination");
  validate_pccl_ipc_memory(src, srcOffset, size, "Source");

  char *dstDevPtr = static_cast<char *>(dst.devicePtr());
  char *srcDevPtr = static_cast<char *>(src.devicePtr());

  if (!dstDevPtr || !srcDevPtr) {
    PCCL_LOG_WARN("CudaIpcConnection::updateAndSync: null device pointer(s) "
                  "(dst: %p, src: %p). Aborting.",
                  static_cast<void *>(dstDevPtr),
                  static_cast<void *>(srcDevPtr));
    return;
  }

  PCCL_LOG_INFO("CudaIpcConnection updateAndSync: from %p (offset %lu) to %p "
                "(offset %lu), size %lu, remoteRank %d",
                static_cast<void *>(srcDevPtr), srcOffset,
                static_cast<void *>(dstDevPtr), dstOffset, size, remoteRank_);

  PCCL_CUDA_CHECK(cudaMemcpyAsync(dstDevPtr + dstOffset, srcDevPtr + srcOffset,
                                  size, cudaMemcpyDeviceToDevice,
                                  cudaStreamDefault));
  PCCL_CUDA_CHECK(cudaStreamSynchronize(cudaStreamDefault));
}

void CudaIpcConnection::flush(int64_t timeoutUsec /*= 3e9*/) {
  if (timeoutUsec >= 0) {
    PCCL_LOG_INFO("CudaIpcConnection flush: timeout %lld usec (potentially "
                  "ignored for IPC stream sync) for remoteRank %d",
                  (long long)timeoutUsec, remoteRank_);
  }
  PCCL_CUDA_CHECK(cudaStreamSynchronize(cudaStreamDefault));
}

Transport CudaIpcConnection::transport() { return Transport::CudaIpc; }

Transport CudaIpcConnection::remoteTransport() { return Transport::CudaIpc; }

int CudaIpcConnection::remoteRank() { return remoteRank_; }

bool CudaIpcConnection::connected() { return true; }

uint64_t CudaIpcConnection::bandwidth() { return 0; }

uint64_t CudaIpcConnection::latency() { return 0; }

// --- Helper: Extract IB MR from RegisteredMemory ---
// Assumes RegisteredMemory 'mem' is the complex one from
// cuda/registered_memory.h and 'mem.pimpl_' is accessible and points to its
// Impl struct.
const IbMr *get_ib_mr_from_pccl_memory(const RegisteredMemory &mem,
                                       Transport ib_transport_type) {
  if (!mem.pimpl_) {
    PCCL_LOG_ERROR(
        "get_ib_mr_from_pccl_memory: RegisteredMemory pimpl_ is null.");
    return nullptr;
  }
  try {
    // The Impl class is pccl::RegisteredMemory::Impl from
    // cuda/registered_memory.h
    const auto &transport_specific_info =
        mem.pimpl_->getTransportInfo(ib_transport_type);
    return transport_specific_info.ibMr;
  } catch (const std::runtime_error &e) {
    PCCL_LOG_ERROR(
        "get_ib_mr_from_pccl_memory: Failed to get IB MR for transport %d: %s",
        static_cast<int>(ib_transport_type), e.what());
    return nullptr;
  }
}

IbMrInfo get_ib_mr_info_from_pccl_memory(const RegisteredMemory &mem,
                                         Transport ib_transport_type) {
  if (!mem.pimpl_) {
    PCCL_LOG_ERROR(
        "get_ib_mr_info_from_pccl_memory: RegisteredMemory pimpl_ is null.");
    return {}; // Return default/empty
  }
  try {
    const auto &transport_specific_info =
        mem.pimpl_->getTransportInfo(ib_transport_type);
    return transport_specific_info.ibMrInfo;
  } catch (const std::runtime_error &e) {
    PCCL_LOG_ERROR("get_ib_mr_info_from_pccl_memory: Failed to get IB MR Info "
                   "for transport %d: %s",
                   static_cast<int>(ib_transport_type), e.what());
    return {}; // Return default/empty
  }
}

// --- IbConnection Implementation ---

// Placeholder: Extract necessary IB setup details from pccl::TransportInfo
// (runtime.h) These functions need to be implemented based on the actual
// structure of pccl::TransportInfo
std::string extract_ib_device_name(const pccl::TransportInfo &conn_info_param) {
  PCCL_LOG_WARN("extract_ib_device_name: Placeholder, returning empty. "
                "Implement based on pccl::TransportInfo structure.");
  // Example: return conn_info_param.get_parameter("IB_DEVICE_NAME");
  return ""; // Needs real implementation
}
int extract_ib_port(const pccl::TransportInfo &conn_info_param) {
  PCCL_LOG_WARN("extract_ib_port: Placeholder, returning 1. Implement based on "
                "pccl::TransportInfo structure.");
  // Example: return std::stoi(conn_info_param.get_parameter("IB_PORT"));
  return 1; // Needs real implementation
}
IbQpInfo extract_remote_ib_qp_info(const pccl::TransportInfo &conn_info_param) {
  PCCL_LOG_WARN("extract_remote_ib_qp_info: Placeholder, returning empty. "
                "Implement based on pccl::TransportInfo structure.");
  // Example: deserialize from
  // conn_info_param.get_parameter("REMOTE_IB_QP_INFO_SERIALIZED");
  return {}; // Needs real implementation
}

IbConnection::IbConnection(pccl::TransportInfo info, int remRank)
    : remoteRank_(remRank), ctx_(nullptr), qp_(nullptr) {

  std::string ib_dev_name = extract_ib_device_name(info);
  int ib_port_num = extract_ib_port(info); // Assuming info provides the port

  if (ib_dev_name.empty()) {
    PCCL_LOG_ERROR(
        "IbConnection Constructor: IB device name is empty for remoteRank %d.",
        remoteRank_);
    throw std::runtime_error("IB device name not provided for IbConnection.");
  }

  try {
    ctx_ = new IbCtx(ib_dev_name); // pccl::IbCtx from plugin/ib.h
    // TODO: Determine maxCqSize, maxCqPollNum, maxSendWr, maxRecvWr,
    // maxWrPerSend from Config or info
    int max_cq_size = pccl::Config::DefaultMaxCqSize;
    int max_cq_poll_num = pccl::Config::DefaultMaxCqPollNum;
    int max_send_wr = pccl::Config::DefaultMaxSendWr;
    int max_recv_wr =
        pccl::Config::DefaultMaxRecvWr; // May not be used if QP is send-only
                                        // for one-way connection
    int max_wr_per_send = pccl::Config::DefaultMaxWrPerSend;

    qp_ = ctx_->createQp(max_cq_size, max_cq_poll_num, max_send_wr, max_recv_wr,
                         max_wr_per_send, ib_port_num);

    if (!qp_) {
      delete ctx_;
      ctx_ = nullptr;
      PCCL_LOG_ERROR(
          "IbConnection Constructor: Failed to create QP for remoteRank %d.",
          remoteRank_);
      throw std::runtime_error("Failed to create IB QP.");
    }

    IbQpInfo remote_qp_info = extract_remote_ib_qp_info(info);
    // TODO: Validate remote_qp_info before using.

    qp_->rtr(remote_qp_info); // Transition to Ready To Receive
    qp_->rts();               // Transition to Ready To Send

    PCCL_LOG_INFO("IbConnection created and connected for remoteRank %d on dev "
                  "%s, port %d. QP num %u.",
                  remoteRank_, ib_dev_name.c_str(), ib_port_num,
                  qp_ ? qp_->getInfo().qpn : 0);

  } catch (const std::exception &e) {
    if (qp_) {
      delete qp_;
      qp_ = nullptr;
    } // qp_ is owned by IbCtx if created via ctx_->createQp, but if standalone:
      // delete qp_;
    if (ctx_) {
      delete ctx_;
      ctx_ = nullptr;
    }
    PCCL_LOG_ERROR("IbConnection Constructor: Exception for remoteRank %d: %s",
                   remoteRank_, e.what());
    throw; // Re-throw
  }
}

IbConnection::~IbConnection() {
  PCCL_LOG_INFO("IbConnection destroying for remoteRank %d. QP num %u.",
                remoteRank_,
                (qp_ && ctx_) ? qp_->getInfo().qpn
                              : 0); // ctx_ check because qp_ is from ctx_
  // IbQp objects are typically managed by IbCtx.
  // If IbCtx dtor handles cleaning up QPs it created, explicit qp_ deletion
  // might not be needed or could be an error if qp_ is a raw pointer from a
  // list in ctx_. For now, assume IbCtx destructor cleans up QPs it created. If
  // createQp returns new IbQp(), then delete qp_; delete ctx_; If IbCtx owns
  // QPs, then only delete ctx_; Given `std::list<std::unique_ptr<IbQp>> qps;`
  // in IbCtx, deleting ctx_ should handle it.
  delete ctx_; // This should trigger unique_ptr destructors for QPs in IbCtx's
               // list.
  ctx_ = nullptr;
  qp_ = nullptr; // qp_ was a raw pointer from ctx_, now dangling if ctx_
                 // deleted. Good to null out.
}

void IbConnection::switchBackend(void * /*backend*/) {
  PCCL_LOG_WARN("IbConnection::switchBackend: NI for remoteRank %d.",
                remoteRank_);
}

void IbConnection::write(RegisteredMemory dst, uint64_t dstOffset,
                         RegisteredMemory src, uint64_t srcOffset,
                         uint64_t size) {
  if (!qp_ || !ctx_) {
    PCCL_LOG_ERROR(
        "IbConnection::write: QP or Ctx not initialized for remoteRank %d.",
        remoteRank_);
    throw std::runtime_error(
        "IbConnection not properly initialized for write.");
  }

  // Determine which IB transport (IB0, IB1) based on RegisteredMemory or
  // connection context. For simplicity, assuming Transport::IB0. This needs a
  // proper mechanism.
  Transport ib_transport = Transport::IB0;
  // TODO: Logic to select correct IB transport based on 'dst' and 'src'
  // properties or connection setup.

  const IbMr *src_mr = get_ib_mr_from_pccl_memory(src, ib_transport);
  IbMrInfo dst_mr_info = get_ib_mr_info_from_pccl_memory(dst, ib_transport);

  if (!src_mr) {
    PCCL_LOG_ERROR(
        "IbConnection::write: Failed to get source IbMr for remoteRank %d.",
        remoteRank_);
    throw std::runtime_error("IbConnection write: Source MR is null.");
  }
  if (dst_mr_info.rkey == 0 &&
      dst_mr_info.addr == 0) { // Basic check for uninitialized remote MR info
    PCCL_LOG_ERROR("IbConnection::write: Destination IbMrInfo appears "
                   "uninitialized for remoteRank %d.",
                   remoteRank_);
    throw std::runtime_error(
        "IbConnection write: Destination MR Info is invalid.");
  }

  PCCL_LOG_INFO("IbConnection write: size %lu to remoteRank %d. (srcOffset "
                "%lu, dstOffset %lu)",
                size, remoteRank_, srcOffset, dstOffset);

  qp_->stageSend(src_mr, dst_mr_info, static_cast<uint32_t>(size),
                 0, // wrId - PCCL IbQp stageSend API does not take wr_id.
                 srcOffset, dstOffset,
                 true); // signaled
  qp_->postSend();
}

void IbConnection::updateAndSync(RegisteredMemory dst, uint64_t dstOffset,
                                 RegisteredMemory src, uint64_t srcOffset,
                                 uint64_t size) {
  if (!qp_ || !ctx_) {
    PCCL_LOG_ERROR("IbConnection::updateAndSync: QP or Ctx not initialized for "
                   "remoteRank %d.",
                   remoteRank_);
    throw std::runtime_error(
        "IbConnection not properly initialized for updateAndSync.");
  }

  Transport ib_transport = Transport::IB0; // Simplified, see note in write()
  const IbMr *src_mr = get_ib_mr_from_pccl_memory(src, ib_transport);
  IbMrInfo dst_mr_info = get_ib_mr_info_from_pccl_memory(dst, ib_transport);

  if (!src_mr) {
    PCCL_LOG_ERROR("IbConnection::updateAndSync: Failed to get source IbMr for "
                   "remoteRank %d.",
                   remoteRank_);
    throw std::runtime_error("IbConnection updateAndSync: Source MR is null.");
  }
  if (dst_mr_info.rkey == 0 && dst_mr_info.addr == 0) {
    PCCL_LOG_ERROR("IbConnection::updateAndSync: Destination IbMrInfo appears "
                   "uninitialized for remoteRank %d.",
                   remoteRank_);
    throw std::runtime_error(
        "IbConnection updateAndSync: Destination MR Info is invalid.");
  }

  PCCL_LOG_INFO("IbConnection updateAndSync: size %lu to remoteRank %d. "
                "(srcOffset %lu, dstOffset %lu)",
                size, remoteRank_, srcOffset, dstOffset);

  qp_->stageSend(src_mr, dst_mr_info, static_cast<uint32_t>(size),
                 0, // wrId - see note in write()
                 srcOffset, dstOffset,
                 true); // signaled for completion
  qp_->postSend();

  PCCL_LOG_INFO(
      "IbConnection updateAndSync: Polling for completion for remoteRank %d.",
      remoteRank_);

  int polls = 0;
  const int max_polls = 10000000; // Safeguard
  bool_t R = 0;
  int R_val = qp_->getWcStatus(R);
  bool R_status = (R_val == 0);
  int wc_num;
  do {
    wc_num = qp_->pollCq();
    if (wc_num < 0) {
      PCCL_LOG_ERROR(
          "IbConnection::updateAndSync: pollCq failed for remoteRank %d.",
          remoteRank_);
      throw std::runtime_error("IB pollCq failed in updateAndSync.");
    }
    if (wc_num > 0)
      break;
    polls++;
  } while (polls < max_polls && wc_num == 0);

  if (wc_num == 0 && polls >= max_polls) {
    PCCL_LOG_WARN("IbConnection::updateAndSync: Max polls reached, completion "
                  "not confirmed for remoteRank %d.",
                  remoteRank_);
    // Potentially throw timeout error
  } else if (wc_num > 0) {
    for (int i = 0; i < wc_num; ++i) {
      // pccl::IbQp::getWcStatus(idx) is const and returns int.
      // WsStatus::Success is enum class.
      if (qp_->getWcStatus(i) != static_cast<int>(WsStatus::Success)) {
        PCCL_LOG_ERROR("IbConnection::updateAndSync: Completion with error for "
                       "remoteRank %d. WC status: %d",
                       remoteRank_, qp_->getWcStatus(i));
        // throw std::runtime_error("IB operation failed in updateAndSync (WC
        // error).");
      }
    }
    PCCL_LOG_INFO("IbConnection updateAndSync: %d WC(s) received for "
                  "remoteRank %d. Assuming completion.",
                  wc_num, remoteRank_);
  } else {
    // Should have been caught by max_polls if wc_num remained 0.
    PCCL_LOG_WARN("IbConnection::updateAndSync: Exited polling loop "
                  "unexpectedly for remoteRank %d.",
                  remoteRank_);
  }
}

void IbConnection::flush(int64_t timeoutUsec) {
  if (!qp_ || !ctx_) {
    PCCL_LOG_ERROR(
        "IbConnection::flush: QP or Ctx not initialized for remoteRank %d.",
        remoteRank_);
    return;
  }
  PCCL_LOG_INFO(
      "IbConnection flushing connection for remoteRank %d (timeout %lld us).",
      remoteRank_, (long long)timeoutUsec);

  // This is a best-effort flush. A true flush often needs to know how many
  // operations are outstanding. Here, we just poll until the CQ is empty for a
  // number of tries or timeout.
  // TODO: Implement proper timeout handling using a timer.
  int polls = 0;
  const int max_empty_polls_streak =
      1000; // If CQ is empty for this many consecutive polls, assume flushed.
  int current_empty_polls = 0;
  int total_wcs_polled = 0;

  while (true) { // TODO: Add timeout check
    int num_wcs = qp_->pollCq();
    if (num_wcs < 0) {
      PCCL_LOG_ERROR("IbConnection::flush: pollCq error for remoteRank %d.",
                     remoteRank_);
      throw std::runtime_error("IB pollCq error during flush.");
    }

    total_wcs_polled += num_wcs;
    for (int i = 0; i < num_wcs; ++i) {
      if (qp_->getWcStatus(i) != static_cast<int>(WsStatus::Success)) {
        PCCL_LOG_WARN("IbConnection::flush: Work completion error for "
                      "remoteRank %d. Status: %d",
                      remoteRank_, qp_->getWcStatus(i));
      }
    }

    if (num_wcs == 0) {
      current_empty_polls++;
      if (current_empty_polls >= max_empty_polls_streak) {
        PCCL_LOG_INFO("IbConnection flush: CQ empty for %d polls. Assuming "
                      "flushed for remoteRank %d.",
                      max_empty_polls_streak, remoteRank_);
        break;
      }
    } else {
      current_empty_polls = 0; // Reset streak
    }

    polls++;
    // TODO: Check timeout here if implemented.
    // if (timer.elapsed() > timeoutUsec && timeoutUsec >=0) {
    // PCCL_LOG_WARN(...); break; }
  }
  PCCL_LOG_INFO("IbConnection flush completed attempt for remoteRank %d. "
                "Polled %d WCs in %d attempts.",
                remoteRank_, total_wcs_polled, polls);
}

Transport IbConnection::transport() {
  return Transport::IB0; /* Or determine dynamically */
}
Transport IbConnection::remoteTransport() {
  return Transport::IB0; /* Or determine dynamically */
}
bool IbConnection::connected() {
  return (ctx_ != nullptr &&
          qp_ != nullptr); // Basic check, real check depends on QP state.
}
uint64_t IbConnection::bandwidth() { return 0; }
uint64_t IbConnection::latency() { return 0; }

// --- Helper: Extract Socket MR from RegisteredMemory ---
const SockMr *get_sock_mr_from_pccl_memory(const RegisteredMemory &mem) {
  if (!mem.pimpl_) {
    PCCL_LOG_ERROR(
        "get_sock_mr_from_pccl_memory: RegisteredMemory pimpl_ is null.");
    return nullptr;
  }
  try {
    // Assuming socket uses Ethernet transport for TransportInfo lookup
    const auto &transport_specific_info =
        mem.pimpl_->getTransportInfo(Transport::Ethernet);
    return transport_specific_info.sockMr;
  } catch (const std::runtime_error &e) {
    PCCL_LOG_ERROR("get_sock_mr_from_pccl_memory: Failed to get Sock MR: %s",
                   e.what());
    return nullptr;
  }
}

SockMrInfo get_sock_mr_info_from_pccl_memory(const RegisteredMemory &mem) {
  if (!mem.pimpl_) {
    PCCL_LOG_ERROR(
        "get_sock_mr_info_from_pccl_memory: RegisteredMemory pimpl_ is null.");
    return {};
  }
  try {
    const auto &transport_specific_info =
        mem.pimpl_->getTransportInfo(Transport::Ethernet);
    return transport_specific_info.sockMrInfo;
  } catch (const std::runtime_error &e) {
    PCCL_LOG_ERROR(
        "get_sock_mr_info_from_pccl_memory: Failed to get Sock MR Info: %s",
        e.what());
    return {};
  }
}

// Placeholder: Extract necessary Socket setup details from pccl::TransportInfo
// (runtime.h)
SockAddr extract_local_sock_addr(const pccl::TransportInfo &conn_info_param) {
  PCCL_LOG_WARN("extract_local_sock_addr: Placeholder. Implement based on "
                "pccl::TransportInfo.");
  // Example: deserialize from
  // conn_info_param.get_parameter("LOCAL_SOCK_ADDR_SERIALIZED");
  return {}; // Needs real implementation
}
SockQpInfo
extract_remote_sock_qp_info(const pccl::TransportInfo &conn_info_param) {
  PCCL_LOG_WARN("extract_remote_sock_qp_info: Placeholder. Implement based on "
                "pccl::TransportInfo.");
  // Example: deserialize from
  // conn_info_param.get_parameter("REMOTE_SOCK_QP_INFO_SERIALIZED");
  return {}; // Needs real implementation
}

// --- SockConnection Implementation ---
SockConnection::SockConnection(pccl::TransportInfo info, int remRank)
    : remoteRank_(remRank), ctx_(nullptr), qp_(nullptr) {

  SockAddr local_addr = extract_local_sock_addr(info);
  // TODO: Validate local_addr

  try {
    // SockCtx constructor takes SockAddr by value.
    // plugin/sock.h indicates SockCtx uses enable_shared_from_this, implies it
    // should be heap allocated and managed by shared_ptr typically. However,
    // plugin/sock.h doesn't show SockCtx being returned as shared_ptr from a
    // factory. MSCCLPP's EthernetConnection creates Sockets directly. Assuming
    // direct construction of SockCtx is okay here, and it internally starts its
    // threads. The lifetime of SockCtx needs to be managed; if multiple
    // connections share a SockCtx, it should be handled by a higher-level
    // context manager. If this connection *owns* the SockCtx, then `new
    // SockCtx` is appropriate.
    ctx_ = new SockCtx(
        local_addr); // This will start listen/recv threads in SockCtx.

    // TODO: Determine maxCqSize, maxWrqSize from Config or info
    int max_cq_size = pccl::Config::DefaultMaxCqSize;
    int max_wr_size = pccl::Config::DefaultMaxSendWr; // Assuming MaxSendWr maps
                                                      // to max_wr for SockQp

    qp_ = ctx_->createQp(max_cq_size,
                         max_wr_size); // Returns std::shared_ptr<SockQp>

    if (!qp_) {
      delete ctx_; // If qp creation failed, cleanup context
      ctx_ = nullptr;
      PCCL_LOG_ERROR(
          "SockConnection Constructor: Failed to create QP for remoteRank %d.",
          remoteRank_);
      throw std::runtime_error("Failed to create Socket QP.");
    }

    SockQpInfo remote_qp_info = extract_remote_sock_qp_info(info);
    // TODO: Validate remote_qp_info

    SockStatus status = qp_->connect(remote_qp_info);
    if (status != SockStatus::SUCCESS) {
      // qp_ is a shared_ptr, will be cleaned up. ctx_ needs manual deletion if
      // new'd.
      delete ctx_;
      ctx_ = nullptr;
      qp_ = nullptr; // Reset shared_ptr
      PCCL_LOG_ERROR("SockConnection Constructor: QP connect failed for "
                     "remoteRank %d with status %d.",
                     remoteRank_, static_cast<int>(status));
      throw std::runtime_error("Socket QP connect failed.");
    }
    PCCL_LOG_INFO("SockConnection created and connected for remoteRank %d. "
                  "Local QPN %lld, Remote QPN %lld.",
                  remoteRank_, qp_ ? qp_->getInfo().qpn : -1,
                  remote_qp_info.qpn);

  } catch (const std::exception &e) {
    // qp_ is shared_ptr, auto cleanup.
    if (ctx_) {
      delete ctx_;
      ctx_ = nullptr;
    } // Only if new'd above
    qp_ = nullptr;
    PCCL_LOG_ERROR(
        "SockConnection Constructor: Exception for remoteRank %d: %s",
        remoteRank_, e.what());
    throw;
  }
}

SockConnection::~SockConnection() {
  PCCL_LOG_INFO("SockConnection destroying for remoteRank %d. QPN %lld.",
                remoteRank_, (qp_ && ctx_) ? qp_->getInfo().qpn : -1);
  // qp_ is a std::shared_ptr<SockQp>, its destruction will be handled
  // automatically when this SockConnection object is destroyed, assuming it's
  // the only owner or last owner. If SockCtx was `new`-ed and is owned solely
  // by this connection:
  delete ctx_;
  ctx_ = nullptr;
  // qp_ = nullptr; // Not strictly necessary for shared_ptr, but good practice
  // if used elsewhere after this.
}

void SockConnection::switchBackend(void * /*backend*/) {
  PCCL_LOG_WARN("SockConnection::switchBackend: NI for remoteRank %d.",
                remoteRank_);
}

void SockConnection::write(RegisteredMemory dst, uint64_t dstOffset,
                           RegisteredMemory src, uint64_t srcOffset,
                           uint64_t size) {
  if (!qp_ || !ctx_) {
    PCCL_LOG_ERROR(
        "SockConnection::write: QP or Ctx not initialized for remoteRank %d.",
        remoteRank_);
    throw std::runtime_error(
        "SockConnection not properly initialized for write.");
  }

  const SockMr *src_mr = get_sock_mr_from_pccl_memory(src);
  SockMrInfo dst_mr_info = get_sock_mr_info_from_pccl_memory(dst);

  if (!src_mr) {
    PCCL_LOG_ERROR(
        "SockConnection::write: Failed to get source SockMr for remoteRank %d.",
        remoteRank_);
    throw std::runtime_error("SockConnection write: Source MR is null.");
  }
  if (dst_mr_info.mr_id == 0 && dst_mr_info.addr == 0) { // Basic check
    PCCL_LOG_ERROR("SockConnection::write: Destination SockMrInfo appears "
                   "uninitialized for remoteRank %d.",
                   remoteRank_);
    throw std::runtime_error(
        "SockConnection write: Destination MR Info is invalid.");
  }

  PCCL_LOG_INFO("SockConnection write: size %lu to remoteRank %d. (srcOffset "
                "%lu, dstOffset %lu)",
                size, remoteRank_, srcOffset, dstOffset);

  // TODO: Manage wr_id properly. A static counter is not safe for concurrency.
  // It should ideally be part of the SockQp or a connection-specific sequence.
  static uint64_t next_sock_wr_id = 1;
  uint64_t current_wr_id = next_sock_wr_id++;

  qp_->stageSend(src_mr, dst_mr_info, static_cast<uint32_t>(size),
                 current_wr_id, srcOffset, dstOffset);
  SockStatus status = qp_->postSend();
  if (status != SockStatus::SUCCESS) {
    PCCL_LOG_ERROR("SockConnection::write: postSend failed for remoteRank %d, "
                   "wrId %lu, status %d.",
                   remoteRank_, current_wr_id, static_cast<int>(status));
    throw std::runtime_error("Socket postSend failed.");
  }
}

void SockConnection::updateAndSync(RegisteredMemory dst, uint64_t dstOffset,
                                   RegisteredMemory src, uint64_t srcOffset,
                                   uint64_t size) {
  if (!qp_ || !ctx_) {
    PCCL_LOG_ERROR("SockConnection::updateAndSync: QP or Ctx not initialized "
                   "for remoteRank %d.",
                   remoteRank_);
    throw std::runtime_error(
        "SockConnection not properly initialized for updateAndSync.");
  }

  const SockMr *src_mr = get_sock_mr_from_pccl_memory(src);
  SockMrInfo dst_mr_info = get_sock_mr_info_from_pccl_memory(dst);

  if (!src_mr) {
    PCCL_LOG_ERROR("SockConnection::updateAndSync: Failed to get source SockMr "
                   "for remoteRank %d.",
                   remoteRank_);
    throw std::runtime_error(
        "SockConnection updateAndSync: Source MR is null.");
  }
  if (dst_mr_info.mr_id == 0 && dst_mr_info.addr == 0) {
    PCCL_LOG_ERROR("SockConnection::updateAndSync: Destination SockMrInfo "
                   "appears uninitialized for remoteRank %d.",
                   remoteRank_);
    throw std::runtime_error(
        "SockConnection updateAndSync: Destination MR Info is invalid.");
  }

  static uint64_t next_sock_sync_wr_id =
      7000001; // Different range for clarity, still not robust.
  uint64_t current_wr_id = next_sock_sync_wr_id++;

  PCCL_LOG_INFO("SockConnection updateAndSync: size %lu to remoteRank %d, wrId "
                "%lu. (srcOffset %lu, dstOffset %lu)",
                size, remoteRank_, current_wr_id, srcOffset, dstOffset);

  qp_->stageSend(src_mr, dst_mr_info, static_cast<uint32_t>(size),
                 current_wr_id, srcOffset, dstOffset);
  SockStatus send_status = qp_->postSend();
  if (send_status != SockStatus::SUCCESS) {
    PCCL_LOG_ERROR("SockConnection::updateAndSync: postSend failed for "
                   "remoteRank %d, wrId %lu with status %d.",
                   remoteRank_, current_wr_id, static_cast<int>(send_status));
    throw std::runtime_error("Socket postSend failed in updateAndSync.");
  }

  PCCL_LOG_INFO("SockConnection updateAndSync: Polling for wrId %lu completion "
                "for remoteRank %d.",
                current_wr_id, remoteRank_);

  int polls = 0;
  const int max_polls = 10000000; // Safeguard
  bool completed = false;
  bool_t R = 0;
  int R_val = qp_->getWcStatus(R).status;
  bool R_status = (R_val == 0);
  while (polls < max_polls) {
    int num_wcs = qp_->pollCq();
    if (num_wcs < 0) {
      PCCL_LOG_ERROR("SockConnection::updateAndSync: pollCq error for "
                     "remoteRank %d, wrId %lu.",
                     remoteRank_, current_wr_id);
      throw std::runtime_error("Socket pollCq error.");
    }
    for (int i = 0; i < num_wcs; ++i) {
      // pccl::SockQp::getWcStatus(idx) returns SockWc by value in plugin/sock.h
      SockWc wc = qp_->getWcStatus(i);
      if (wc.wr_id == current_wr_id) {
        if (wc.status != SockStatus::SUCCESS) {
          PCCL_LOG_ERROR("SockConnection::updateAndSync: wrId %lu completed "
                         "with error %d for remoteRank %d.",
                         current_wr_id, static_cast<int>(wc.status),
                         remoteRank_);
          // throw std::runtime_error("Socket operation completed with error.");
        } else {
          PCCL_LOG_INFO("SockConnection updateAndSync: wrId %lu completed "
                        "successfully for remoteRank %d.",
                        current_wr_id, remoteRank_);
        }
        completed = true;
        break;
      }
    }
    if (completed)
      break;
    polls++;
    // TODO: Consider a short sleep/yield if num_wcs is often 0.
  }

  if (!completed) {
    PCCL_LOG_WARN("SockConnection::updateAndSync: Max polls reached for wrId "
                  "%lu, remoteRank %d. Completion not confirmed.",
                  current_wr_id, remoteRank_);
    // Potentially throw timeout error
  }
}

void SockConnection::flush(int64_t timeoutUsec) {
  if (!qp_ || !ctx_) {
    PCCL_LOG_ERROR(
        "SockConnection::flush: QP or Ctx not initialized for remoteRank %d.",
        remoteRank_);
    return;
  }
  PCCL_LOG_INFO(
      "SockConnection flushing connection for remoteRank %d (timeout %lld us).",
      remoteRank_, (long long)timeoutUsec);

  // Similar to IbConnection::flush, this is a best-effort flush.
  // TODO: Implement proper timeout handling.
  int polls = 0;
  const int max_empty_polls_streak = 1000;
  int current_empty_polls = 0;
  int total_wcs_polled = 0;

  while (true) { // TODO: Add timeout check
    int num_wcs = qp_->pollCq();
    if (num_wcs < 0) {
      PCCL_LOG_ERROR("SockConnection::flush: pollCq error for remoteRank %d.",
                     remoteRank_);
      break;
    }

    total_wcs_polled += num_wcs;
    for (int i = 0; i < num_wcs; ++i) {
      SockWc wc = qp_->getWcStatus(i);
      if (wc.status != SockStatus::SUCCESS) {
        PCCL_LOG_WARN("SockConnection::flush: WR %lu completed with error %d "
                      "for remoteRank %d.",
                      wc.wr_id, static_cast<int>(wc.status), remoteRank_);
      }
    }

    if (num_wcs == 0) {
      current_empty_polls++;
      if (current_empty_polls >= max_empty_polls_streak) {
        PCCL_LOG_INFO("SockConnection flush: CQ empty for %d polls. Assuming "
                      "flushed for remoteRank %d.",
                      max_empty_polls_streak, remoteRank_);
        break;
      }
    } else {
      current_empty_polls = 0; // Reset streak
    }
    polls++;
    // TODO: Check timeout here.
  }
  PCCL_LOG_INFO("SockConnection flush completed attempt for remoteRank %d. "
                "Polled %d WCs in %d (total) polls.",
                remoteRank_, total_wcs_polled, polls);
}

Transport SockConnection::transport() {
  return Transport::Ethernet;
} // Corrected from Socket
Transport SockConnection::remoteTransport() {
  return Transport::Ethernet;
} // Corrected
bool SockConnection::connected() {
  return (ctx_ != nullptr && qp_ != nullptr &&
          qp_->getInfo().qpn != 0); // Added qpn check, connect() sets it.
}
uint64_t SockConnection::bandwidth() { return 0; }
uint64_t SockConnection::latency() { return 0; }

} // namespace pccl