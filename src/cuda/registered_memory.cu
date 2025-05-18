#include <algorithm>

#include "component/logging.h"
#include "config.h"
#include "cuda/registered_memory.h"
#include "device.h"
#include "utils.h"

namespace {
CUmemAllocationHandleType getNvlsMemHandleType() {
#if defined(NVLS_SUPPORT)
  if (::pccl::nvlsCompatibleMemHandleType & CU_MEM_HANDLE_TYPE_FABRIC) {
    return CU_MEM_HANDLE_TYPE_FABRIC;
  } else {
    return CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  }
#else
  throw std::runtime_error(
      "CUDA does not support NVLS. Please ensure your CUDA version supports NVLS to use this "
      "feature.");
#endif
}
}  // namespace

namespace pccl {

static void *gpu_buffer;
static void *host_buffer;
static void *lib_buffer;

RegisteredMemory::Impl::Impl(bool isHostMemory, bool isLibMemory, TransportFlags transports,
                             ConnectionContext::Impl &contextImpl)
    : host_ptr(nullptr),
      device_ptr(nullptr),
      origianl_ptr(nullptr),
      original_rank(0),
      is_host_memory(isHostMemory),
      is_lib_memory(isLibMemory),
      size(0),
      hostHash(getHostHash()),
      pidHash(getPidHash()),
      transports(transports) {
  if (isLibMemory) {
    size = Config::WORKSPACE_SIZE;
  } else {
    if (isHostMemory) {
      size = Config::HOST_BUFFER_SIZE;
    } else {
      size = Config::DEVICE_BUFFER_SIZE;
    }
  }
  if (isHostMemory) {
    host_ptr = gpuCallocHost(size);
#if defined(USE_CUDA)
    CUDACHECK(cudaHostGetDevicePointer(&device_ptr, host_ptr, 0));
#elif defined(USE_HIP)
    device_ptr = host_ptr;
#else
    throw std::runtime_error("Unsupported device type");
#endif
  } else {
    host_ptr = nullptr;
    device_ptr = gpuCalloc(size);
  }
  if (transports.has(Transport::CudaIpc)) {
    TransportInfo transportInfo;
    transportInfo.transport = Transport::CudaIpc;
    void *baseDataPtr;
    size_t baseDataSize;
    CUCHECK(
        cuMemGetAddressRange((CUdeviceptr *)&baseDataPtr, &baseDataSize, (CUdeviceptr)device_ptr));
    this->isCuMemMapAlloc = isCuMemMapAllocated(baseDataPtr);
    if (this->isCuMemMapAlloc) {
      CUmemGenericAllocationHandle handle;
      CUCHECK(cuMemRetainAllocationHandle(&handle, baseDataPtr));
      if (getNvlsMemHandleType() == CU_MEM_HANDLE_TYPE_FABRIC) {
      } else {
        transportInfo.rootPid = getpid();
        if (transportInfo.rootPid < 0) {
          throw std::runtime_error("getpid() failed");
        }
        CUCHECK(cuMemExportToShareableHandle(&transportInfo.fileDesc, handle,
                                             getNvlsMemHandleType(), 0));
        this->fileDesc = transportInfo.fileDesc;
      }
      transportInfo.offsetFromBase = (char *)data - (char *)baseDataPtr;
      CUCHECK(cuMemRelease(handle));
    } else {
      cudaIpcMemHandle_t handle;
      CUDACHECK(cudaIpcGetMemHandle(&handle, baseDataPtr));
      transportInfo.cudaIpcBaseHandle = handle;
      transportInfo.cudaIpcOffsetFromBase = (char *)data - (char *)baseDataPtr;
    }
    this->transportInfos.push_back(transportInfo);
  }
  if ((transports.has(Transport::IB0)) || (transports.has(Transport::IB1))) {
    auto addIb = [&](Transport ibTransport) {
      TransportInfo transportInfo;
      transportInfo.transport = ibTransport;
      const IbMr *mr = contextImpl.getIbContext(ibTransport)->registerMr(data, size);
      transportInfo.ibMr = mr;
      transportInfo.ibLocal = true;
      transportInfo.ibMrInfo = mr->getInfo();
      this->transportInfos.push_back(transportInfo);
    };
    addIb(Transport::IB0);
    addIb(Transport::IB1);
  }
}

PCCL_API RegisteredMemory::RegisteredMemory(int global_buffer_id) {
  TransportFlags transports;
  auto env = getEnv();
  if (!env->ibDevice0.empty()) {
    transports.set(static_cast<size_t>(Transport::IB0));
  }
  if (!env->ibDevice1.empty()) {
    transports.set(static_cast<size_t>(Transport::IB1));
  }
  if (!env->socketAddr.empty()) {
    transports.set(static_cast<size_t>(Transport::Ethernet));
  }
  transports.set(static_cast<size_t>(Transport::CudaIpc));
#if defined(NVLS_SUPPORT)
  transports.set(static_cast<size_t>(Transport::NVLS));
#endif
  if (global_buffer_id < Config::MAX_LIB_BUFFER) {
    auto buffer = GpuBuffer<char>(Config::MAX_LIB_BUFFER_SIZE, true);
    pimpl_ = std::make_shared<Impl>(buffer.data(), Config::MAX_LIB_BUFFER_SIZE, transports,
                                    *contextImpl);
  } else if (global_buffer_id < Config::MAX_LIB_BUFFER + Config::MAX_DEVICE_BUFFER) {
    auto buffer = GpuBuffer<char>(Config::MAX_DEVICE_BUFFER_SIZE, false);
  } else if (global_buffer_id <
             Config::MAX_LIB_BUFFER + Config::MAX_DEVICE_BUFFER + Config::MAX_HOST_BUFFER) {
    auto buffer = GpuBuffer<char>(Config::MAX_HOST_BUFFER_SIZE, true);
  } else {
    throw std::runtime_error("Invalid global buffer id");
  }
}

PCCL_API RegisteredMemory::RegisteredMemory(std::shared_ptr<Impl> pimpl) : pimpl_(pimpl) {}

PCCL_API RegisteredMemory::~RegisteredMemory() = default;

PCCL_API void *RegisteredMemory::data() const { return pimpl_->data; }

PCCL_API void *RegisteredMemory::originalDataPtr() const { return pimpl_->originalDataPtr; }

PCCL_API size_t RegisteredMemory::size() { return pimpl_->size; }

PCCL_API TransportFlags RegisteredMemory::transports() { return pimpl_->transports; }

PCCL_API std::vector<char> RegisteredMemory::serialize() {
  std::vector<char> result;
  std::copy_n(reinterpret_cast<char *>(&pimpl_->originalDataPtr), sizeof(pimpl_->originalDataPtr),
              std::back_inserter(result));
  std::copy_n(reinterpret_cast<char *>(&pimpl_->size), sizeof(pimpl_->size),
              std::back_inserter(result));
  std::copy_n(reinterpret_cast<char *>(&pimpl_->hostHash), sizeof(pimpl_->hostHash),
              std::back_inserter(result));
  std::copy_n(reinterpret_cast<char *>(&pimpl_->pidHash), sizeof(pimpl_->pidHash),
              std::back_inserter(result));
  std::copy_n(reinterpret_cast<char *>(&pimpl_->isCuMemMapAlloc), sizeof(pimpl_->isCuMemMapAlloc),
              std::back_inserter(result));
  std::copy_n(reinterpret_cast<char *>(&pimpl_->transports), sizeof(pimpl_->transports),
              std::back_inserter(result));
  if (pimpl_->transportInfos.size() > static_cast<size_t>(std::numeric_limits<int8_t>::max())) {
    throw std::runtime_error("Too many transport info entries");
  }
  int8_t transportCount = pimpl_->transportInfos.size();
  std::copy_n(reinterpret_cast<char *>(&transportCount), sizeof(transportCount),
              std::back_inserter(result));
  for (auto &entry : pimpl_->transportInfos) {
    std::copy_n(reinterpret_cast<char *>(&entry.transport), sizeof(entry.transport),
                std::back_inserter(result));
    if (entry.transport == Transport::CudaIpc) {
      if (pimpl_->isCuMemMapAlloc) {
        if (getNvlsMemHandleType() == CU_MEM_HANDLE_TYPE_FABRIC) {
          std::copy_n(reinterpret_cast<char *>(&entry.shareableHandle),
                      sizeof(entry.shareableHandle), std::back_inserter(result));
        } else {
          std::copy_n(reinterpret_cast<char *>(&entry.rootPid), sizeof(entry.rootPid),
                      std::back_inserter(result));
          std::copy_n(reinterpret_cast<char *>(&entry.fileDesc), sizeof(entry.fileDesc),
                      std::back_inserter(result));
        }
        std::copy_n(reinterpret_cast<char *>(&entry.offsetFromBase), sizeof(entry.offsetFromBase),
                    std::back_inserter(result));
      } else {
        std::copy_n(reinterpret_cast<char *>(&entry.cudaIpcBaseHandle),
                    sizeof(entry.cudaIpcBaseHandle), std::back_inserter(result));
        std::copy_n(reinterpret_cast<char *>(&entry.cudaIpcOffsetFromBase),
                    sizeof(entry.cudaIpcOffsetFromBase), std::back_inserter(result));
      }
    } else if (entry.transport == Transport::IB0 || entry.transport == Transport::IB1) {
      std::copy_n(reinterpret_cast<char *>(&entry.ibMrInfo), sizeof(entry.ibMrInfo),
                  std::back_inserter(result));
    } else {
      throw std::runtime_error("Unknown transport");
    }
  }
  return result;
}

PCCL_API RegisteredMemory RegisteredMemory::deserialize(const std::vector<char> &data) {
  return RegisteredMemory(std::make_shared<Impl>(data));
}

RegisteredMemory::Impl::Impl(const std::vector<char> &serialization) {
  auto it = serialization.begin();
  std::copy_n(it, sizeof(this->originalDataPtr), reinterpret_cast<char *>(&this->originalDataPtr));
  it += sizeof(this->originalDataPtr);
  std::copy_n(it, sizeof(this->size), reinterpret_cast<char *>(&this->size));
  it += sizeof(this->size);
  std::copy_n(it, sizeof(this->hostHash), reinterpret_cast<char *>(&this->hostHash));
  it += sizeof(this->hostHash);
  std::copy_n(it, sizeof(this->pidHash), reinterpret_cast<char *>(&this->pidHash));
  it += sizeof(this->pidHash);
  std::copy_n(it, sizeof(this->isCuMemMapAlloc), reinterpret_cast<char *>(&this->isCuMemMapAlloc));
  it += sizeof(this->isCuMemMapAlloc);
  std::copy_n(it, sizeof(this->transports), reinterpret_cast<char *>(&this->transports));
  it += sizeof(this->transports);
  int8_t transportCount;
  std::copy_n(it, sizeof(transportCount), reinterpret_cast<char *>(&transportCount));
  it += sizeof(transportCount);
  for (int i = 0; i < transportCount; ++i) {
    TransportInfo transportInfo;
    std::copy_n(it, sizeof(transportInfo.transport),
                reinterpret_cast<char *>(&transportInfo.transport));
    it += sizeof(transportInfo.transport);
    if (transportInfo.transport == Transport::CudaIpc) {
      if (this->isCuMemMapAlloc) {
        if (getNvlsMemHandleType() == CU_MEM_HANDLE_TYPE_FABRIC) {
          std::copy_n(it, sizeof(transportInfo.shareableHandle),
                      reinterpret_cast<char *>(&transportInfo.shareableHandle));
          it += sizeof(transportInfo.shareableHandle);
        } else {
          std::copy_n(it, sizeof(transportInfo.rootPid),
                      reinterpret_cast<char *>(&transportInfo.rootPid));
          it += sizeof(transportInfo.rootPid);
          std::copy_n(it, sizeof(transportInfo.fileDesc),
                      reinterpret_cast<char *>(&transportInfo.fileDesc));
          it += sizeof(transportInfo.fileDesc);
        }
        std::copy_n(it, sizeof(transportInfo.offsetFromBase),
                    reinterpret_cast<char *>(&transportInfo.offsetFromBase));
        it += sizeof(transportInfo.offsetFromBase);
      } else {
        std::copy_n(it, sizeof(transportInfo.cudaIpcBaseHandle),
                    reinterpret_cast<char *>(&transportInfo.cudaIpcBaseHandle));
        it += sizeof(transportInfo.cudaIpcBaseHandle);
        std::copy_n(it, sizeof(transportInfo.cudaIpcOffsetFromBase),
                    reinterpret_cast<char *>(&transportInfo.cudaIpcOffsetFromBase));
        it += sizeof(transportInfo.cudaIpcOffsetFromBase);
      }
    } else if (transportInfo.transport == Transport::IB0 ||
               transportInfo.transport == Transport::IB1) {
      std::copy_n(it, sizeof(transportInfo.ibMrInfo),
                  reinterpret_cast<char *>(&transportInfo.ibMrInfo));
      it += sizeof(transportInfo.ibMrInfo);
      transportInfo.ibLocal = false;
    } else {
      throw std::runtime_error("Unknown transport");
    }
    this->transportInfos.push_back(transportInfo);
  }
  if (it != serialization.end()) {
    throw std::runtime_error("Serialization failed");
  }

  // Next decide how to set this->data
  if (getHostHash() == this->hostHash && getPidHash() == this->pidHash) {
    // The memory is local to the process, so originalDataPtr is valid as is
    this->data = this->originalDataPtr;
  } else if (transports.has(Transport::CudaIpc) && getHostHash() == this->hostHash) {
    // The memory is local to the machine but not to the process, so we need to open the CUDA IPC
    // handle
    auto entry = getTransportInfo(Transport::CudaIpc);
    void *base;
    if (this->isCuMemMapAlloc) {
#if defined(NVLS_SUPPORT)
      CUmemGenericAllocationHandle handle;
      if (getNvlsMemHandleType() == CU_MEM_HANDLE_TYPE_FABRIC) {
        CUCHECK(
            cuMemImportFromShareableHandle(&handle, entry.shareableHandle, getNvlsMemHandleType()));
      } else {
        int rootPidFd = syscall(SYS_pidfd_open, entry.rootPid, 0);
        if (rootPidFd < 0) {
          throw std::runtime_error("pidfd_open() failed");
        }
        int fd = syscall(SYS_pidfd_getfd, rootPidFd, entry.fileDesc, 0);
        if (fd < 0) {
          throw std::runtime_error("pidfd_getfd() failed");
        }
        CUCHECK(cuMemImportFromShareableHandle(&handle, reinterpret_cast<void *>(fd),
                                               CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));
        close(rootPidFd);
        close(fd);
      }
      size_t minGran = getMulticastGranularity(size, CU_MULTICAST_GRANULARITY_MINIMUM);
      size_t recommendedGran = getMulticastGranularity(size, CU_MULTICAST_GRANULARITY_RECOMMENDED);
      size_t size = (this->size + recommendedGran - 1) / recommendedGran * recommendedGran;
      CUCHECK(cuMemAddressReserve((CUdeviceptr *)&base, size, minGran, 0, 0));
      CUCHECK(cuMemMap((CUdeviceptr)base, size, 0, handle, 0));
      setRWAccess(base, size);
      this->data = static_cast<char *>(base) + entry.offsetFromBase;
#else
      throw ::std::runtime_error(
          "CUDA does not support NVLS. Please ensure your CUDA version supports NVLS to use this "
          "feature.");
#endif
    } else {
      CUDACHECK(
          cudaIpcOpenMemHandle(&base, entry.cudaIpcBaseHandle, cudaIpcMemLazyEnablePeerAccess));
      this->data = static_cast<char *>(base) + entry.cudaIpcOffsetFromBase;
    }
  } else {
    // No valid data pointer can be set
    this->data = nullptr;
  }
}

RegisteredMemory::Impl::~Impl() {
  if (data && transports.has(Transport::CudaIpc) && getHostHash() == this->hostHash &&
      getPidHash() != this->pidHash) {
    void *base =
        static_cast<char *>(data) - getTransportInfo(Transport::CudaIpc).cudaIpcOffsetFromBase;
    if (this->isCuMemMapAlloc) {
      CUmemGenericAllocationHandle handle;
      size_t size = 0;
      CUCHECK(cuMemRetainAllocationHandle(&handle, base));
      CUCHECK(cuMemRelease(handle));
      CUCHECK(cuMemGetAddressRange(NULL, &size, (CUdeviceptr)base));
      CUCHECK(cuMemUnmap((CUdeviceptr)base, size));
      CUCHECK(cuMemRelease(handle));
      CUCHECK(cuMemAddressFree((CUdeviceptr)base, size));
      if (getNvlsMemHandleType() == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR && fileDesc >= 0) {
        close(fileDesc);
      }
    } else {
      cudaError_t err = cudaIpcCloseMemHandle(base);
      if (err != cudaSuccess) {
        LOG_ERROR << "Failed to close CUDA IPC handle at pointer " << base << ": "
                  << cudaGetErrorString(err) << std::endl;
      }
    }
    data = nullptr;
    fileDesc = -1;
  }
}

const TransportInfo &RegisteredMemory::Impl::getTransportInfo(Transport transport) const {
  for (auto &entry : transportInfos) {
    if (entry.transport == transport) {
      return entry;
    }
  }
  throw std::runtime_error("Transport data not found");
}
}  // namespace pccl