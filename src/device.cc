#include "device.h"

namespace pccl {

AvoidCudaGraphCaptureGuard::AvoidCudaGraphCaptureGuard()
    : mode_(cudaStreamCaptureModeRelaxed) {
  CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode_));
}

AvoidCudaGraphCaptureGuard::~AvoidCudaGraphCaptureGuard() {
  cudaThreadExchangeStreamCaptureMode(&mode_);
}

CudaStreamWithFlags::CudaStreamWithFlags(unsigned int flags) {
  CUDACHECK(cudaStreamCreateWithFlags(&stream_, flags));
  flags_ = flags;
}

CudaStreamWithFlags::CudaStreamWithFlags(cudaStream_t stream) {
  stream_ = stream;
  flags_ = 0x7f7f7f7f;
}

CudaStreamWithFlags::~CudaStreamWithFlags() {
  if (!empty() && flags_ != 0x7f7f7f7f) (void)cudaStreamDestroy(stream_);
}

void CudaStreamWithFlags::set(unsigned int flags) {
  if (!empty()) throw ::std::runtime_error("CudaStreamWithFlags already set");
  CUDACHECK(cudaStreamCreateWithFlags(&stream_, flags));
}

bool CudaStreamWithFlags::empty() const { return stream_ == nullptr; }

void setRWAccess(void *base, size_t size) {
  CUmemAccessDesc accessDesc = {};
  int deviceId;
  CUDACHECK(cudaGetDevice(&deviceId));
  accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  accessDesc.location.id = deviceId;
  accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  CUCHECK(cuMemSetAccess((CUdeviceptr)base, size, &accessDesc, 1));
}

void *gpuCalloc(size_t bytes) {
  AvoidCudaGraphCaptureGuard cgcGuard;
  void *ptr;
  CudaStreamWithFlags stream(cudaStreamNonBlocking);
  CUDACHECK(cudaMalloc(&ptr, bytes));
  CUDACHECK(cudaMemsetAsync(ptr, 0, bytes, stream));
  CUDACHECK(cudaStreamSynchronize(stream));
  return ptr;
}

void *gpuCallocHost(size_t bytes) {
  AvoidCudaGraphCaptureGuard cgcGuard;
  void *ptr;
  CUDACHECK(cudaHostAlloc(&ptr, bytes,
                          cudaHostAllocMapped | cudaHostAllocWriteCombined));
  ::memset(ptr, 0, bytes);
  return ptr;
}

void gpuFree(void *ptr) {
  AvoidCudaGraphCaptureGuard cgcGuard;
  CUDACHECK(cudaFree(ptr));
}

void gpuFreeHost(void *ptr) {
  AvoidCudaGraphCaptureGuard cgcGuard;
  CUDACHECK(cudaFreeHost(ptr));
}

#if defined(USE_HIP)
void *gpuCallocUncached(size_t bytes) {
  AvoidCudaGraphCaptureGuard cgcGuard;
  void *ptr;
  CudaStreamWithFlags stream(cudaStreamNonBlocking);
  CUDACHECK(
      hipExtMallocWithFlags((void **)&ptr, bytes, hipDeviceMallocUncached));
  CUDACHECK(cudaMemsetAsync(ptr, 0, bytes, stream));
  CUDACHECK(cudaStreamSynchronize(stream));
  return ptr;
}
#endif

#if defined(NVLS_SUPPORT)
size_t getMulticastGranularity(size_t size,
                               CUmulticastGranularity_flags granFlag) {
  size_t gran = 0;
  int numDevices = 0;
  CUDACHECK(cudaGetDeviceCount(&numDevices));

  CUmulticastObjectProp prop = {};
  prop.size = size;
  prop.numDevices = numDevices;
  prop.handleTypes =
      (CUmemAllocationHandleType)(CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR |
                                  CU_MEM_HANDLE_TYPE_FABRIC);
  prop.flags = 0;
  CUCHECK(cuMulticastGetGranularity(&gran, &prop, granFlag));
  return gran;
}

void *gpuCallocPhysical(size_t bytes, size_t gran, size_t align) {
  AvoidCudaGraphCaptureGuard cgcGuard;
  int deviceId = -1;
  CUdevice currentDevice;
  CUDACHECK(cudaGetDevice(&deviceId));
  CUCHECK(cuDeviceGet(&currentDevice, deviceId));

  int requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  int isFabricSupported;
  CUCHECK(cuDeviceGetAttribute(&isFabricSupported,
                               CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED,
                               currentDevice));
  if (isFabricSupported) {
    requestedHandleTypes |= CU_MEM_HANDLE_TYPE_FABRIC;
  }
  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.requestedHandleTypes = (CUmemAllocationHandleType)(requestedHandleTypes);
  prop.location.id = currentDevice;

  if (gran == 0) {
    gran = getMulticastGranularity(bytes, CU_MULTICAST_GRANULARITY_RECOMMENDED);
  }

  // allocate physical memory
  CUmemGenericAllocationHandle memHandle;
  size_t nbytes = (bytes + gran - 1) / gran * gran;
  CUresult result = cuMemCreate(&memHandle, nbytes, &prop, 0);
  if (requestedHandleTypes & CU_MEM_HANDLE_TYPE_FABRIC &&
      (result == CUDA_ERROR_NOT_PERMITTED ||
       result == CUDA_ERROR_NOT_SUPPORTED)) {
    requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    prop.requestedHandleTypes = (CUmemAllocationHandleType)requestedHandleTypes;
    CUCHECK(cuMemCreate(&memHandle, nbytes, &prop, 0));
  } else {
    CUCHECK(result);
  }
  nvlsCompatibleMemHandleType = (CUmemAllocationHandleType)requestedHandleTypes;

  if (align == 0) {
    align = getMulticastGranularity(nbytes, CU_MULTICAST_GRANULARITY_MINIMUM);
  }

  void *devicePtr = nullptr;
  CUCHECK(cuMemAddressReserve((CUdeviceptr *)&devicePtr, nbytes, align, 0U, 0));
  CUCHECK(cuMemMap((CUdeviceptr)devicePtr, nbytes, 0, memHandle, 0));
  setRWAccess(devicePtr, nbytes);
  CudaStreamWithFlags stream(cudaStreamNonBlocking);
  CUDACHECK(cudaMemsetAsync(devicePtr, 0, nbytes, stream));
  CUDACHECK(cudaStreamSynchronize(stream));
  return devicePtr;
}

void gpuFreePhysical(void *ptr) {
  AvoidCudaGraphCaptureGuard cgcGuard;
  CUmemGenericAllocationHandle handle;
  size_t size = 0;
  CUCHECK(cuMemRetainAllocationHandle(&handle, ptr));
  CUCHECK(cuMemRelease(handle));
  CUCHECK(cuMemGetAddressRange(NULL, &size, (CUdeviceptr)ptr));
  CUCHECK(cuMemUnmap((CUdeviceptr)ptr, size));
  CUCHECK(cuMemRelease(handle));
  CUCHECK(cuMemAddressFree((CUdeviceptr)ptr, size));
}
#endif

void gpuMemcpyAsync(void *dst, const void *src, size_t bytes,
                    cudaStream_t stream, cudaMemcpyKind kind) {
  AvoidCudaGraphCaptureGuard cgcGuard;
  CUDACHECK(cudaMemcpyAsync(dst, src, bytes, kind, stream));
}

void gpuMemcpy(void *dst, const void *src, size_t bytes, cudaMemcpyKind kind) {
  AvoidCudaGraphCaptureGuard cgcGuard;
  CudaStreamWithFlags stream(cudaStreamNonBlocking);
  CUDACHECK(cudaMemcpyAsync(dst, src, bytes, kind, stream));
  CUDACHECK(cudaStreamSynchronize(stream));
}

bool isNvlsSupported() {
  [[maybe_unused]] static bool result = false;
  [[maybe_unused]] static bool isChecked = false;
#if defined(NVLS_SUPPORT)
  if (!isChecked) {
    int isMulticastSupported;
    CUdevice dev;
    CUCHECK(cuCtxGetDevice(&dev));
    CUCHECK(cuDeviceGetAttribute(&isMulticastSupported,
                                 CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED, dev));
    return isMulticastSupported == 1;
  }
  return result;
#endif
  return false;
}

bool isCuMemMapAllocated([[maybe_unused]] void *ptr) {
#if defined(USE_HIP)
  return false;
#else
  CUmemGenericAllocationHandle handle;
  CUresult result = cuMemRetainAllocationHandle(&handle, ptr);
  if (result != CUDA_SUCCESS) {
    return false;
  }
  CUCHECK(cuMemRelease(handle));
  if (!isNvlsSupported()) {
    throw ::std::runtime_error("cuMemMap is used in env without NVLS support");
  }
  return true;
#endif
}

}  // namespace pccl
