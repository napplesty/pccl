#pragma once

#include <cstddef>
#include <memory>
#include <utility>

#if defined(USE_HIP) || defined(USE_CUDA)
#define PCCL_CUDA_DEVICE_COMPILE
#define PCCL_CUDA_DEVICE_INLINE __forceinline__ __device__
#define PCCL_CUDA_HOST_DEVICE_INLINE __forceinline__ __host__ __device__
#else
// For other kind of device
#endif

#if defined(USE_HIP)

#include <hip/hip_runtime.h>

using cudaError_t = hipError_t;
using cudaGraph_t = hipGraph_t;
using cudaGraphExec_t = hipGraphExec_t;
using cudaDeviceProp = hipDeviceProp_t;
using cudaStream_t = hipStream_t;
using cudaStreamCaptureMode = hipStreamCaptureMode;
using cudaMemcpyKind = hipMemcpyKind;
using cudaIpcMemHandle_t = hipIpcMemHandle_t;

using CUresult = hipError_t;
using CUdeviceptr = hipDeviceptr_t;
using CUmemGenericAllocationHandle = hipMemGenericAllocationHandle_t;
using CUmemAllocationProp = hipMemAllocationProp;
using CUmemAccessDesc = hipMemAccessDesc;
using CUmemAllocationHandleType = hipMemAllocationHandleType;

constexpr auto cudaSuccess = hipSuccess;
constexpr auto cudaStreamNonBlocking = hipStreamNonBlocking;
constexpr auto cudaStreamCaptureModeGlobal = hipStreamCaptureModeGlobal;
constexpr auto cudaStreamCaptureModeRelaxed = hipStreamCaptureModeRelaxed;
constexpr auto cudaHostAllocMapped = hipHostMallocMapped;
constexpr auto cudaHostAllocWriteCombined = hipHostMallocWriteCombined;
constexpr auto cudaMemcpyDefault = hipMemcpyDefault;
constexpr auto cudaMemcpyDeviceToDevice = hipMemcpyDeviceToDevice;
constexpr auto cudaMemcpyHostToDevice = hipMemcpyHostToDevice;
constexpr auto cudaMemcpyDeviceToHost = hipMemcpyDeviceToHost;
constexpr auto cudaIpcMemLazyEnablePeerAccess = hipIpcMemLazyEnablePeerAccess;

constexpr auto CU_MEM_ALLOCATION_TYPE_PINNED = hipMemAllocationTypePinned;
constexpr auto CU_MEM_LOCATION_TYPE_DEVICE = hipMemLocationTypeDevice;
constexpr auto CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR = hipMemHandleTypePosixFileDescriptor;
constexpr auto CU_MEM_ACCESS_FLAGS_PROT_READWRITE = hipMemAccessFlagsProtReadWrite;

#ifndef CUDA_SUCCESS
#define CUDA_SUCCESS hipSuccess
#endif  // CUDA_SUCCESS

#define cudaGetErrorString(...) hipGetErrorString(__VA_ARGS__)
#define cudaGetDevice(...) hipGetDevice(__VA_ARGS__)
#define cudaGetDeviceCount(...) hipGetDeviceCount(__VA_ARGS__)
#define cudaGetDeviceProperties(...) hipGetDeviceProperties(__VA_ARGS__)
#define cudaGetLastError(...) hipGetLastError(__VA_ARGS__)
#define cudaSetDevice(...) hipSetDevice(__VA_ARGS__)
#define cudaDeviceSynchronize(...) hipDeviceSynchronize(__VA_ARGS__)
#define cudaDeviceGetPCIBusId(...) hipDeviceGetPCIBusId(__VA_ARGS__)
#define cudaHostAlloc(...) hipHostMalloc(__VA_ARGS__)
#define cudaMalloc(...) hipMalloc(__VA_ARGS__)
#define cudaFree(...) hipFree(__VA_ARGS__)
#define cudaFreeHost(...) hipHostFree(__VA_ARGS__)
#define cudaMemset(...) hipMemset(__VA_ARGS__)
#define cudaMemsetAsync(...) hipMemsetAsync(__VA_ARGS__)
#define cudaMemcpy(...) hipMemcpy(__VA_ARGS__)
#define cudaMemcpyAsync(...) hipMemcpyAsync(__VA_ARGS__)
#define cudaMemcpyToSymbol(...) hipMemcpyToSymbol(__VA_ARGS__)
#define cudaMemcpyToSymbolAsync(...) hipMemcpyToSymbolAsync(__VA_ARGS__)
#define cudaStreamCreate(...) hipStreamCreate(__VA_ARGS__)
#define cudaStreamCreateWithFlags(...) hipStreamCreateWithFlags(__VA_ARGS__)
#define cudaStreamSynchronize(...) hipStreamSynchronize(__VA_ARGS__)
#define cudaStreamBeginCapture(...) hipStreamBeginCapture(__VA_ARGS__)
#define cudaStreamEndCapture(...) hipStreamEndCapture(__VA_ARGS__)
#define cudaStreamDestroy(...) hipStreamDestroy(__VA_ARGS__)
#define cudaGraphInstantiate(...) hipGraphInstantiate(__VA_ARGS__)
#define cudaGraphLaunch(...) hipGraphLaunch(__VA_ARGS__)
#define cudaGraphDestroy(...) hipGraphDestroy(__VA_ARGS__)
#define cudaGraphExecDestroy(...) hipGraphExecDestroy(__VA_ARGS__)
#define cudaThreadExchangeStreamCaptureMode(...) hipThreadExchangeStreamCaptureMode(__VA_ARGS__)
#define cudaIpcGetMemHandle(...) hipIpcGetMemHandle(__VA_ARGS__)
#define cudaIpcOpenMemHandle(...) hipIpcOpenMemHandle(__VA_ARGS__)
#define cudaIPCCL_CUDAoseMemHandle(...) hipIPCCL_CUDAoseMemHandle(__VA_ARGS__)

#define cuGetErrorString(...) hipDrvGetErrorString(__VA_ARGS__)
#define cuMemAddressReserve(...) hipMemAddressReserve(__VA_ARGS__)
#define cuMemAddressFree(...) hipMemAddressFree(__VA_ARGS__)
#define cuMemGetAddressRange(...) hipMemGetAddressRange(__VA_ARGS__)
#define cuMemCreate(...) hipMemCreate(__VA_ARGS__)
#define cuMemRelease(...) hipMemRelease(__VA_ARGS__)
#define cuMemSetAccess(...) hipMemSetAccess(__VA_ARGS__)
#define cuMemMap(...) hipMemMap(__VA_ARGS__)
#define cuMemUnmap(...) hipMemUnmap(__VA_ARGS__)
#define cuMemRetainAllocationHandle(...) hipMemRetainAllocationHandle(__VA_ARGS__)
#define cuMemExportToShareableHandle(...) hipMemExportToShareableHandle(__VA_ARGS__)
#define cuMemImportFromShareableHandle(...) hipMemImportFromShareableHandle(__VA_ARGS__)

#define __syncshm() asm volatile("s_waitcnt lgkmcnt(0) \n s_barrier");

constexpr auto memoryOrderRelaxed = __ATOMIC_RELAXED;
constexpr auto memoryOrderAcquire = __ATOMIC_ACQUIRE;
constexpr auto memoryOrderRelease = __ATOMIC_RELEASE;
constexpr auto memoryOrderAcqRel = __ATOMIC_ACQ_REL;
constexpr auto memoryOrderSeqCst = __ATOMIC_SEQ_CST;

namespace pccl {

template <typename T>
PCCL_CUDA_HOST_DEVICE_INLINE T atomicLoad(const T *ptr, int memoryOrder) {
  return __atomic_load_n(ptr, memoryOrder);
}

template <typename T>
PCCL_CUDA_HOST_DEVICE_INLINE void atomicStore(T *ptr, const T &val, int memoryOrder) {
  __atomic_store_n(ptr, val, memoryOrder);
}

template <typename T>
PCCL_CUDA_HOST_DEVICE_INLINE T atomicFetchAdd(T *ptr, const T &val, int memoryOrder) {
  return __atomic_fetch_add(ptr, val, memoryOrder);
}

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
using __bfloat16 = __hip_bfloat16;
using __bfloat162 = __hip_bfloat162;
using __float16 = __half;
using __float162 = __half2;

}  // namespace pccl

extern "C" __device__ void __assert_fail(const char *__assertion, const char *__file,
                                         unsigned int __line, const char *__function);

#elif defined(USE_CUDA)

#include <cuda.h>
#include <cuda_runtime.h>

#include <cuda/atomic>
#define __syncshm() __syncthreads();

constexpr cuda::memory_order memoryOrderRelaxed = cuda::memory_order_relaxed;
constexpr cuda::memory_order memoryOrderAcquire = cuda::memory_order_acquire;
constexpr cuda::memory_order memoryOrderRelease = cuda::memory_order_release;
constexpr cuda::memory_order memoryOrderAcqRel = cuda::memory_order_acq_rel;
constexpr cuda::memory_order memoryOrderSeqCst = cuda::memory_order_seq_cst;

namespace pccl {

template <typename T>
PCCL_CUDA_HOST_DEVICE_INLINE T atomicLoad(T *ptr, cuda::memory_order memoryOrder) {
  return cuda::atomic_ref<T, cuda::thread_scope_system>{*ptr}.load(memoryOrder);
}

template <typename T>
PCCL_CUDA_HOST_DEVICE_INLINE void atomicStore(T *ptr, const T &val,
                                              cuda::memory_order memoryOrder) {
  cuda::atomic_ref<T, cuda::thread_scope_system>{*ptr}.store(val, memoryOrder);
}

template <typename T>
PCCL_CUDA_HOST_DEVICE_INLINE T atomicFetchAdd(T *ptr, const T &val,
                                              cuda::memory_order memoryOrder) {
  return cuda::atomic_ref<T, cuda::thread_scope_system>{*ptr}.fetch_add(val, memoryOrder);
}

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime_api.h>
using __bfloat16 = __nv_bfloat16;
using __bfloat162 = __nv_bfloat162;
using __float16 = __nv_half;
using __float162 = __nv_half2;
using __float8_e4m3 = __nv_fp8_e4m3;
using __float8_e5m2 = __nv_fp8_e5m2;
using __float8_e4m32 = __nv_fp8x2_e4m3;
using __float8_e5m22 = __nv_fp8x2_e5m2;
using __float8_e4m34 = __nv_fp8x4_e4m3;
using __float8_e5m24 = __nv_fp8x4_e5m2;

#define NVLS_SUPPORT

}  // namespace pccl

#endif

#define POLL_MAYBE_JAILBREAK(__cond, __max_spin_cnt)                     \
  do {                                                                   \
    int64_t __spin_cnt = 0;                                              \
    while (__cond) {                                                     \
      if (__max_spin_cnt >= 0 && __spin_cnt++ == __max_spin_cnt) {       \
        __assert_fail(#__cond, __FILE__, __LINE__, __PRETTY_FUNCTION__); \
        break; /* Defensively break after assert */                      \
      }                                                                  \
      /* Optional: Add a yield/pause instruction here if appropriate */  \
    }                                                                    \
  } while (0);

#define OR_POLL_MAYBE_JAILBREAK(__cond1, __cond2, __max_spin_cnt)              \
  do {                                                                         \
    int64_t __spin_cnt = 0;                                                    \
    /* Loop while both conditions are true */                                  \
    while ((__cond1) && (__cond2)) {                                           \
      /* Check spin count *before* potentially infinite spinning */            \
      if (__max_spin_cnt >= 0 && __spin_cnt++ == __max_spin_cnt) {             \
        /* Conditions failed to become false within the spin limit */          \
        __assert_fail("(" #__cond1 ") && (" #__cond2 ")", __FILE__, __LINE__,  \
                      __PRETTY_FUNCTION__);                                    \
        /* __assert_fail might not return, but break defensively */            \
        break;                                                                 \
      }                                                                        \
      /* Optional: Add a yield/pause instruction here if appropriate for       \
       * target */                                                             \
      /* e.g., __asm__ volatile("pause"); for x86 */                           \
      /* e.g., __nanosleep(1); // Caution: May not be suitable for GPU kernels \
       */                                                                      \
    }                                                                          \
  } while (0);

#if defined(PCCL_CUDA_DEVICE_COMPILE)

#define CUDACHECK(cmd)                                                                         \
  do {                                                                                         \
    cudaError_t err = cmd;                                                                     \
    if (err != cudaSuccess) {                                                                  \
      throw ::std::runtime_error(::std::string("Call to " #cmd " failed. ") + __FILE__ + ":" + \
                                 ::std::to_string(__LINE__));                                  \
    }                                                                                          \
  } while (false)

#define CUCHECK(cmd)                                                                           \
  do {                                                                                         \
    CUresult err = cmd;                                                                        \
    if (err != CUDA_SUCCESS) {                                                                 \
      throw ::std::runtime_error(::std::string("Call to " #cmd " failed. ") + __FILE__ + ":" + \
                                 ::std::to_string(__LINE__));                                  \
    }                                                                                          \
  } while (false)

namespace pccl {

struct AvoidCudaGraphCaptureGuard {
  AvoidCudaGraphCaptureGuard();
  ~AvoidCudaGraphCaptureGuard();
  cudaStreamCaptureMode mode_;
};

struct CudaStreamWithFlags {
  CudaStreamWithFlags() : stream_(nullptr) {}
  CudaStreamWithFlags(unsigned int flags);
  CudaStreamWithFlags(cudaStream_t stream);
  ~CudaStreamWithFlags();
  void set(unsigned int flags);
  bool empty() const;
  operator cudaStream_t() const { return stream_; }
  cudaStream_t stream_;
  unsigned int flags_;
};

void setRWAccess(void *base, size_t size);
void *gpuCalloc(size_t size);
void *gpuCallocHost(size_t size);
void gpuFree(void *ptr);
void gpuFreeHost(void *ptr);
void gpuMemcpy(void *dst, const void *src, size_t size, cudaMemcpyKind kind = cudaMemcpyDefault);
void gpuMemcpyAsync(void *dst, const void *src, size_t size,
                    cudaMemcpyKind kind = cudaMemcpyDefault, cudaStream_t stream = 0);

#if defined(USE_HIP)
void *gpuCallocUncached(size_t bytes);
#elif defined(NVLS_SUPPORT)
extern CUmemAllocationHandleType nvlsCompatibleMemHandleType;
void *gpuCallocPhysical(size_t bytes, size_t gran = 0, size_t align = 0);
void gpuFreePhysical(void *ptr);
#endif

template <class T, class Deleter, class Memory, typename Alloc, typename... Args>
Memory safeAlloc(Alloc alloc, size_t nelems, Args &&...args) {
  T *ptr = nullptr;
  try {
    ptr = reinterpret_cast<T *>(alloc(nelems * sizeof(T), ::std::forward<Args>(args)...));
  } catch (...) {
    if (ptr) {
      Deleter()(ptr);
    }
    throw;
  }
  return Memory(ptr, Deleter());
}

template <class T = void>
struct GpuDeleter {
  void operator()(void *ptr) { gpuFree(ptr); }
};

template <class T = void>
struct GpuHostDeleter {
  void operator()(void *ptr) { gpuFreeHost(ptr); }
};

template <class T>
using UniqueGpuPtr = ::std::unique_ptr<T, GpuDeleter<T>>;

template <class T>
using UniqueGpuHostPtr = ::std::unique_ptr<T, GpuHostDeleter<T>>;

template <class T>
auto gpuCallocShared(size_t nelems = 1) {
  return safeAlloc<T, GpuDeleter<T>, ::std::shared_ptr<T>>(gpuCalloc, nelems);
}

template <class T>
auto gpuCallocUnique(size_t nelems = 1) {
  return safeAlloc<T, GpuDeleter<T>, UniqueGpuPtr<T>>(gpuCalloc, nelems);
}

template <class T>
auto gpuCallocHostShared(size_t nelems = 1) {
  return safeAlloc<T, GpuHostDeleter<T>, ::std::shared_ptr<T>>(gpuCallocHost, nelems);
}

template <class T>
auto gpuCallocHostUnique(size_t nelems = 1) {
  return safeAlloc<T, GpuHostDeleter<T>, UniqueGpuHostPtr<T>>(gpuCallocHost, nelems);
}

#if defined(USE_HIP)

template <class T>
auto gpuCallocUncachedShared(size_t nelems = 1) {
  return safeAlloc<T, GpuDeleter<T>, ::std::shared_ptr<T>>(gpuCallocUncached, nelems);
}

template <class T>
auto gpuCallocUncachedUnique(size_t nelems = 1) {
  return safeAlloc<T, GpuDeleter<T>, UniqueGpuPtr<T>>(gpuCallocUncached, nelems);
}

#elif defined(NVLS_SUPPORT)

template <class T = void>
struct GpuPhysicalDeleter {
  void operator()(void *ptr) { gpuFreePhysical(ptr); }
};

template <class T>
using UniqueGpuPhysicalPtr = ::std::unique_ptr<T, GpuPhysicalDeleter<T>>;

template <class T>
auto gpuCallocPhysicalShared(size_t nelems = 1, size_t gran = 0, size_t align = 0) {
  return safeAlloc<T, GpuPhysicalDeleter<T>, ::std::shared_ptr<T>>(gpuCallocPhysical, nelems, gran,
                                                                   align);
}

template <class T>
auto gpuCallocPhysicalUnique(size_t nelems = 1, size_t gran = 0, size_t align = 0) {
  return safeAlloc<T, GpuPhysicalDeleter<T>, UniqueGpuPhysicalPtr<T>>(gpuCallocPhysical, nelems,
                                                                      gran, align);
}

size_t getMulticastGranularity(size_t size, CUmulticastGranularity_flags granFlag);

#endif

template <class T = char>
void gpuMemcpyAsync(T *dst, const T *src, size_t nelems, cudaStream_t stream,
                    cudaMemcpyKind kind = cudaMemcpyDefault) {
  gpuMemcpyAsync(dst, src, nelems * sizeof(T), stream, kind);
}

template <class T = char>
void gpuMemcpy(T *dst, const T *src, size_t nelems, cudaMemcpyKind kind = cudaMemcpyDefault) {
  gpuMemcpy(dst, src, nelems * sizeof(T), kind);
}

bool isNvlsSupported();
bool isCuMemMapAllocated([[maybe_unused]] void *ptr);

template <class T = char>
class GpuBuffer {
 public:
  GpuBuffer(size_t nelems, bool host_memory = false) : host_memory_(host_memory), nelems_(nelems) {
    if (nelems == 0) {
      bytes_ = 0;
      return;
    }
    CUDACHECK(cudaGetDevice(&deviceId_));
    if (host_memory) {
      memory_ = gpuCallocHostShared<T>(nelems);
      return;
    }
#if defined(NVLS_SUPPORT)
    if (isNvlsSupported()) {
      size_t gran =
          getMulticastGranularity(nelems * sizeof(T), CU_MULTICAST_GRANULARITY_RECOMMENDED);
      bytes_ = (nelems * sizeof(T) + gran - 1) / gran * gran / sizeof(T) * sizeof(T);
      memory_ = gpuCallocPhysicalShared<T>(nelems, gran);
      return;
    }
#endif  // NVLS_SUPPORT
    bytes_ = nelems * sizeof(T);
#if defined(USE_HIP)
    memory_ = gpuCallocUncachedShared<T>(nelems);
#else
    memory_ = gpuCallocShared<T>(nelems);
#endif
  }

  inline size_t nelems() const { return nelems_; }
  inline size_t bytes() const { return bytes_; }
  inline ::std::shared_ptr<T> memory() { return memory_; }
  inline T *data() { return memory_.get(); }
  inline int deviceId() const { return deviceId_; }
  inline bool isHostMemory() const { return host_memory_; }

 private:
  bool host_memory_;
  size_t nelems_;
  size_t bytes_;
  int deviceId_;
  ::std::shared_ptr<T> memory_;
};

GpuBuffer<char> &getGlobalBuffers(int global_buffer_id);

}  // namespace pccl
#endif
