#pragma once

#include <exception>
#include <string>
#include <sstream>
#include <cassert>
#include <format>
#include <source_location>

#ifdef __CUDACC__
#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>
#endif

namespace pccl {

class Exception : public std::exception {
public:
  Exception(std::source_location location, const std::string& msg)
    : m_location(location), m_msg(msg) {
    format_what();
  }

  virtual ~Exception() noexcept = default;

  const char* what() const noexcept override {
    return m_what.c_str();
  }

  const std::string& message() const noexcept {
    return m_msg;
  }

  const char* file() const noexcept {
    return m_location.file_name();
  }

  int line() const noexcept {
    return m_location.line();
  }

  const char* function() const noexcept {
    return m_location.function_name();
  }

protected:
  void format_what() {
    if (m_location.function_name() && m_location.function_name()[0] != '\0') {
      m_what = std::format("Exception at {}:{} in {}: {}", 
        m_location.file_name(), m_location.line(), 
        m_location.function_name(), m_msg);
    } else {
      m_what = std::format("Exception at {}:{}: {}", 
        m_location.file_name(), m_location.line(), m_msg);
    }
  }

  std::source_location m_location;
  std::string m_msg;
  std::string m_what;
};

class RuntimeException : public Exception {
public:
  RuntimeException(std::source_location location, const std::string& msg)
    : Exception(location, msg) {}
};

class LogicException : public Exception {
public:
  LogicException(std::source_location location, const std::string& msg)
    : Exception(location, msg) {}
};

class InvalidArgumentException : public LogicException {
public:
  InvalidArgumentException(std::source_location location, const std::string& msg)
    : LogicException(location, msg) {}
};

class OutOfRangeException : public LogicException {
public:
  OutOfRangeException(std::source_location location, const std::string& msg)
    : LogicException(location, msg) {}
};

class BadAllocException : public RuntimeException {
public:
  BadAllocException(std::source_location location, const std::string& msg)
    : RuntimeException(location, msg) {}
};

#ifdef __CUDACC__

class CudaException : public Exception {
public:
  CudaException(std::source_location location, 
                const std::string& msg, const std::string& cudaError)
    : Exception(location, msg), m_cudaError(cudaError) {
    format_cuda_what();
  }

  const std::string& cudaError() const noexcept {
    return m_cudaError;
  }

private:
  void format_cuda_what() {
    m_what += std::format(" [CUDA Error: {}]", m_cudaError);
  }

  std::string m_cudaError;
};

inline std::string getCudaErrorString(CUresult error) {
  const char* errorStr;
  cuGetErrorString(error, &errorStr);
  return errorStr ? std::string(errorStr) : std::format("Unknown CUDA driver error: {}", static_cast<int>(error));
}

inline std::string getCudaErrorString(cudaError_t error) {
  return cudaGetErrorString(error);
}

inline std::string getNvrtcErrorString(nvrtcResult error) {
  return nvrtcGetErrorString(error);
}

#endif // __CUDACC__

template<typename... Args>
std::string format_message(Args&&... args) {
  std::ostringstream oss;
  (oss << ... << std::forward<Args>(args));
  return oss.str();
}

#define PCCL_THROW(ExceptionType, ...) \
  throw ExceptionType(std::source_location::current(), ::pccl::format_message(__VA_ARGS__))

#define PCCL_THROW_IF(condition, ExceptionType, ...) \
  if (condition) { \
    throw ExceptionType(std::source_location::current(), ::pccl::format_message(__VA_ARGS__)); \
  }

#ifdef NDEBUG
  #define PCCL_ASSERT(condition, ...) \
    do { \
      if (!(condition)) { \
        throw ::pccl::LogicException(std::source_location::current(), \
            ::pccl::format_message("Assertion failed: ", #condition, " - ", __VA_ARGS__)); \
      } \
    } while (0)
#else
  #define PCCL_ASSERT(condition, ...) \
    do { \
      if (!(condition)) { \
        std::cerr << "Assertion failed: " << #condition << " - " << ::pccl::format_message(__VA_ARGS__) << std::endl; \
        assert(condition); \
      } \
    } while (0)
#endif

#define PCCL_ASSERT_ARG(condition, ...) \
  PCCL_THROW_IF(!(condition), ::pccl::InvalidArgumentException, __VA_ARGS__)

#define PCCL_ASSERT_RANGE(condition, ...) \
  PCCL_THROW_IF(!(condition), ::pccl::OutOfRangeException, __VA_ARGS__)

#if defined(__GNUC__) || defined(__clang__)
  #define PCCL_UNREACHABLE() \
    do { \
      assert(false && "Unreachable code reached"); \
      __builtin_unreachable(); \
    } while (0)
#else
  #define PCCL_UNREACHABLE() \
    do { \
      assert(false && "Unreachable code reached"); \
    } while (0)
#endif

#define PCCL_STATIC_ASSERT(condition, msg) static_assert(condition, msg)

#define PCCL_HOST_ASSERT(condition, ...) \
  PCCL_ASSERT(condition, __VA_ARGS__)

#ifdef __CUDACC__

#define PCCL_CUDA_DRIVER_CHECK(err) \
  do { \
    auto result = (err); \
    if (result != CUDA_SUCCESS) { \
      throw ::pccl::CudaException(std::source_location::current(), \
        ::pccl::format_message("CUDA driver error: ", result), \
        ::pccl::getCudaErrorString(result)); \
    } \
  } while (0)

#define PCCL_CUDA_RUNTIME_CHECK(err) \
  do { \
    auto result = (err); \
    if (result != cudaSuccess) { \
      throw ::pccl::CudaException(std::source_location::current(), \
        ::pccl::format_message("CUDA runtime error: ", result), \
        ::pccl::getCudaErrorString(result)); \
    } \
  } while (0)

#define PCCL_NVRTC_CHECK(err) \
  do { \
    auto result = (err); \
    if (result != NVRTC_SUCCESS) { \
      throw ::pccl::CudaException(std::source_location::current(), \
        ::pccl::format_message("NVRTC error: ", result), \
        ::pccl::getNvrtcErrorString(result)); \
    } \
  } while (0)

#define PCCL_CHECK_LAST_CUDA_ERROR() \
  do { \
    auto err = cudaGetLastError(); \
    if (err != cudaSuccess) { \
      throw ::pccl::CudaException(std::source_location::current(), \
        "CUDA kernel execution error", \
        ::pccl::getCudaErrorString(err)); \
    } \
  } while (0)

#define PCCL_CUDA_SYNC_CHECK() \
  do { \
    auto err = cudaDeviceSynchronize(); \
    if (err != cudaSuccess) { \
      throw ::pccl::CudaException(std::source_location::current(), \
        "CUDA device synchronization error", \
        ::pccl::getCudaErrorString(err)); \
    } \
  } while (0)

#endif // __CUDACC__

} // namespace pccl

