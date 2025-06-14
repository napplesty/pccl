#pragma once

#include <chrono>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>

namespace pccl {

enum class LogLevel { DEBUG, INFO, WARNING, ERROR, FATAL };

static LogLevel g_minLogLevel = LogLevel::INFO;
static std::ostream *g_logStream = &std::cerr;
static std::mutex g_logMutex;
static std::unique_ptr<std::ofstream> g_logFileStream;
static std::string g_logBuffer;
static size_t g_maxLogBufferSize = 1024;

inline void flushLog();

inline void setMinLogLevel(LogLevel level) { g_minLogLevel = level; }

inline void setLogStream(std::ostream &stream) { g_logStream = &stream; }

inline bool setLogFile(const std::string &filePath,
                       std::ios_base::openmode mode = std::ios_base::app) {
  g_logFileStream = std::make_unique<std::ofstream>(filePath, mode);
  if (!g_logFileStream->is_open()) {
    std::cerr << "Error: Could not open log file: " << filePath << std::endl;
    g_logFileStream.reset();
    setLogStream(std::cerr);
    return false;
  }
  setLogStream(*g_logFileStream);
  return true;
}

inline void setMaxLogBufferSize(size_t size) {
  std::lock_guard<std::mutex> lock(g_logMutex);
  g_maxLogBufferSize = size;
}

inline const char *logLevelToString(LogLevel level) {
  switch (level) {
  case LogLevel::DEBUG:
    return "DEBUG";
  case LogLevel::INFO:
    return "INFO ";
  case LogLevel::WARNING:
    return "WARN ";
  case LogLevel::ERROR:
    return "ERROR";
  case LogLevel::FATAL:
    return "FATAL";
  default:
    return "?????";
  }
}

inline void internalFlush_locked() {
  if (g_logStream && !g_logBuffer.empty()) {
    try {
      (*g_logStream) << g_logBuffer;
      g_logStream->flush();
      g_logBuffer.clear();
    } catch (const std::exception &e) {
      std::cerr << "Exception during log flush: " << e.what() << std::endl;
      g_logBuffer.clear();
    }
  }
}

inline void flushLog() {
  std::lock_guard<std::mutex> lock(g_logMutex);
  internalFlush_locked();
}

class LogMessage {
public:
  LogMessage(LogLevel level, const char *file, int line) : level_(level) {
    if (level >= g_minLogLevel) {
      prefix_ = formatPrefix(level, file, line);
    }
  }

  ~LogMessage() {
    if (level_ >= g_minLogLevel) {
      std::string message = prefix_ + stream_.str() + "\n";
      bool should_abort = (level_ == LogLevel::FATAL);
      bool force_flush = (level_ >= LogLevel::ERROR);

      {
        std::lock_guard<std::mutex> lock(g_logMutex);
        g_logBuffer += message;

        if (force_flush || g_logBuffer.size() >= g_maxLogBufferSize) {
          internalFlush_locked();
        }
      }

      if (should_abort) {
        std::abort();
      }
    }
  }

  template <typename T> 
  LogMessage &operator<<(const T &&value) {
    stream_ << value;
    return *this;
  }

  LogMessage &operator<<(std::ostream &(*manip)(std::ostream &)) {
    stream_ << manip;
    return *this;
  }

private:
  std::ostringstream stream_;
  std::string prefix_;
  LogLevel level_;

  static std::string formatPrefix(LogLevel level, const char *file, int line) {
    auto now = std::chrono::system_clock::now();
    auto now_c = std::chrono::system_clock::to_time_t(now);
    auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                      now.time_since_epoch()) %
                         1000;

    const char *filename = file;
    const char *last_slash = strrchr(file, '/');
    if (last_slash) {
      filename = last_slash + 1;
    }

    std::ostringstream prefix_ss;
    prefix_ss << std::put_time(std::localtime(&now_c), "%Y-%m-%d %H:%M:%S");
    prefix_ss << '.' << std::setfill('0') << std::setw(3) << now_ms.count();
    prefix_ss << " [" << logLevelToString(level) << "]";
    prefix_ss << " [" << filename << ":" << line << "] ";
    return prefix_ss.str();
  }

  LogMessage(const LogMessage &) = delete;
  LogMessage &operator=(const LogMessage &) = delete;
};

namespace {
struct LogFlusher {
  ~LogFlusher() { ::pccl::flushLog(); }
};
static LogFlusher g_logAtExitFlusher;
} // namespace

#define IS_LOG_LEVEL_ACTIVE(level) (level >= ::pccl::g_minLogLevel)

#define LOG(level)                                                             \
  if (!IS_LOG_LEVEL_ACTIVE(level))                                             \
    ;                                                                          \
  else                                                                         \
    ::pccl::LogMessage(level, __FILE__, __LINE__)

#define LOG_DEBUG LOG(::pccl::LogLevel::DEBUG)
#define LOG_INFO LOG(::pccl::LogLevel::INFO)
#define LOG_WARNING LOG(::pccl::LogLevel::WARNING)
#define LOG_ERROR LOG(::pccl::LogLevel::ERROR)
#define LOG_FATAL LOG(::pccl::LogLevel::FATAL)

#ifndef NDEBUG
#define DLOG(level) LOG(level)
#define DLOG_DEBUG LOG_DEBUG
#define DLOG_INFO LOG_INFO
#define DLOG_WARNING LOG_WARNING
#define DLOG_ERROR LOG_ERROR
#define DLOG_FATAL LOG_FATAL
#else
#define DLOG(level)                                                            \
  if (true) {                                                                  \
  } else                                                                       \
    ::pccl::LogMessage(level, __FILE__, __LINE__)
#define DLOG_DEBUG DLOG(::pccl::LogLevel::DEBUG)
#define DLOG_INFO DLOG(::pccl::LogLevel::INFO)
#define DLOG_WARNING DLOG(::pccl::LogLevel::WARNING)
#define DLOG_ERROR DLOG(::pccl::LogLevel::ERROR)
#define DLOG_FATAL DLOG(::pccl::LogLevel::FATAL)
#endif

} // namespace pccl
