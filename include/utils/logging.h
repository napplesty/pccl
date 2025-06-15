#pragma once

#include <iostream>
#include <memory>
#include <mutex>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_sinks.h>
#include <spdlog/spdlog.h>
#include <string>

namespace pccl {

enum class LogLevel { DEBUG, INFO, WARNING, ERROR, FATAL };

class LoggerWrapper {
public:
  static void init();
  static void flushLog();
  static void log(LogLevel level, const char *file, int line,
                  const std::string &message);

private:
  static std::shared_ptr<spdlog::logger> logger_;
  static std::shared_ptr<spdlog::sinks::sink> console_sink_;
  static std::shared_ptr<spdlog::sinks::sink> file_sink_;
  static std::mutex sink_mutex_;
  static std::once_flag init_flag_;
  static std::unordered_map<LogLevel, spdlog::level::level_enum> level_map_;
};

class LogMessage {
public:
  LogMessage(LogLevel level, const char *file, int line)
      : level_(level), file_(file), line_(line) {}

  ~LogMessage() { LoggerWrapper::log(level_, file_, line_, stream_.str()); }

  template <typename T> LogMessage &operator<<(const T &value) {
    stream_ << value;
    return *this;
  }

  LogMessage &operator<<(std::ostream &(*manip)(std::ostream &)) {
    stream_ << manip;
    return *this;
  }

private:
  std::ostringstream stream_;
  LogLevel level_;
  const char *file_;
  int line_;
};

#define LOG(level) ::pccl::LogMessage(level, __FILE__, __LINE__)

#define LOG_DEBUG LOG(::pccl::LogLevel::DEBUG)
#define LOG_INFO LOG(::pccl::LogLevel::INFO)
#define LOG_WARNING LOG(::pccl::LogLevel::WARNING)
#define LOG_ERROR LOG(::pccl::LogLevel::ERROR)
#define LOG_FATAL LOG(::pccl::LogLevel::FATAL)

} // namespace pccl