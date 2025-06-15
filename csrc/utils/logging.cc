#include "utils/logging.h"
#include "config.h"
#include "utils/defs.h"
#include <format>
#include <stdexcept>

using namespace pccl;
using namespace std;

PCCL_API void pccl::LoggerWrapper::init() {
  std::call_once(init_flag_, []() {
    console_sink_ = std::make_shared<spdlog::sinks::stderr_sink_mt>();
    logger_ = std::make_shared<spdlog::logger>("pccl_logger", console_sink_);
    spdlog::register_logger(logger_);
    logger_->set_pattern("%Y-%m-%d %H:%M:%S.%e [%l] [%g:%#] %v");
    logger_->set_level(spdlog::level::trace);
    level_map_ = {{LogLevel::DEBUG, spdlog::level::debug},
                  {LogLevel::INFO, spdlog::level::info},
                  {LogLevel::WARNING, spdlog::level::warn},
                  {LogLevel::ERROR, spdlog::level::err},
                  {LogLevel::FATAL, spdlog::level::critical}};
    if (Config::LOG_LEVEL == "DEBUG")
      logger_->set_level(spdlog::level::debug);
    else if (Config::LOG_LEVEL == "INFO")
      logger_->set_level(spdlog::level::info);
    else if (Config::LOG_LEVEL == "WARNING")
      logger_->set_level(spdlog::level::warn);
    else if (Config::LOG_LEVEL == "ERROR")
      logger_->set_level(spdlog::level::err);
    else if (Config::LOG_LEVEL == "FATAL")
      logger_->set_level(spdlog::level::critical);
    else
      throw runtime_error(
          std::format("Invalid log level: {}", Config::LOG_LEVEL));

    if (!Config::LOG_PATH.empty()) {
      create_dir(Config::LOG_PATH);
      file_sink_ = std::make_shared<spdlog::sinks::basic_file_sink_mt>(
          std::format("{}/pccl.{}.log", Config::LOG_PATH,
                      get_start_timestamp()),
          true);
      logger_->sinks() = {file_sink_};
    } else {
      console_sink_ = std::make_shared<spdlog::sinks::stderr_sink_mt>();
      logger_->sinks() = {console_sink_};
    }
  });
}

PCCL_API void pccl::LoggerWrapper::flushLog() {
  init();
  logger_->flush();
}

PCCL_API void pccl::LoggerWrapper::log(LogLevel level, const char *file,
                                       int line, const std::string &message) {
  init();
  auto it = level_map_.find(level);
  if (it == level_map_.end())
    return;

  const char *filename = file;
  if (const char *last_slash = strrchr(file, '/')) {
    filename = last_slash + 1;
  } else if (const char *last_backslash = strrchr(file, '\\')) {
    filename = last_backslash + 1;
  }

  logger_->log(spdlog::source_loc{filename, line, SPDLOG_FUNCTION}, it->second,
               message);

  if (level == LogLevel::FATAL) {
    logger_->flush();
    std::abort();
  }
}

std::shared_ptr<spdlog::logger> pccl::LoggerWrapper::logger_ = nullptr;
std::shared_ptr<spdlog::sinks::sink> pccl::LoggerWrapper::console_sink_ =
    nullptr;
std::shared_ptr<spdlog::sinks::sink> pccl::LoggerWrapper::file_sink_ = nullptr;
std::once_flag pccl::LoggerWrapper::init_flag_{};
std::unordered_map<LogLevel, spdlog::level::level_enum>
    pccl::LoggerWrapper::level_map_{};