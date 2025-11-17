#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <common/logging.h>
#include <common/environ.h>
#include <common/time.h>
#include <fmt/format.h>

static auto select_spdlog_level(std::string_view level_str) {
  if (level_str == "TRACE") {
    return spdlog::level::trace;
  }
  if (level_str == "DEBUG") {
    return spdlog::level::debug;
  }
  if (level_str == "INFO") {
    return spdlog::level::info;
  }
  if (level_str == "WARN") {
    return spdlog::level::warn;
  }
  if (level_str == "ERROR") {
    return spdlog::level::err;
  }
  if (level_str == "CRITICAL") {
    return spdlog::level::critical;
  }
  return PCCL_DEFAULT_LOG_LEVEL;
}

std::shared_ptr<spdlog::logger> engine_c::utils::getLogger() {
  static auto logger = []() {
    auto pccl_log_level = engine_c::utils::LaunchEnvironments::getEnv("PCCL_LOG_LEVEL");
    auto pccl_log_to_terminal = engine_c::utils::LaunchEnvironments::getEnv("PCCL_LOG_TO_TERMINAL");
    auto pccl_log_filename = engine_c::utils::LaunchEnvironments::getEnv("PCCL_LOG_FILENAME");

    std::string default_log_filename = fmt::format("{}.log", get_launch_time_stamp());
    std::string_view log_file_name = pccl_log_filename == "" ? default_log_filename : pccl_log_filename;

    auto log_level = pccl_log_level == "" ? PCCL_DEFAULT_LOG_LEVEL : select_spdlog_level(pccl_log_level);
    bool log_to_terminal = pccl_log_to_terminal == "true" || pccl_log_to_terminal == "1";

    std::vector<spdlog::sink_ptr> sinks;
    
    auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(fmt::format("/tmp/pccl/log/{}", log_file_name), true);
    file_sink->set_level(log_level);
    sinks.push_back(file_sink);
    
    if (log_to_terminal) {
      auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
      console_sink->set_level(log_level);
      sinks.push_back(console_sink);
    }
    
    auto logger = std::make_shared<spdlog::logger>("pccl", sinks.begin(), sinks.end());
    logger->set_level(log_level);
    logger->flush_on(log_level);
    
    return logger;
  }();
  
  return logger;
}
