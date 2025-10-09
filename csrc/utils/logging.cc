#include "utils/logging.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/sinks/basic_file_sink.h"
#include <memory>

namespace pccl {

std::shared_ptr<spdlog::logger> getLogger() {
  static auto logger = []() {
    // auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    // console_sink->set_level(PCCL_LOG_LEVEL);
    
    auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>("pccl.log", true);
    file_sink->set_level(PCCL_LOG_LEVEL);
    
    std::vector<spdlog::sink_ptr> sinks{file_sink};
    auto logger = std::make_shared<spdlog::logger>("pccl", sinks.begin(), sinks.end());
    logger->set_level(PCCL_LOG_LEVEL);
    logger->flush_on(PCCL_LOG_LEVEL);
    
    return logger;
  }();
  
  return logger;
}

} // namespace pccl
