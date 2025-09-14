#include "utils/logging.h"
#include "spdlog/logger.h"
#include <memory>

namespace pccl {

std::shared_ptr<spdlog::logger> getLogger() {
  static auto logger = std::make_shared<spdlog::logger>("pccl");
  return logger;
}

} // namespace pccl

