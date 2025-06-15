#include "config.h"
#include "utils.h"

using namespace pccl;

PCCL_API unsigned long pccl::Config::DEVICE_BUFFER_SIZE = 256 * 1024 * 1024;
PCCL_API unsigned long pccl::Config::HOST_BUFFER_SIZE = 256 * 1024 * 1024;
PCCL_API unsigned long pccl::Config::SLOT_GRANULARITY = 8 * 1024 * 1024;
PCCL_API int pccl::Config::PROXY_FLUSH_PERIOD = 4;
PCCL_API int pccl::Config::PROXY_MAX_FLUSH_SIZE = 128;
PCCL_API int pccl::Config::PROXY_CHECK_STOP_PERIOD = 4000;

PCCL_API std::string pccl::Config::LOG_LEVEL = "INFO";
PCCL_API std::string pccl::Config::LOG_PATH = "pccl_log";
PCCL_API std::string pccl::Config::PROFILE_PATH = "pccl_profile";