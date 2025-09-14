#pragma once

#include <spdlog/logger.h>

#ifndef PCCL_LOG_LEVEL
#define PCCL_LOG_LEVEL spdlog::level::trace
#endif

namespace pccl {

std::shared_ptr<spdlog::logger> getLogger();

#define PCCL_LOG(level, ...) ::pccl::getLogger()->level(__VA_ARGS__)

#define PCCL_LOG_TRACE(...)   PCCL_LOG(trace, __VA_ARGS__)
#define PCCL_LOG_DEBUG(...)   PCCL_LOG(debug, __VA_ARGS__)
#define PCCL_LOG_INFO(...)    PCCL_LOG(info, __VA_ARGS__)
#define PCCL_LOG_WARN(...)    PCCL_LOG(warn, __VA_ARGS__)
#define PCCL_LOG_ERROR(...)   PCCL_LOG(error, __VA_ARGS__)
#define PCCL_LOG_CRITICAL(...) PCCL_LOG(critical, __VA_ARGS__)

#ifdef PCCL_DEBUG
#define PCCL_DLOG(level, ...) PCCL_LOG(level, __VA_ARGS__)
#else
#define PCCL_DLOG(level, ...) (void)0
#endif

#define PCCL_DLOG_TRACE(...)   PCCL_DLOG(trace, __VA_ARGS__)
#define PCCL_DLOG_DEBUG(...)   PCCL_DLOG(debug, __VA_ARGS__)
#define PCCL_DLOG_INFO(...)    PCCL_DLOG(info, __VA_ARGS__)
#define PCCL_DLOG_WARN(...)    PCCL_DLOG(warn, __VA_ARGS__)
#define PCCL_DLOG_ERROR(...)   PCCL_DLOG(error, __VA_ARGS__)
#define PCCL_DLOG_CRITICAL(...) PCCL_DLOG(critical, __VA_ARGS__)

} // namespace pccl
