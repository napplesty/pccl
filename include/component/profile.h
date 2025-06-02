#pragma once

#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>

#ifndef PCCL_PROFILE_ENABLED
#ifdef NDEBUG
#define PCCL_PROFILE_ENABLED 0
#else
#define PCCL_PROFILE_ENABLED 1
#endif
#endif

namespace pccl {
namespace profile {

using TimePoint = std::chrono::high_resolution_clock::time_point;
using Duration = std::chrono::nanoseconds;

using EventId = uint64_t;

enum class ExportFormat { CSV, JSON, CHROME_TRACE, BINARY };

struct ProfilerConfig {
  bool enableProfiling = true;
  size_t maxEvents = 3000000;
  bool enableAutoFlush = false;
  std::string autoFlushPath;
  ExportFormat defaultFormat = ExportFormat::CSV;
};

struct EventRecord {
  EventId eventId;
  std::string_view eventName;
  std::string extraInfo;
  std::thread::id threadId;
  TimePoint startTime;
  TimePoint endTime;
  Duration duration;
  uint32_t depth;
};

class ProfilerImpl;

class ProfileScope {
public:
  ProfileScope(EventId eventId, std::string_view eventName,
               std::string_view extraInfo = "");
  ~ProfileScope();

  ProfileScope(const ProfileScope &) = delete;
  ProfileScope &operator=(const ProfileScope &) = delete;
  ProfileScope(ProfileScope &&) = delete;
  ProfileScope &operator=(ProfileScope &&) = delete;

private:
  EventId eventId_;
  TimePoint startTime_;
  bool active_;
};

class Profiler {
public:
  static Profiler &getInstance();

  void configure(const ProfilerConfig &config);
  const ProfilerConfig &getConfig() const;

  EventId registerEvent(std::string_view eventName);
  std::string_view getEventName(EventId eventId) const;

  void startProfile(EventId eventId, std::string_view extraInfo = "");
  void endProfile(EventId eventId);

  bool exportProfile(const std::string &exportPath,
                     ExportFormat format = ExportFormat::CSV);
  bool exportProfile(const std::string &exportPath,
                     const std::function<void(const std::vector<EventRecord> &)>
                         &customExporter);

  void clearProfileData();
  size_t getEventCount() const;
  std::vector<EventRecord> getEventRecords() const;

  struct Statistics {
    size_t totalEvents;
    size_t activeEvents;
    Duration totalDuration;
    std::unordered_map<EventId, size_t> eventCounts;
    std::unordered_map<EventId, Duration> eventTotalTimes;
  };
  Statistics getStatistics() const;

  using EventCallback = std::function<void(const EventRecord &)>;
  void setEventCallback(EventCallback callback);

private:
  Profiler();
  ~Profiler();

  std::unique_ptr<ProfilerImpl> impl_;
};

} // namespace profile
} // namespace pccl

#if PCCL_PROFILE_ENABLED

#define PCCL_EVENT_ID(name)                                                    \
  []() -> ::pccl::profile::EventId {                                           \
    static ::pccl::profile::EventId id =                                       \
        ::pccl::profile::Profiler::getInstance().registerEvent(name);          \
    return id;                                                                 \
  }()

#define PCCL_PROFILE_SCOPE(name)                                               \
  ::pccl::profile::ProfileScope _pccl_profile_scope(PCCL_EVENT_ID(name), name)

#define PCCL_PROFILE_SCOPE_WITH_INFO(name, info)                               \
  ::pccl::profile::ProfileScope _pccl_profile_scope(PCCL_EVENT_ID(name), name, \
                                                    info)

#define PCCL_PROFILE_FUNCTION() PCCL_PROFILE_SCOPE(__PRETTY_FUNCTION__)

#define PCCL_PROFILE_START(name)                                               \
  ::pccl::profile::Profiler::getInstance().startProfile(PCCL_EVENT_ID(name))

#define PCCL_PROFILE_START_WITH_INFO(name, info)                               \
  ::pccl::profile::Profiler::getInstance().startProfile(PCCL_EVENT_ID(name),   \
                                                        info)

#define PCCL_PROFILE_END(name)                                                 \
  ::pccl::profile::Profiler::getInstance().endProfile(PCCL_EVENT_ID(name))

#define PCCL_PROFILE_EXPORT(path)                                              \
  ::pccl::profile::Profiler::getInstance().exportProfile(path)

#define PCCL_PROFILE_EXPORT_FORMAT(path, format)                               \
  ::pccl::profile::Profiler::getInstance().exportProfile(path, format)

#define PCCL_PROFILE_CLEAR()                                                   \
  ::pccl::profile::Profiler::getInstance().clearProfileData()

#define PCCL_PROFILE_CONFIGURE(config)                                         \
  ::pccl::profile::Profiler::getInstance().configure(config)

#else

#define PCCL_EVENT_ID(name) 0
#define PCCL_PROFILE_SCOPE(name)                                               \
  do {                                                                         \
  } while (0)
#define PCCL_PROFILE_SCOPE_WITH_INFO(name, info)                               \
  do {                                                                         \
  } while (0)
#define PCCL_PROFILE_FUNCTION()                                                \
  do {                                                                         \
  } while (0)
#define PCCL_PROFILE_START(name)                                               \
  do {                                                                         \
  } while (0)
#define PCCL_PROFILE_START_WITH_INFO(name, info)                               \
  do {                                                                         \
  } while (0)
#define PCCL_PROFILE_END(name)                                                 \
  do {                                                                         \
  } while (0)
#define PCCL_PROFILE_EXPORT(path) false
#define PCCL_PROFILE_EXPORT_FORMAT(path, format) false
#define PCCL_PROFILE_CLEAR()                                                   \
  do {                                                                         \
  } while (0)
#define PCCL_PROFILE_CONFIGURE(config)                                         \
  do {                                                                         \
  } while (0)

#endif
