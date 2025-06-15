#pragma once

#include <chrono>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace pccl {

class Profiler {
public:
  static Profiler &getInstance() {
    static Profiler instance;
    return instance;
  }

  void registerEvent(uint64_t uid, const std::string &name);
  void recordStart(uint64_t uid);
  void recordEnd(uint64_t uid);
  void exportToChromeTracing(const std::string &filePath);
  Profiler(const Profiler &) = delete;
  Profiler &operator=(const Profiler &) = delete;

private:
  Profiler();
  ~Profiler();
  struct EventRecord {
    char type;
    uint64_t uid;
    uint64_t timestamp; // in nanoseconds
    std::thread::id tid;
  };

  std::mutex m_mutex;
  std::vector<EventRecord> m_events;
  std::unordered_map<uint64_t, std::string> m_eventNames;
  unsigned long m_pid;

  static uint64_t getTimestamp() {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::time_point_cast<std::chrono::nanoseconds>(now)
        .time_since_epoch()
        .count();
  }
};

class ProfileScope {
public:
  ProfileScope(uint64_t uid);
  ~ProfileScope();

private:
  uint64_t m_uid;
};

#define PROFILE_REGISTER_EVENT(name, uid)                                      \
  Profiler::getInstance().registerEvent(uid, name)

#define PROFILE_REGISTER_START(uid) Profiler::getInstance().recordStart(uid)

#define PROFILE_REGISTER_END(uid) Profiler::getInstance().recordEnd(uid)

#define PROFILE_SCOPE(name)                                                    \
  static uint64_t ANON_UID_##__LINE__ = [] {                                   \
    static std::atomic<uint64_t> counter{0};                                   \
    return counter.fetch_add(1, std::memory_order_relaxed);                    \
  }();                                                                         \
  PROFILE_REGISTER_EVENT(name, ANON_UID_##__LINE__);                           \
  ProfileScope ANON_PROFILE_##__LINE__(ANON_UID_##__LINE__)

} // namespace pccl