#include "config.h"
#include "utils.h"
#include <cstdint>
#include <format>
#include <fstream>
#include <iostream>
#include <ostream>
#include <unistd.h>

using namespace std;
using namespace pccl;

PCCL_API pccl::Profiler::Profiler() { m_pid = static_cast<uint64_t>(getpid()); }

Profiler::~Profiler() {
  exportToChromeTracing(std::format("pccl.{}.json", get_start_timestamp()));
}

PCCL_API void pccl::Profiler::registerEvent(uint64_t uid, const string &name) {
  lock_guard<mutex> lock(m_mutex);
  m_eventNames[uid] = name;
}

PCCL_API void pccl::Profiler::recordStart(uint64_t uid) {
  lock_guard<mutex> lock(m_mutex);
  m_events.push_back({'B', uid, getTimestamp(), this_thread::get_id()});
}

PCCL_API void pccl::Profiler::recordEnd(uint64_t uid) {
  lock_guard<mutex> lock(m_mutex);
  m_events.push_back({'E', uid, getTimestamp(), this_thread::get_id()});
}

PCCL_API void pccl::Profiler::exportToChromeTracing(const string &filePath) {
  lock_guard<mutex> lock(m_mutex);
  create_dir(Config::PROFILE_PATH);
  ofstream out(std::format("{}/{}", Config::PROFILE_PATH, filePath));
  if (!out.is_open()) {
    throw runtime_error("Profiler failed to open file " + filePath);
  }
  out << "[\n";
  bool first = true;
  for (const auto &event : m_events) {
    auto it = m_eventNames.find(event.uid);
    if (it == m_eventNames.end())
      continue;

    if (!first)
      out << ",\n";
    first = false;
    out << "{";
    out << "\"name\": \"" << it->second << "\",";
    out << "\"cat\": \"PERF\",";
    out << "\"ph\": \"" << event.type << "\",";
    out << "\"ts\": " << event.timestamp << ",";
    out << "\"pid\": " << m_pid << ",";
    out << "\"tid\": " << hash<thread::id>{}(event.tid);
    out << "}";
  }

  out << "\n]";
  out.close();
}

PCCL_API pccl::ProfileScope::ProfileScope(uint64_t uid) : m_uid(uid) {
  Profiler::getInstance().recordStart(m_uid);
}

PCCL_API pccl::ProfileScope::~ProfileScope() {
  Profiler::getInstance().recordEnd(m_uid);
}