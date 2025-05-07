#include "component/profile.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

namespace pccl {
namespace profile {

using TimePoint = std::chrono::high_resolution_clock::time_point;
using Duration = std::chrono::duration<double, std::micro>;

struct EventRecord {
  std::string eventName;
  std::string extraInfo;
  std::thread::id threadId;
  TimePoint startTime;
  TimePoint endTime;
  Duration duration;
};

static std::vector<EventRecord> g_eventRecords;
static std::mutex g_recordsMutex;
static thread_local std::map<std::string, TimePoint> tl_activeEvents;

// --- Implementation ---

void internalStartProfile(const char* eventName, const char* extraInfo) {
  tl_activeEvents[eventName] = std::chrono::high_resolution_clock::now();
}

void internalEndProfile(const char* eventName) {
  TimePoint endTime = std::chrono::high_resolution_clock::now();
  std::string eventNameStr(eventName);
  auto it = tl_activeEvents.find(eventNameStr);
  if (it == tl_activeEvents.end()) {
    return;
  }

  TimePoint startTime = it->second;
  tl_activeEvents.erase(it);

  // Create the record
  EventRecord record;
  record.eventName = eventNameStr;
  record.threadId = std::this_thread::get_id();
  record.startTime = startTime;
  record.endTime = endTime;
  record.duration = endTime - startTime;

  // Add the completed record to the global storage (thread-safe)
  {
    std::lock_guard<std::mutex> lock(g_recordsMutex);
    g_eventRecords.push_back(std::move(record));
  }
}

void clearProfileData() {
  std::lock_guard<std::mutex> lock(g_recordsMutex);
  g_eventRecords.clear();
  tl_activeEvents.clear();
}

void exportProfile(const std::string& exportDir) {
  std::vector<EventRecord> recordsCopy;
  {
    std::lock_guard<std::mutex> lock(g_recordsMutex);
    if (g_eventRecords.empty()) {
      return;
    }
    recordsCopy = g_eventRecords;  // Make a copy to process outside the lock
  }  // Mutex released here

  // 2. Create export directory
  try {
    std::filesystem::create_directories(exportDir);
  } catch (const std::exception& e) {
    return;
  }

  // 3. Write Raw Data
  std::string rawFilePath = exportDir + "/profile_raw.csv";
  std::ofstream rawFile(rawFilePath);
  if (!rawFile.is_open()) {
    return;
  }

  rawFile << std::fixed << std::setprecision(3);  // Precision for microseconds
  rawFile
      << "EventName,ThreadID,StartTime_us,EndTime_us,Duration_us,ExtraInfo\n";
  for (const auto& record : recordsCopy) {
    auto startUs = std::chrono::duration_cast<std::chrono::microseconds>(
                       record.startTime.time_since_epoch())
                       .count();
    auto endUs = std::chrono::duration_cast<std::chrono::microseconds>(
                     record.endTime.time_since_epoch())
                     .count();
    rawFile << record.eventName << "," << record.threadId << "," << startUs
            << ","  // Represent time points relative to epoch in microseconds
            << endUs << "," << record.duration.count() << "," << "\""
            << record.extraInfo
            << "\""  // Quote extra info in case it contains commas
            << "\n";
  }
  rawFile.close();

  std::map<std::string, std::vector<double>> durationsByName;
  for (const auto& record : recordsCopy) {
    durationsByName[record.eventName].push_back(record.duration.count());
  }

  std::string summaryFilePath = exportDir + "/profile_summary.csv";
  std::ofstream summaryFile(summaryFilePath);
  if (!summaryFile.is_open()) {
    return;
  }

  summaryFile << std::fixed
              << std::setprecision(3);  // Precision for microseconds
  summaryFile << "EventName,Count,Avg_us,P95_us,P99_us,Max_us\n";

  for (auto& pair : durationsByName) {
    const std::string& eventName = pair.first;
    std::vector<double>& durations = pair.second;
    size_t count = durations.size();

    if (count == 0) continue;

    std::sort(durations.begin(), durations.end());

    double sum = std::accumulate(durations.begin(), durations.end(), 0.0);
    double avg = sum / count;
    double maxVal = durations.back();

    // Percentiles
    double p95 = durations[static_cast<size_t>(std::ceil(0.95 * count)) - 1];
    double p99 = durations[static_cast<size_t>(std::ceil(0.99 * count)) - 1];

    summaryFile << eventName << "," << count << "," << avg << "," << p95 << ","
                << p99 << "," << maxVal << "\n";
  }
  summaryFile.close();
}

}  // namespace profile
}  // namespace pccl