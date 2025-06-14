// #include "component/profile.h"

// #include <algorithm>
// #include <atomic>
// #include <chrono>
// #include <cmath>
// #include <filesystem>
// #include <fstream>
// #include <iomanip>
// #include <iostream>
// #include <mutex>
// #include <numeric>
// #include <sstream>
// #include <stack>
// #include <thread>
// #include <unordered_map>
// #include <vector>

// namespace pccl {
// namespace profile {

// #if PCCL_PROFILE_ENABLED

// // 线程局部数据结构，避免锁竞争
// struct ThreadLocalData {
//   std::stack<std::pair<EventId, TimePoint>> activeEvents; // 支持嵌套
//   std::vector<EventRecord> localRecords; // 线程局部事件记录
//   uint32_t currentDepth = 0;
// };

// // Profiler的内部实现
// class ProfilerImpl {
// public:
//   ProfilerImpl() : nextEventId_(1), totalEvents_(0) {
//     config_.enableProfiling = true;
//   }

//   ~ProfilerImpl() {
//     if (config_.enableAutoFlush && !config_.autoFlushPath.empty()) {
//       exportProfile(config_.autoFlushPath, config_.defaultFormat);
//     }
//   }

//   void configure(const ProfilerConfig &config) {
//     std::lock_guard<std::mutex> lock(configMutex_);
//     config_ = config;
//   }

//   const ProfilerConfig &getConfig() const {
//     std::lock_guard<std::mutex> lock(configMutex_);
//     return config_;
//   }

//   EventId registerEvent(std::string_view eventName) {
//     std::lock_guard<std::mutex> lock(eventMapMutex_);

//     // 查找是否已经注册过
//     std::string nameStr(eventName);
//     auto it = eventNameToId_.find(nameStr);
//     if (it != eventNameToId_.end()) {
//       return it->second;
//     }

//     // 注册新事件
//     EventId newId = nextEventId_++;
//     eventNameToId_[nameStr] = newId;
//     eventIdToName_[newId] = nameStr;
//     return newId;
//   }

//   std::string_view getEventName(EventId eventId) const {
//     std::lock_guard<std::mutex> lock(eventMapMutex_);
//     auto it = eventIdToName_.find(eventId);
//     return it != eventIdToName_.end() ? std::string_view(it->second)
//                                       : std::string_view("");
//   }

//   void startProfile(EventId eventId, std::string_view extraInfo) {
//     if (!config_.enableProfiling)
//       return;

//     ThreadLocalData &tld = getThreadLocalData();
//     TimePoint now = std::chrono::high_resolution_clock::now();

//     tld.activeEvents.push({eventId, now});
//     tld.currentDepth++;

//     // 如果需要额外信息，预先创建记录
//     if (!extraInfo.empty() || eventCallback_) {
//       EventRecord record;
//       record.eventId = eventId;
//       record.eventName = getEventName(eventId);
//       record.extraInfo = std::string(extraInfo);
//       record.threadId = std::this_thread::get_id();
//       record.startTime = now;
//       record.depth = tld.currentDepth;

//       // 如果有回调，立即调用
//       if (eventCallback_) {
//         eventCallback_(record);
//       }
//     }
//   }

//   void endProfile(EventId eventId) {
//     if (!config_.enableProfiling)
//       return;

//     ThreadLocalData &tld = getThreadLocalData();
//     if (tld.activeEvents.empty())
//       return;

//     TimePoint endTime = std::chrono::high_resolution_clock::now();

//     // 查找匹配的事件（支持嵌套）
//     std::stack<std::pair<EventId, TimePoint>> tempStack;
//     std::pair<EventId, TimePoint> targetEvent{0, {}};
//     bool found = false;

//     while (!tld.activeEvents.empty()) {
//       auto current = tld.activeEvents.top();
//       tld.activeEvents.pop();

//       if (current.first == eventId) {
//         targetEvent = current;
//         found = true;
//         break;
//       }
//       tempStack.push(current);
//     }

//     // 恢复栈
//     while (!tempStack.empty()) {
//       tld.activeEvents.push(tempStack.top());
//       tempStack.pop();
//     }

//     if (!found)
//       return;

//     // 创建事件记录
//     EventRecord record;
//     record.eventId = eventId;
//     record.eventName = getEventName(eventId);
//     record.threadId = std::this_thread::get_id();
//     record.startTime = targetEvent.second;
//     record.endTime = endTime;
//     record.duration =
//         std::chrono::duration_cast<Duration>(endTime - targetEvent.second);
//     record.depth = tld.currentDepth;

//     tld.currentDepth--;

//     // 添加到本地记录
//     tld.localRecords.push_back(record);
//     totalEvents_++;

//     // 检查是否需要刷新
//     if (config_.enableAutoFlush && totalEvents_ % 10000 == 0) {
//       flushLocalRecords();
//     }

//     // 调用回调
//     if (eventCallback_) {
//       eventCallback_(record);
//     }
//   }

//   bool exportProfile(const std::string &exportPath, ExportFormat format) {
//     // 收集所有数据
//     std::vector<EventRecord> allRecords = collectAllRecords();

//     try {
//       std::filesystem::create_directories(
//           std::filesystem::path(exportPath).parent_path());
//     } catch (const std::exception &) {
//       return false;
//     }

//     switch (format) {
//     case ExportFormat::CSV:
//       return exportCSV(exportPath, allRecords);
//     case ExportFormat::JSON:
//       return exportJSON(exportPath, allRecords);
//     case ExportFormat::CHROME_TRACE:
//       return exportChromeTrace(exportPath, allRecords);
//     case ExportFormat::BINARY:
//       return exportBinary(exportPath, allRecords);
//     default:
//       return false;
//     }
//   }

//   void clearProfileData() {
//     // 清除全局记录
//     {
//       std::lock_guard<std::mutex> lock(recordsMutex_);
//       globalRecords_.clear();
//     }

//     // 清除线程局部数据
//     ThreadLocalData &tld = getThreadLocalData();
//     tld.localRecords.clear();
//     while (!tld.activeEvents.empty()) {
//       tld.activeEvents.pop();
//     }
//     tld.currentDepth = 0;

//     totalEvents_ = 0;
//   }

//   size_t getEventCount() const { return totalEvents_; }

//   std::vector<EventRecord> getEventRecords() const {
//     return collectAllRecords();
//   }

//   Profiler::Statistics getStatistics() const {
//     auto records = collectAllRecords();
//     Profiler::Statistics stats;

//     stats.totalEvents = records.size();
//     stats.activeEvents = 0; // TODO: 计算活动事件数
//     stats.totalDuration = Duration::zero();

//     for (const auto &record : records) {
//       stats.totalDuration += record.duration;
//       stats.eventCounts[record.eventId]++;
//       stats.eventTotalTimes[record.eventId] += record.duration;
//     }

//     return stats;
//   }

//   void setEventCallback(Profiler::EventCallback callback) {
//     eventCallback_ = callback;
//   }

// private:
//   ThreadLocalData &getThreadLocalData() {
//     thread_local ThreadLocalData tld;
//     return tld;
//   }

//   void flushLocalRecords() {
//     ThreadLocalData &tld = getThreadLocalData();
//     if (tld.localRecords.empty())
//       return;

//     std::lock_guard<std::mutex> lock(recordsMutex_);
//     globalRecords_.insert(globalRecords_.end(), tld.localRecords.begin(),
//                           tld.localRecords.end());
//     tld.localRecords.clear();
//   }

//   std::vector<EventRecord> collectAllRecords() const {
//     std::vector<EventRecord> allRecords;

//     // 获取全局记录
//     {
//       std::lock_guard<std::mutex> lock(recordsMutex_);
//       allRecords = globalRecords_;
//     }

//     // 添加当前线程的局部记录
//     const ThreadLocalData &tld = getThreadLocalData();
//     allRecords.insert(allRecords.end(), tld.localRecords.begin(),
//                       tld.localRecords.end());

//     // 按时间排序
//     std::sort(allRecords.begin(), allRecords.end(),
//               [](const EventRecord &a, const EventRecord &b) {
//                 return a.startTime < b.startTime;
//               });

//     return allRecords;
//   }

//   bool exportCSV(const std::string &exportPath,
//                  const std::vector<EventRecord> &records) {
//     std::ofstream file(exportPath);
//     if (!file.is_open())
//       return false;

//     file << std::fixed << std::setprecision(3);
//     file << "EventID,EventName,ThreadID,StartTime_ns,EndTime_ns,Duration_ns,"
//             "Depth,ExtraInfo\n";

//     for (const auto &record : records) {
//       auto startNs = std::chrono::duration_cast<std::chrono::nanoseconds>(
//                          record.startTime.time_since_epoch())
//                          .count();
//       auto endNs = std::chrono::duration_cast<std::chrono::nanoseconds>(
//                        record.endTime.time_since_epoch())
//                        .count();

//       file << record.eventId << "," << "\"" << record.eventName << "\","
//            << record.threadId << "," << startNs << "," << endNs << ","
//            << record.duration.count() << "," << record.depth << "," << "\""
//            << record.extraInfo << "\"\n";
//     }

//     return true;
//   }

//   bool exportJSON(const std::string &exportPath,
//                   const std::vector<EventRecord> &records) {
//     std::ofstream file(exportPath);
//     if (!file.is_open())
//       return false;

//     file << "{\n  \"events\": [\n";

//     for (size_t i = 0; i < records.size(); ++i) {
//       const auto &record = records[i];
//       auto startNs = std::chrono::duration_cast<std::chrono::nanoseconds>(
//                          record.startTime.time_since_epoch())
//                          .count();
//       auto endNs = std::chrono::duration_cast<std::chrono::nanoseconds>(
//                        record.endTime.time_since_epoch())
//                        .count();

//       file << "    {\n"
//            << "      \"eventId\": " << record.eventId << ",\n"
//            << "      \"eventName\": \"" << record.eventName << "\",\n"
//            << "      \"threadId\": \"" << record.threadId << "\",\n"
//            << "      \"startTime\": " << startNs << ",\n"
//            << "      \"endTime\": " << endNs << ",\n"
//            << "      \"duration\": " << record.duration.count() << ",\n"
//            << "      \"depth\": " << record.depth << ",\n"
//            << "      \"extraInfo\": \"" << record.extraInfo << "\"\n"
//            << "    }";

//       if (i < records.size() - 1)
//         file << ",";
//       file << "\n";
//     }

//     file << "  ]\n}\n";
//     return true;
//   }

//   bool exportChromeTrace(const std::string &exportPath,
//                          const std::vector<EventRecord> &records) {
//     std::ofstream file(exportPath);
//     if (!file.is_open())
//       return false;

//     file << "{\n  \"traceEvents\": [\n";

//     for (size_t i = 0; i < records.size(); ++i) {
//       const auto &record = records[i];
//       auto startUs = std::chrono::duration_cast<std::chrono::microseconds>(
//                          record.startTime.time_since_epoch())
//                          .count();
//       auto durationUs =
//           std::chrono::duration_cast<std::chrono::microseconds>(record.duration)
//               .count();

//       file << "    {\n"
//            << "      \"name\": \"" << record.eventName << "\",\n"
//            << "      \"cat\": \"profile\",\n"
//            << "      \"ph\": \"X\",\n"
//            << "      \"ts\": " << startUs << ",\n"
//            << "      \"dur\": " << durationUs << ",\n"
//            << "      \"pid\": 1,\n"
//            << "      \"tid\": \"" << record.threadId << "\",\n"
//            << "      \"args\": {\n"
//            << "        \"eventId\": " << record.eventId << ",\n"
//            << "        \"depth\": " << record.depth << ",\n"
//            << "        \"extraInfo\": \"" << record.extraInfo << "\"\n"
//            << "      }\n"
//            << "    }";

//       if (i < records.size() - 1)
//         file << ",";
//       file << "\n";
//     }

//     file << "  ]\n}\n";
//     return true;
//   }

//   bool exportBinary(const std::string &exportPath,
//                     const std::vector<EventRecord> &records) {
//     std::ofstream file(exportPath, std::ios::binary);
//     if (!file.is_open())
//       return false;

//     // 简单的二进制格式：记录数量 + 记录数据
//     uint32_t count = static_cast<uint32_t>(records.size());
//     file.write(reinterpret_cast<const char *>(&count), sizeof(count));

//     for (const auto &record : records) {
//       file.write(reinterpret_cast<const char *>(&record.eventId),
//                  sizeof(record.eventId));
//       // 注意：这里简化了，实际应该序列化所有字段
//     }

//     return true;
//   }

//   ProfilerConfig config_;
//   mutable std::mutex configMutex_;

//   std::atomic<EventId> nextEventId_;
//   std::unordered_map<std::string, EventId> eventNameToId_;
//   std::unordered_map<EventId, std::string> eventIdToName_;
//   mutable std::mutex eventMapMutex_;

//   std::vector<EventRecord> globalRecords_;
//   mutable std::mutex recordsMutex_;

//   std::atomic<size_t> totalEvents_;
//   Profiler::EventCallback eventCallback_;
// };

// // ProfileScope implementation
// ProfileScope::ProfileScope(EventId eventId, std::string_view eventName,
//                            std::string_view extraInfo)
//     : eventId_(eventId),
//     startTime_(std::chrono::high_resolution_clock::now()),
//       active_(true) {
//   Profiler::getInstance().startProfile(eventId, extraInfo);
// }

// ProfileScope::~ProfileScope() {
//   if (active_) {
//     Profiler::getInstance().endProfile(eventId_);
//   }
// }

// // Profiler implementation
// Profiler::Profiler() : impl_(std::make_unique<ProfilerImpl>()) {}

// Profiler::~Profiler() = default;

// Profiler &Profiler::getInstance() {
//   static Profiler instance;
//   return instance;
// }

// void Profiler::configure(const ProfilerConfig &config) {
//   impl_->configure(config);
// }

// const ProfilerConfig &Profiler::getConfig() const { return
// impl_->getConfig(); }

// EventId Profiler::registerEvent(std::string_view eventName) {
//   return impl_->registerEvent(eventName);
// }

// std::string_view Profiler::getEventName(EventId eventId) const {
//   return impl_->getEventName(eventId);
// }

// void Profiler::startProfile(EventId eventId, std::string_view extraInfo) {
//   impl_->startProfile(eventId, extraInfo);
// }

// void Profiler::endProfile(EventId eventId) { impl_->endProfile(eventId); }

// bool Profiler::exportProfile(const std::string &exportPath,
//                              ExportFormat format) {
//   return impl_->exportProfile(exportPath, format);
// }

// bool Profiler::exportProfile(
//     const std::string &exportPath,
//     const std::function<void(const std::vector<EventRecord> &)>
//         &customExporter) {
//   auto records = impl_->getEventRecords();
//   customExporter(records);
//   return true;
// }

// void Profiler::clearProfileData() { impl_->clearProfileData(); }

// size_t Profiler::getEventCount() const { return impl_->getEventCount(); }

// std::vector<EventRecord> Profiler::getEventRecords() const {
//   return impl_->getEventRecords();
// }

// Profiler::Statistics Profiler::getStatistics() const {
//   return impl_->getStatistics();
// }

// void Profiler::setEventCallback(EventCallback callback) {
//   impl_->setEventCallback(callback);
// }

// #else // PCCL_PROFILE_ENABLED == 0

// // 空实现用于release模式
// ProfileScope::ProfileScope(EventId, std::string_view, std::string_view)
//     : eventId_(0), active_(false) {}
// ProfileScope::~ProfileScope() {}

// Profiler::Profiler() {}
// Profiler::~Profiler() {}

// Profiler &Profiler::getInstance() {
//   static Profiler instance;
//   return instance;
// }

// void Profiler::configure(const ProfilerConfig &) {}
// const ProfilerConfig &Profiler::getConfig() const {
//   static ProfilerConfig config;
//   return config;
// }

// EventId Profiler::registerEvent(std::string_view) { return 0; }
// std::string_view Profiler::getEventName(EventId) const { return ""; }
// void Profiler::startProfile(EventId, std::string_view) {}
// void Profiler::endProfile(EventId) {}
// bool Profiler::exportProfile(const std::string &, ExportFormat) {
//   return false;
// }
// bool Profiler::exportProfile(
//     const std::string &,
//     const std::function<void(const std::vector<EventRecord> &)> &) {
//   return false;
// }
// void Profiler::clearProfileData() {}
// size_t Profiler::getEventCount() const { return 0; }
// std::vector<EventRecord> Profiler::getEventRecords() const { return {}; }
// Profiler::Statistics Profiler::getStatistics() const { return {}; }
// void Profiler::setEventCallback(EventCallback) {}

// #endif // PCCL_PROFILE_ENABLED

// } // namespace profile
// } // namespace pccl