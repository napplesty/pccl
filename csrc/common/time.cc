#include <common/time.h>
#include <string>
#include <chrono>
#include <fmt/format.h>

namespace engine_c::utils {

static std::string _launch_time_stamp;

namespace {
std::string format_time_with_microseconds(const std::chrono::system_clock::time_point& time_point) {
  auto time_t = std::chrono::system_clock::to_time_t(time_point);
  auto microseconds = std::chrono::duration_cast<std::chrono::microseconds>(
      time_point.time_since_epoch()) % 1000000;
  
  auto tm = *std::localtime(&time_t);
  return fmt::format("{:02d}-{:02d}-{:02d}.{:06d}", 
                      tm.tm_hour, tm.tm_min, tm.tm_sec, microseconds.count());
}
}

class TimeInitializer {
public:
  TimeInitializer() {
    _launch_time_stamp = format_time_with_microseconds(std::chrono::system_clock::now());
  }
};

static TimeInitializer _time_initializer;

std::string_view get_launch_time_stamp() {
  return _launch_time_stamp;
}

std::string get_current() {
  return format_time_with_microseconds(std::chrono::system_clock::now());
}

}
