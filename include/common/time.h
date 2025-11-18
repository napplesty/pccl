#pragma once

#include <string_view>
#include <cstdint>

namespace engine_c::utils {

std::string_view get_launch_time_stamp();
uint64_t getCurrentTimeNanos();
std::string get_current();

}
