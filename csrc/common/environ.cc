#include <common/environ.h>
#include <cstdlib>
#include <unistd.h>
#include <stdlib.h>

namespace engine_c::utils {

LaunchEnvironments::LaunchEnvironments() {
  extern char** environ;
  for (char** env = environ; *env != nullptr; ++env) {
    std::string env_str(*env);
    size_t pos = env_str.find('=');
    if (pos != std::string::npos) {
      std::string key = env_str.substr(0, pos);
      std::string value = env_str.substr(pos + 1);
      env_cache_[std::move(key)] = std::move(value);
    }
  }
}

const std::string_view LaunchEnvironments::_getEnv(const std::string &env) const {
  auto it = env_cache_.find(env);
  if (it != env_cache_.end()) {
    return it->second;
  }
  
  static const std::string empty_string;
  return empty_string;
}

}
