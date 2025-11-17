#pragma once

#include <unordered_map>
#include <string>
#include <string_view>

namespace engine_c::utils {

class LaunchEnvironments {
  LaunchEnvironments();
  const std::string_view _getEnv(const std::string &env) const;
public:
  static const LaunchEnvironments& getInstance() {
    static LaunchEnvironments instance;
    return instance;
  }

  static const std::string_view getEnv(const std::string &env) {
    return getInstance()._getEnv(env);
  }

private:
  std::unordered_map<std::string, std::string> env_cache_;
};

}
