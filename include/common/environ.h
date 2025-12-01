#pragma once

#include <mutex>
#include <vector>
#include <unordered_map>
#include <string>
#include <string_view>

namespace engine_c::utils {

class LaunchEnvironments {
  LaunchEnvironments();
  const std::string_view _getEnv(const std::string &env) const;
public:
  static LaunchEnvironments& getInstance() {
    static LaunchEnvironments instance;
    return instance;
  }

  static const std::string_view getEnv(const std::string &env) {
    return getInstance()._getEnv(env);
  }

  static const std::vector<std::string> &listOpt() {
    return getInstance().opts_;
  }

  static void registerOpt(std::string option) {
    std::lock_guard<std::mutex> lock(getInstance().initializer_mutex_);
    getInstance().opts_.push_back(std::move(option));
  }

private:
  std::unordered_map<std::string, std::string> env_cache_;
  std::vector<std::string> opts_;
  std::mutex initializer_mutex_;
};

}


