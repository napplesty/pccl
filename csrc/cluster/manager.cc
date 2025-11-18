#include <cluster/manager.h>
#include <string>
#include <nlohmann/json.hpp>
#include <common.h>

namespace engine_c {

std::string NodeMeta::serialize() {
  std::map<std::string, std::string> buffer_infos;
  for (auto &kv : buffer_infos_) {
    buffer_infos[utils::serialize(reinterpret_cast<const void *>(&kv.first), sizeof(kv.first))] \
        = utils::serialize(reinterpret_cast<void *>(&kv.second), sizeof(kv.second));
  }

  nlohmann::json j;
  j["buffer_infos"] = buffer_infos;
  j["endpoint_configs"] = endpoint_configs_;

  return j.dump(-1,' ', true);
}


}

