#pragma once

#include <string>
#include "configs/verbs_config.h"
#include "utils/logging.h"

namespace pccl {

static std::string getEnv(const std::string &env_name) {
  const char *env = std::getenv(env_name.c_str());
  if (env == nullptr) {
    return "";
  }
  PCCL_DLOG_INFO(std::format("roce config environment: {}:{}", env_name, env));
  return env;
}

static constexpr std::string default_lib_path = "libibverbs.so";
static constexpr std::string default_device_name = "";
static constexpr int default_port_num = 1;
static constexpr int default_gid_index = 1;
static constexpr int default_qkey = 0x76543210;


VerbsConfig::VerbsConfig() {
  std::string pccl_env_ib_lib_path = getEnv("PCCL_IB_LIB");
  std::string pccl_env_ib_device_name = getEnv("PCCL_IB_DEVICE");
  std::string pccl_env_ib_port_num = getEnv("PCCL_IB_PORT_NUM");
  std::string pccl_env_ib_gid_index = getEnv("PCCL_IB_GID_INDEX");
  std::string pccl_env_ib_qkey = getEnv("PCCL_IB_QKEY");
  if (pccl_env_ib_lib_path == "") {
    lib_path = default_lib_path;
  } else {
    lib_path = pccl_env_ib_lib_path;
  }

  if (pccl_env_ib_device_name == "") {
    device_name = default_device_name;
  } else {
    device_name = pccl_env_ib_device_name;
  }

  if (pccl_env_ib_port_num == "") {
    port_num = default_port_num; 
  } else {
    port_num = std::stoi(pccl_env_ib_port_num);
  }
  
  if (pccl_env_ib_gid_index == "") {
    gid_index = default_gid_index;
  } else {
    gid_index = std::stoi(pccl_env_ib_gid_index);
  }
  
  if (pccl_env_ib_qkey == "") {
    qkey = default_qkey;
  } else {
    qkey = std::stoi(pccl_env_ib_qkey);
  }
}

} // namespace pccl
