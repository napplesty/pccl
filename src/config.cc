#include "config.h"

#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>

namespace pccl {

static std::string get_env_var_str(const char *name,
                                   const std::string &default_value = "") {
  const char *value = std::getenv(name);
  return value ? std::string(value) : default_value;
}

static int get_env_var_int(const char *name, int default_value = 0) {
  const char *value_str = std::getenv(name);
  if (value_str) {
    try {
      return std::stoi(value_str);
    } catch (const std::invalid_argument & /*e*/) {
      throw std::invalid_argument("Invalid argument for " + std::string(name) +
                                  ": " + value_str);
    } catch (const std::out_of_range & /*e*/) {
      throw std::invalid_argument("Invalid argument for " + std::string(name) +
                                  ": out of range");
    }
  }
  return default_value;
}

std::shared_ptr<env> getEnv() {
  int rank_val = get_env_var_int("PCCL_RANK", 0);
  int local_rank_val = get_env_var_int("PCCL_LOCAL_RANK", 0);
  int world_size_val = get_env_var_int(
      "PCCL_WORLD_SIZE", 1); // A world size of 1 is a common default.

  std::string socket_family_val = get_env_var_str(
      "PCCL_SOCKET_FAMILY", "AF_INET"); // AF_INET is a common default.
  std::string socket_addr_val = get_env_var_str("PCCL_SOCKET_ADDR");
  std::string socket_port_val = get_env_var_str("PCCL_SOCKET_PORT");
  std::string ib_device0_val = get_env_var_str("PCCL_IB_DEVICE0");
  std::string ib_device1_val = get_env_var_str("PCCL_IB_DEVICE1");
  std::string ib_port0_val = get_env_var_str("PCCL_IB_PORT0");
  std::string ib_port1_val = get_env_var_str("PCCL_IB_PORT1");
  std::string net_conf_file_val = get_env_var_str("PCCL_NET_CONF_FILE");
  std::string net_conf_addr_val = get_env_var_str("PCCL_NET_CONF_ADDR");
  std::string net_conf_port_val = get_env_var_str("PCCL_NET_CONF_PORT");
  std::string net_conf_model_val = get_env_var_str("PCCL_NET_CONF_MODEL");
  std::string profile_dir_val = get_env_var_str("PCCL_PROFILE_DIR");
  std::string enable_transport_list_val =
      get_env_var_str("PCCL_ENABLE_TRANSPORT_LIST");

  return std::make_shared<env>(
      env{rank_val, local_rank_val, world_size_val, socket_family_val,
          socket_addr_val, socket_port_val, ib_device0_val, ib_device1_val,
          ib_port0_val, ib_port1_val, net_conf_addr_val, net_conf_port_val,
          net_conf_model_val, profile_dir_val});
}

} // namespace pccl