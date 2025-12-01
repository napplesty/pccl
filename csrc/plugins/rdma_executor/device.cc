#include <plugins/rdma_executor/device.h>
#include <plugins/rdma_executor/utils/rdma_utils.h>
#include <common/environ.h>
#include <common/serialize.h>
#include <cstring>
#include <unordered_map>
#include <mutex>
#include <memory>

namespace engine_c {

struct TMP_STATIC_INITIALIZER {
  TMP_STATIC_INITIALIZER() {
    utils::LaunchEnvironments::registerOpt("PCCL_DISABLE_IB");
    utils::LaunchEnvironments::registerOpt("PCCL_IB_DEVICE");
    utils::LaunchEnvironments::registerOpt("PCCL_IB_GID_INDEX");
    utils::LaunchEnvironments::registerOpt("PCCL_IB_PORT_NUM");
    auto device = TypeRegistry::registerDeviceType("cpu");
    regDev(device, std::make_shared<DeviceBase>(new RdmaDevice));
  }
};

static TMP_STATIC_INITIALIZER _____tmp;

struct MemInfo {
  int rank_;

};


static std::unordered_map<void*, MemInfo> mem_map;
static std::mutex mem_mutex;
static std::unique_ptr<rdma::VerbsManager> ctx;
static std::string device_name;
static int port_num, gid_index;
static ibv_gid self_gid;

bool RdmaDevice::remoteCommAvailable() {
  if (utils::LaunchEnvironments::getEnv("PCCL_DISABLE_IB") != "") {
    return false;
  }

  device_name = \
    std::string(utils::LaunchEnvironments::getEnv("PCCL_IB_DEVICE"));

  port_num = \
    std::stoi(std::string(utils::LaunchEnvironments::getEnv("PCCL_IB_PORT_NUM")));

  gid_index = \
    std::stoi(std::string(utils::LaunchEnvironments::getEnv("PCCL_IB_GID_INDEX")));

  ctx = std::make_unique<rdma::VerbsManager>();

  return ctx->initialize(device_name, port_num);
}

std::string RdmaDevice::activate() {
  auto context = ctx->getContext();
  rdma::VerbsLib::getInstance().queryGid(context->get(), port_num, gid_index, &self_gid);
  return utils::serialize(&self_gid, sizeof(self_gid));
}

std::string RdmaDevice::registerBuffer(void *addr, long size) {
  
}

std::string RdmaDevice::registerLocal() {
  
}

void RdmaDevice::connect(std::string handle) {
  
}

void RdmaDevice::disconnect(std::string handle) {

}

}
