#include "base/chunk.h"
#include "base/device.h"
#include "base/operator.h"
#include "base/registry.h"
#include "cluster/manager.h"
#include <common/environ.h>
#include <common/serialize.h>
#include <fmt/format.h>
#include <engine.h>
#include <string>
#include <fstream>

namespace engine_c {

static constexpr long WORKSPACE_SIZE = 1l << 28;
static constexpr long GRAN = 1l << 20;
static constexpr long SLOT = WORKSPACE_SIZE / GRAN;
static constexpr long TOTAL_SIZE = WORKSPACE_SIZE + SLOT * sizeof(int);

static std::string readFileToString(const std::string& filePath) {
  std::ifstream file(filePath, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file: " + filePath + " (" + std::system_category().message(errno) + ")");
  }
  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);
  if (size == 0) {
    return "";
  }
  std::string content;
  content.resize(size);
  if (!file.read(&content[0], size)) {
    throw std::runtime_error("Failed to read file content: " + filePath);
  }
  return content;
}

Engine::Engine(int rank, int world_size)
: rank_(rank), world_size_(world_size)
{}

void Engine::initEngine() {
  buffers_ = std::make_unique<BufferManager>();
  available_remote_devices_.clear();
  available_memory_devices_.clear();

  std::map<std::string, std::string> configs;

  std::string host_id = readFileToString("/proc/sys/kernel/random/boot_id");
  configs["host_id"] = host_id;

  auto &registered_device_names = TypeRegistry::getDeviceTypes();
  for (auto &device : registered_device_names) {
    auto device_name = TypeRegistry::getTypeName(device);
    
    if (getDev(device)->remoteCommAvailable()) {
      // Todo
      std::string handle = getDev(device)->activate();

      configs[fmt::format("remote.{}.available", device_name)] = "true";
      configs[fmt::format("remote.{}.handle", device_name)] = handle;

      available_remote_devices_.push_back(std::string(device_name));
    }
    if (getDev(device)->allocatorAvailable() and \
        getDev(device)->IPCAvailable()) {
      void *ipc_buffer, *signal_buffer;
      auto handle = getDev(device)->allocateIpcBuffer(&ipc_buffer, TOTAL_SIZE);
      signal_buffer = (char *)ipc_buffer + WORKSPACE_SIZE;
      buffers_->regBuffer(ipc_buffer, WORKSPACE_SIZE, 
        (int *)signal_buffer, SLOT, rank_, TypeRegistry::getTypeId(device_name));
      configs[fmt::format("ipc.{}.handle", device_name)] = handle;
      available_memory_devices_.push_back(std::string(device_name));
    } else if (getDev(device)->allocatorAvailable()){
      void *buffer, *signal_buffer;
      buffer = getDev(device)->allocate(TOTAL_SIZE);
      signal_buffer = (char *)buffer + WORKSPACE_SIZE;
      buffers_->regBuffer(buffer, WORKSPACE_SIZE,
        (int *)signal_buffer, SLOT, rank_, TypeRegistry::getTypeId(device_name));
      available_memory_devices_.push_back(std::string(device_name));
    }
  }

  for (auto &comm_dev : available_remote_devices_) {
    for (auto &mem_dev : available_memory_devices_) {
      auto mem_device = TypeRegistry::getTypeId(mem_dev);
      auto comm_device = TypeRegistry::getTypeId(comm_dev);
      auto buffer_signal = buffers_->getDevBuffer(mem_device);
      auto wksp_handle = getDev(comm_device)->regBuffer(std::get<0>(buffer_signal), WORKSPACE_SIZE);
      auto signal_handle = getDev(comm_device)->regBuffer(std::get<1>(buffer_signal), sizeof(int) * SLOT);

      configs[fmt::format("remote.{}.buffer.{}.wksp_handle", comm_dev, mem_dev)] = wksp_handle;
      configs[fmt::format("remote.{}.buffer.{}.signal_handle", comm_dev, mem_dev)] = signal_handle;
    }
  }


  cluster_ = std::make_unique<ClusterManager>(configs);
  operators_ = std::make_unique<OperatorManager>();
}

void Engine::regOp(const std::string &name, const std::string &filepath) {
  
}

}

