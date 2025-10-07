#include "runtime/api/runtime.h"
#include "runtime/communicator/oob_comm.h"
#include "runtime/engine/graph_executor.h"
#include "runtime/engine/memory_manager.h"
#include "runtime/communicator/channel.h"
#include "plugins/atcp/tcp_adapter.h"
#include "plugins/aroce/roce_adapter.h"
#include "plugins/aroce/roce_utils.h"
#include "utils/logging.h"
#include <cstring>
#include <format>
#include <infiniband/verbs.h>
#include <thread>
#include <chrono>
#include <ifaddrs.h>
#include <netinet/in.h>
#include <arpa/inet.h>

namespace pccl::runtime {

static std::unique_ptr<communicator::AsioOobChannel> oob_channel = nullptr;
static std::unique_ptr<engine::MemoryManager> memory_manager = nullptr;
static std::unique_ptr<communicator::ChannelManager> channel_manager = nullptr;
static std::unique_ptr<engine::GraphExecutor> graph_executor = nullptr;
static std::map<int, RuntimeConfig> cluster_configs;
static int current_rank = -1;
static int world_size = 0;

class RuntimeCommInterface : public engine::MemoryManagerCommInterface {
public:
  RuntimeCommInterface(communicator::ChannelManager* channel_mgr)
    : channel_manager_(channel_mgr) {}

  bool registerMemoryRegion(const engine::GlobalBufferID& buffer_id) override {
    if (!channel_manager_) return false;
    
    auto endpoints = channel_manager_->getConnectedEndpoints();
    for (const auto& endpoint : endpoints) {
      auto channel = channel_manager_->getChannel(endpoint);
      if (channel) {
        communicator::MemRegion region;
        region.ptr_ = buffer_id.addr;
        region.size_ = buffer_id.getBufferSize();
        region.lkey_ = 0;
        region.rkey_ = 0;
        
        channel->registerMemoryRegion(region);
      }
    }
    return true;
  }

  bool deregisterMemoryRegion(const engine::GlobalBufferID& buffer_id) override {
    if (!channel_manager_) return false;
    
    auto endpoints = channel_manager_->getConnectedEndpoints();
    for (const auto& endpoint : endpoints) {
      auto channel = channel_manager_->getChannel(endpoint);
      if (channel) {
        communicator::MemRegion region;
        region.ptr_ = buffer_id.addr;
        region.size_ = buffer_id.getBufferSize();
        channel->deregisterMemoryRegion(region);
      }
    }
    return true;
  }

  bool syncWorkspaceAllocation(const engine::WorkspaceHandle& handle) override {
    if (!oob_channel) return false;
    
    nlohmann::json allocation_info;
    allocation_info["operator_id"] = handle.operator_id;
    allocation_info["participants"] = handle.participant_ranks;
    
    communicator::OobMessage msg;
    msg.type_ = communicator::OobMsgType::BUFFER_UPDATE;
    msg.src_rank = current_rank;
    msg.payload = allocation_info.dump();
    
    for (int rank : handle.participant_ranks) {
      if (rank == current_rank) continue;
      
      auto it = cluster_configs.find(rank);
      if (it != cluster_configs.end()) {
        communicator::Endpoint peer_endpoint;
        peer_endpoint.attributes_ = it->second.endpoint_configs;
        oob_channel->send(msg, peer_endpoint);
      }
    }
    
    return true;
  }

  bool waitForSignal(uint64_t signal_id, int timeout_ms) override {
    auto start_time = std::chrono::steady_clock::now();
    while (std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start_time).count() < timeout_ms) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return true;
  }

  bool sendSignal(uint64_t signal_id, int target_rank) override {
    if (!oob_channel) return false;
    
    communicator::OobMessage msg;
    msg.type_ = communicator::OobMsgType::HEARTBEAT;
    msg.src_rank = current_rank;
    msg.payload = std::to_string(signal_id);
    
    auto it = cluster_configs.find(target_rank);
    if (it != cluster_configs.end()) {
      communicator::Endpoint peer_endpoint;
      peer_endpoint.attributes_ = it->second.endpoint_configs;
      return oob_channel->send(msg, peer_endpoint);
    }
    
    return false;
  }

private:
  communicator::ChannelManager* channel_manager_;
};

void registerCommunicationResources(RuntimeConfig& config) {
  auto& verbs_lib = pccl::communicator::VerbsLib::getInstance();
  verbs_lib.load("libibverbs.so");
  int num_devices;
  auto devices = verbs_lib.getDeviceList(&num_devices);
  for (int idx = 0; idx < num_devices; idx ++) {
    auto context = verbs_lib.openDevice(devices[0]);
    if (context && verbs_lib.getDeviceName(devices[idx]) == config.endpoint_configs["pccl.roce.device_name"]) {
      ibv_gid self_gid;
      int port_num = std::stoi(config.endpoint_configs["pccl.roce.port_num"]);
      int gid_index = std::stoi(config.endpoint_configs["pccl.roce.gid_index"]);
      verbs_lib.queryGid(context, port_num, gid_index, &self_gid);
      char tmp[sizeof(self_gid) + 1];
      memcpy(tmp, (void *)&self_gid, sizeof(tmp));
      config.endpoint_configs["pccl.roce.gid"] = tmp;
      verbs_lib.closeDevice(context);
      break;
    }
    verbs_lib.closeDevice(context);
  }
  verbs_lib.freeDeviceList(devices);
}

std::string RuntimeConfig::toJson() {
  nlohmann::json j;
  j["rank"] = rank;
  j["world_size"] = world_size;
  
  nlohmann::json buffers_json;
  for (const auto& [exec_type, count] : buffer_nums) {
    buffers_json[std::to_string(static_cast<int>(exec_type))] = count;
  }
  j["buffer_nums"] = buffers_json;
  
  nlohmann::json sizes_json;
  for (const auto& [exec_type, size] : buffer_sizes) {
    sizes_json[std::to_string(static_cast<int>(exec_type))] = size;
  }
  j["buffer_sizes"] = sizes_json;
  
  j["endpoint_configs"] = endpoint_configs;
  return j.dump();
}

RuntimeConfig RuntimeConfig::fromJson(std::string json) {
  RuntimeConfig config;
  try {
    auto j = nlohmann::json::parse(json);
    config.rank = j.value("rank", 0);
    config.world_size = j.value("world_size", 0);
    
    if (j.contains("buffer_nums")) {
      for (auto it = j["buffer_nums"].begin(); it != j["buffer_nums"].end(); ++it) {
        ExecutorType exec_type = static_cast<ExecutorType>(std::stoi(it.key()));
        config.buffer_nums[exec_type] = it.value();
      }
    }
    
    if (j.contains("buffer_sizes")) {
      for (auto it = j["buffer_sizes"].begin(); it != j["buffer_sizes"].end(); ++it) {
        ExecutorType exec_type = static_cast<ExecutorType>(std::stoi(it.key()));
        config.buffer_sizes[exec_type] = it.value();
      }
    }
    
    if (j.contains("endpoint_configs")) {
      config.endpoint_configs = j["endpoint_configs"].get<std::map<std::string, std::string>>();
    }
  } catch (const std::exception& e) {
    PCCL_LOG_ERROR("Failed to parse RuntimeConfig from JSON: {}", e.what());
  }
  return config;
}

bool initializeRuntime(std::vector<RuntimeConfig>& runtime_configs, int rank, int world_size_) {
  current_rank = rank;
  world_size = world_size_;
  
  for (int i = 0; i < world_size; i++) {
    cluster_configs[i] = runtime_configs[i];
  }

  RuntimeConfig& self_config = cluster_configs[rank];
  PCCL_LOG_INFO("Initializing runtime for rank {}", rank);

  registerCommunicationResources(self_config);

  oob_channel = std::make_unique<communicator::AsioOobChannel>();
  communicator::Endpoint self_endpoint;
  self_endpoint.attributes_ = self_config.endpoint_configs;
  
  if (!oob_channel->init(self_endpoint)) {
    PCCL_LOG_ERROR("Failed to initialize OOB communication");
    return false;
  }
  PCCL_LOG_DEBUG("OOB communication initialized");

  memory_manager = std::make_unique<engine::MemoryManager>();
  
  if (!memory_manager->initialize(self_config)) {
    PCCL_LOG_ERROR("Failed to initialize memory manager");
    return false;
  }

  PCCL_LOG_DEBUG("Memory manager initialized");

  auto local_buffers = memory_manager->getLocalBuffers();
  
  for (const auto& buffer_id : local_buffers) {
    std::string key = std::format("pccl.buffer.{}.{}.{}", 
      rank, 
      static_cast<int>(buffer_id.getExecutorType()), 
      buffer_id.getBufferIdx());
    self_config.endpoint_configs[key] = buffer_id.toJson().dump();
    PCCL_LOG_DEBUG("Registered buffer: {}", key);
  }

  self_config.endpoint_configs["pccl.runtime.initialized"] = "true";
  self_config.endpoint_configs["pccl.runtime.rank"] = std::to_string(rank);

  channel_manager = std::make_unique<communicator::ChannelManager>();

  bool use_tcp = self_config.endpoint_configs.count("pccl.runtime.use_tcp") > 0 &&
                  self_config.endpoint_configs.at("pccl.runtime.use_tcp") == "true";
  
  bool use_roce = self_config.endpoint_configs.count("pccl.runtime.use_roce") > 0 &&
                  self_config.endpoint_configs.at("pccl.runtime.use_roce") == "true";

  // if (use_tcp) {
  //   channel_manager->registerEngineFactory(communicator::ChannelType::TCP,
  //     [](const communicator::Endpoint& local, const communicator::Endpoint& remote) {
  //       return std::make_unique<communicator::TCPAdapter>(local, remote);
  //     });
  //   PCCL_LOG_DEBUG("Registered TCP engine factory");
  // }
  
  // if (use_roce) {
  //   channel_manager->registerEngineFactory(communicator::ChannelType::RDMA,
  //     [](const communicator::Endpoint& local, const communicator::Endpoint& remote) {
  //       return std::make_unique<communicator::RoCEAdapter>(local, remote);
  //     });
  //   PCCL_LOG_DEBUG("Registered RoCE engine factory");
  // }

  // if (!channel_manager->initialize(self_endpoint)) {
  //   PCCL_LOG_ERROR("Failed to initialize channel manager");
  //   return false;
  // }
  // PCCL_LOG_DEBUG("Channel manager initialized");

  // self_config.endpoint_configs["pccl.channel.initialized"] = "true";

  // std::atomic<int> configs_received = 1;
  // std::mutex config_mutex;
  // std::condition_variable config_cv;
  
  // auto config_handler = [&](const communicator::OobMessage& msg) {
  //   if (msg.src_rank == rank) return;
    
  //   try {
  //     RuntimeConfig peer_config = RuntimeConfig::fromJson(msg.payload);
  //     {
  //       std::lock_guard<std::mutex> lock(config_mutex);
  //       cluster_configs[msg.src_rank] = peer_config;
  //       int received = ++configs_received;
  //       PCCL_LOG_DEBUG("Received config from rank {}, total received: {}", 
  //                     msg.src_rank, received);
        
  //       if (received == world_size) {
  //         config_cv.notify_all();
  //       }
  //     }
  //   } catch (const std::exception& e) {
  //     PCCL_LOG_ERROR("Failed to process config from rank {}: {}", msg.src_rank, e.what());
  //   }
  // };
  
  // oob_channel->registerHandler(communicator::OobMsgType::CONFIG_SYNC, config_handler);

  // communicator::OobMessage sync_msg;
  // sync_msg.type_ = communicator::OobMsgType::CONFIG_SYNC;
  // sync_msg.src_rank = rank;
  // sync_msg.payload = self_config.toJson();
  // sync_msg.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
  //   std::chrono::system_clock::now().time_since_epoch()).count();

  // std::vector<communicator::Endpoint> peer_endpoints;
  // for (int i = 0; i < world_size; i++) {
  //   if (i != rank) {
  //     communicator::Endpoint peer_endpoint;
  //     peer_endpoint.attributes_ = &runtime_configs[i].endpoint_configs;
  //     peer_endpoints.push_back(peer_endpoint);
  //   }
  // }

  // if (!peer_endpoints.empty() && !oob_channel->broadcast(sync_msg, peer_endpoints)) {
  //   PCCL_LOG_ERROR("Failed to broadcast configuration to some peers");
  // }
  // PCCL_LOG_DEBUG("Configuration broadcasted to {} peers", peer_endpoints.size());

  // std::unique_lock<std::mutex> lock(config_mutex);
  // if (world_size > 1) {
  //   auto timeout = std::chrono::steady_clock::now() + std::chrono::seconds(30);
  //   if (!config_cv.wait_until(lock, timeout, [&]() { return configs_received == world_size; })) {
  //     PCCL_LOG_ERROR("Configuration synchronization timeout. Received {}/{} configs", 
  //                   configs_received.load(), world_size);
  //     return false;
  //   }
  // }
  // PCCL_LOG_INFO("Configuration synchronization completed");

  // auto comm_interface = std::make_shared<RuntimeCommInterface>(channel_manager.get());
  // memory_manager->setCommInterface(comm_interface);

  // if (!memory_manager->initialize_cluster(cluster_configs)) {
  //   PCCL_LOG_ERROR("Failed to initialize memory manager cluster");
  //   return false;
  // }
  // PCCL_LOG_DEBUG("Memory manager cluster initialized");

  // for (int i = 0; i < world_size; i++) {
  //   if (i != rank) {
  //     communicator::Endpoint peer_endpoint;
  //     peer_endpoint.attributes_ = &cluster_configs[i].endpoint_configs;
      
  //     if (peer_endpoint.attributes_->count("pccl.runtime.initialized") == 0) {
  //       PCCL_LOG_WARN("Rank {} not initialized, skipping channel creation", i);
  //       continue;
  //     }
      
  //     auto channel = channel_manager->getOrCreateChannel(peer_endpoint);
  //     if (!channel) {
  //       PCCL_LOG_WARN("Failed to create channel to rank {}", i);
  //     } else {
  //       PCCL_LOG_DEBUG("Channel to rank {} established", i);
  //     }
  //   }
  // }

  // graph_executor = std::make_unique<engine::GraphExecutor>();
  
  PCCL_LOG_INFO("Runtime initialized successfully for rank {}", rank);
  return true;
}

bool updatePeer(RuntimeConfig& peer_config) {
  return false;
}

void shutdownRuntime() {
  PCCL_LOG_INFO("Shutting down runtime");
  
  if (graph_executor) {
    graph_executor.reset();
    PCCL_LOG_DEBUG("Graph executor shutdown");
  }
  
  if (memory_manager) {
    memory_manager->shutdown();
    memory_manager.reset();
    PCCL_LOG_DEBUG("Memory manager shutdown");
  }

  if (channel_manager) {
    channel_manager->shutdown();
    channel_manager.reset();
    PCCL_LOG_DEBUG("Channel manager shutdown");
  }
  
  if (oob_channel) {
    oob_channel->shutdown();
    oob_channel.reset();
    PCCL_LOG_DEBUG("OOB channel shutdown");
  }
  
  cluster_configs.clear();
  current_rank = -1;
  world_size = 0;
  
  PCCL_LOG_INFO("Runtime shutdown completed");
}

bool executeGraph(const PrimitiveGrpah& graph, 
                 std::vector<int>& participants, 
                 torch::Tensor& input, torch::Tensor& output) {
  return false;
}

uint64_t generateOperatorId() {
  static std::atomic<uint64_t> next_id{1};
  return next_id.fetch_add(1);
}


std::map<ExecutorType, int> getExecutorConfig(const PrimitiveGrpah& graph) {
  std::map<ExecutorType, int> config;
  
  auto executors = graph.getExecutors();
  for (auto executor_type : executors) {
    config[executor_type] = 1;
  }
  
  if (config.empty()) {
    config[ExecutorType::CUDA] = 12;
    config[ExecutorType::CPU] = 1;
  }
  
  return config;
}

} // namespace pccl::runtime
