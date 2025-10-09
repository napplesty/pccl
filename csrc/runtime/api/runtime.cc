#include "runtime/api/runtime.h"
#include "runtime/api/repr.h"
#include "runtime/communicator/oob_comm.h"
#include "runtime/engine/graph_executor.h"
#include "runtime/engine/memory_manager.h"
#include "runtime/communicator/channel.h"
#include "plugins/atcp/tcp_adapter.h"
#include "plugins/aroce/roce_adapter.h"
#include "plugins/aroce/roce_utils.h"
#include "utils/logging.h"
#include "utils/hex_utils.hpp"
#include "utils/debug.hpp"
#include <cstdint>
#include <cstring>
#include <format>
#include <infiniband/verbs.h>
#include <map>
#include <memory>
#include <chrono>
#include <ifaddrs.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <tuple>
#include <vector>

namespace pccl::runtime {

static std::unique_ptr<communicator::AsioOobChannel> oob_channel = nullptr;
static std::unique_ptr<engine::MemoryManager> memory_manager = nullptr;
static std::unique_ptr<communicator::ChannelManager> channel_manager = nullptr;
static std::unique_ptr<engine::GraphExecutor> graph_executor = nullptr;
static std::map<int, RuntimeConfig> cluster_configs;
static int current_rank = -1;
static int world_size = 0;
static std::unique_ptr<pccl::communicator::VerbsManager> global_verbs_manager = nullptr;

static std::tuple<int, int, int> get_signature(engine::GlobalBufferID& buffer_id) {
  int rank = buffer_id.getRank();
  int type = (int)buffer_id.getExecutorType();
  int index = buffer_id.getBufferIdx();
  return std::make_tuple(rank, type, index);
}

static void portable_roce_read_sync(int src_rank, engine::GlobalBufferID& dstBuffer, engine::GlobalBufferID& srcBuffer) {
  int self_rank = current_rank;
  std::map<std::string, std::string> &self_attrs = cluster_configs[self_rank].endpoint_configs;
  uint64_t qp_id, conn_id;
  utils::unmarshal_from_hex_str(&qp_id, self_attrs.at(std::format("pccl.roce.{}.{}.qp_id", self_rank, src_rank)));
  utils::unmarshal_from_hex_str(&conn_id, self_attrs.at("pccl.roce.conn_id.global"));

  uint32_t lkey, rkey;
  utils::unmarshal_from_hex_str(&lkey, dstBuffer.shareable_handles.at("lkey"));
  utils::unmarshal_from_hex_str(&rkey, srcBuffer.shareable_handles.at("lkey"));
  
  ibv_send_wr wr{};
  ibv_sge sge{};

  sge.addr = reinterpret_cast<uint64_t>(dstBuffer.addr);
  sge.length = dstBuffer.getBufferSize();
  sge.lkey = lkey;

  wr.wr_id = 0xbf16f32f16f8;
  wr.next = nullptr;
  wr.sg_list = &sge;

  wr.num_sge = 1;
  wr.opcode = IBV_WR_RDMA_READ;
  wr.send_flags = IBV_SEND_SIGNALED;

  wr.wr.rdma.remote_addr = reinterpret_cast<uint64_t>(srcBuffer.addr);
  wr.wr.rdma.rkey = rkey;

  if (!global_verbs_manager->postSend(conn_id, qp_id, &wr, nullptr)) {
    PCCL_LOG_CRITICAL("Fail to read");
  }

  ibv_wc wc;
  while(true) {
    global_verbs_manager->pollCQ(conn_id, 1, &wc);
    if(wc.wr_id == 0xbf16f32f16f8) {
      if (wc.status == IBV_WC_SUCCESS) {
        break;
      } else {
        PCCL_LOG_CRITICAL("Fail to read");
      }
    }
  }
  
  return;
}

static std::vector<engine::GlobalBufferID> portable_parse_signals(void *signals, uint64_t operator_id, int src_rank, int num_expected) {
  int num_loaded = 0;
  int64_t *signal_buffer_ptr = (int64_t *)signals;
  for(int i = 0; i < engine::max_buffers_per_type; i++) {
    if (signal_buffer_ptr[(int)ExecutorType::CUDA * engine::max_buffers_per_type + i] == (int64_t)operator_id) {
      num_loaded ++;
    }
  }
  if (num_expected != num_loaded) {
    return {};
  }
  std::vector<engine::GlobalBufferID> buff;
  for(int i = 0; i < engine::max_buffers_per_type; i++) {
    if (signal_buffer_ptr[(int)ExecutorType::CUDA * engine::max_buffers_per_type + i] == (int64_t)operator_id) {
      std::tuple<int, int, int> triple = std::make_tuple(src_rank, (int)ExecutorType::CUDA, i);
      buff.push_back(memory_manager->get_buffer_with_triple(triple));
    }
  }
  return buff;
}

static std::map<std::tuple<int, int, int>, std::shared_ptr<pccl::communicator::VerbsMemoryRegion>> cache;

class RuntimeCommInterface : public engine::MemoryManagerCommInterface {
public:
  bool registerMemoryRegion(engine::GlobalBufferID& buffer_id) override {
    if (!global_verbs_manager) return false;
    int access = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | 
                 IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_ATOMIC;
    auto mr = std::make_shared<pccl::communicator::VerbsMemoryRegion>(
      *global_verbs_manager->getPD(), buffer_id.addr, buffer_id.getBufferSize(), access);
    
    std::tuple<int, int, int> rti = get_signature(buffer_id);
    cache[rti] = mr;

    uint32_t rkey = mr->getRKey();
    uint32_t lkey = mr->getLKey();
    
    buffer_id.shareable_handles["rkey"] = utils::marshal_to_hex_str(&rkey, sizeof(rkey));
    buffer_id.shareable_handles["lkey"] = utils::marshal_to_hex_str(&lkey, sizeof(lkey));

    return true;
  }

  bool deregisterMemoryRegion(engine::GlobalBufferID& buffer_id) override {
    return true;
  }

  bool syncWorkspaceAllocation(engine::WorkspaceHandle& handle) override {
    std::vector<int> &participants = handle.participant_ranks;
    engine::GlobalBufferID &tmp_buf = memory_manager->get_tmp_buffer();
    for (int participant : participants) {
      if (participant == current_rank) {
        continue;
      }
      auto triple = std::make_tuple(participant, (int)ExecutorType::CPU, 0);
      engine::GlobalBufferID &src_buf = memory_manager->get_buffer_with_triple(triple);
      while(true) {
        portable_roce_read_sync(participant, tmp_buf, src_buf);
        std::vector<engine::GlobalBufferID> bufs = \
            portable_parse_signals(tmp_buf.addr, handle.operator_id, participant, handle.buffers.at(current_rank).size());
        if (bufs.size() > 0) {
          handle.buffers[participant] = bufs;
          break;
        }
      }
    }
    return true;
  }

  bool waitForSignal(uint64_t signal_id, int timeout_ms) override {
    return true;
  }

  bool sendSignal(uint64_t signal_id, int target_rank) override {
    return true;
  }
};

void registerCommunicationResources(RuntimeConfig& config) {
  if (config.endpoint_configs.at("pccl.runtime.use_roce") == "true") {
    auto& verbs_lib = pccl::communicator::VerbsLib::getInstance();
    verbs_lib.load("libibverbs.so");
    
    std::string device_name = config.endpoint_configs["pccl.roce.device_name"];
    uint8_t port_num = std::stoi(config.endpoint_configs["pccl.roce.port_num"]);
    
    global_verbs_manager = std::make_unique<pccl::communicator::VerbsManager>();
    if (!global_verbs_manager->initialize(device_name, port_num)) {
      PCCL_LOG_ERROR("Failed to initialize Verbs manager for QP precreation");
      return;
    }

    config.endpoint_configs["pccl.roce.verbsManager.global"] = \
        utils::marshal_to_hex_str(&global_verbs_manager, sizeof(global_verbs_manager));
    
    int gid_index = std::stoi(config.endpoint_configs["pccl.roce.gid_index"]);
    
    ibv_gid self_gid;
    auto context = global_verbs_manager->getContext();
    if (context) {
      verbs_lib.queryGid(context->get(), port_num, gid_index, &self_gid);
      config.endpoint_configs["pccl.roce.gid"] = utils::marshal_to_hex_str((void *)&self_gid, sizeof(self_gid));
    } else {
      PCCL_LOG_ERROR("Failed to get context for GID query");
      return;
    }
    
    pccl::communicator::VerbsManager::ConnectionConfig conn_config;
    conn_config.port_num = port_num;
    conn_config.gid_index = gid_index;
    conn_config.max_qp_per_connection = world_size - 1;
    conn_config.cq_size = 8192;
    
    uint64_t conn_id = global_verbs_manager->createConnection(conn_config);
    if (conn_id == 0) {
      PCCL_LOG_ERROR("Failed to create connection for QP precreation");
      return;
    }

    config.endpoint_configs["pccl.roce.conn_id.global"] = \
        utils::marshal_to_hex_str(&conn_id, sizeof(conn_id));
    
    pccl::communicator::VerbsManager::QPConfig qp_config;
    qp_config.qp_type = IBV_QPT_RC;
    qp_config.max_send_wr = 1024;
    qp_config.max_recv_wr = 1024;
    qp_config.max_send_sge = 1;
    qp_config.max_recv_sge = 1;
    qp_config.max_inline_data = 0;
    
    for (int peer_rank = 0; peer_rank < world_size; peer_rank++) {
      if (peer_rank == current_rank) continue;
      
      uint64_t qp_id = global_verbs_manager->createQP(conn_id, qp_config);
      if (qp_id == 0) {
        PCCL_LOG_ERROR("Failed to create QP for peer rank {}", peer_rank);
        continue;
      }
      
      if (!global_verbs_manager->modifyQPToInit(conn_id, qp_id)) {
        PCCL_LOG_ERROR("Failed to modify QP to INIT for peer rank {}", peer_rank);
        continue;
      }
      
      auto metadata = global_verbs_manager->getLocalMetadata(conn_id, qp_id);
      
      std::string qp_id_key = std::format("pccl.roce.{}.{}.qp_id", current_rank, peer_rank);
      std::string qp_key = std::format("pccl.roce.{}.{}.qp_num", current_rank, peer_rank);
      std::string lid_key = std::format("pccl.roce.{}.{}.lid", current_rank, peer_rank);
      std::string gid_key = std::format("pccl.roce.{}.{}.gid", current_rank, peer_rank);
      
      config.endpoint_configs[qp_id_key] = utils::marshal_to_hex_str(&qp_id, sizeof(qp_id));
      config.endpoint_configs[qp_key] = std::to_string(metadata.qp_num);
      config.endpoint_configs[lid_key] = std::to_string(metadata.lid);
      config.endpoint_configs[gid_key] = utils::marshal_to_hex_str(&metadata.gid, sizeof(metadata.gid));
      
      PCCL_LOG_DEBUG("Precreated QP for peer {}: qp_num={}, lid={}", peer_rank, metadata.qp_num, metadata.lid);
    }
  }
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

  self_config.endpoint_configs["pccl.runtime.rank"] = std::to_string(rank);
  self_config.endpoint_configs["pccl.runtime.world_size"] = std::to_string(world_size);

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
  auto comm_interface = std::make_shared<RuntimeCommInterface>();
  memory_manager->setCommInterface(comm_interface);
  
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

  channel_manager = std::make_unique<communicator::ChannelManager>();

  bool use_tcp = self_config.endpoint_configs.count("pccl.runtime.use_tcp") > 0 &&
                  self_config.endpoint_configs.at("pccl.runtime.use_tcp") == "true";
  
  bool use_roce = self_config.endpoint_configs.count("pccl.runtime.use_roce") > 0 &&
                  self_config.endpoint_configs.at("pccl.runtime.use_roce") == "true";

  if (use_tcp) {
    channel_manager->registerEngineFactory(communicator::ChannelType::TCP,
      [](const communicator::Endpoint& local, const communicator::Endpoint& remote) {
        return std::make_unique<communicator::TCPAdapter>(local, remote);
      });
    PCCL_LOG_DEBUG("Registered TCP engine factory");
  }
  
  if (use_roce) {
    channel_manager->registerEngineFactory(communicator::ChannelType::RDMA,
      [](const communicator::Endpoint& local, const communicator::Endpoint& remote) {
        return std::make_unique<communicator::RoCEAdapter>(local, remote);
      });
    PCCL_LOG_DEBUG("Registered RoCE engine factory");
  }

  if (!channel_manager->initialize(self_endpoint)) {
    PCCL_LOG_ERROR("Failed to initialize channel manager");
    return false;
  }
  PCCL_LOG_DEBUG("Channel manager initialized");

  self_config.endpoint_configs["pccl.channel.initialized"] = "true";

  std::atomic<int> configs_received = 1;
  std::mutex config_mutex;
  std::condition_variable config_cv;
  
  auto config_handler = [&](const communicator::OobMessage& msg) {
    if (msg.src_rank == rank) return;
    
    try {
      RuntimeConfig peer_config = RuntimeConfig::fromJson(msg.payload);
      {
        std::lock_guard<std::mutex> lock(config_mutex);
        cluster_configs[msg.src_rank] = peer_config;
        int received = ++configs_received;
        PCCL_LOG_DEBUG("Received config from rank {}, total received: {}", 
                      msg.src_rank, received);
        
        if (received == world_size) {
          config_cv.notify_all();
        }
      }
    } catch (const std::exception& e) {
      PCCL_LOG_ERROR("Failed to process config from rank {}: {}", msg.src_rank, e.what());
    }
  };
  
  oob_channel->registerHandler(communicator::OobMsgType::CONFIG_SYNC, config_handler);

  communicator::OobMessage sync_msg;
  sync_msg.type_ = communicator::OobMsgType::CONFIG_SYNC;
  sync_msg.src_rank = rank;
  sync_msg.payload = self_config.toJson();
  sync_msg.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
    std::chrono::system_clock::now().time_since_epoch()).count();

  std::vector<communicator::Endpoint> peer_endpoints;
  for (int i = 0; i < world_size; i++) {
    if (i != rank) {
      communicator::Endpoint peer_endpoint;
      peer_endpoint.attributes_ = runtime_configs[i].endpoint_configs;
      peer_endpoints.push_back(peer_endpoint);
    }
  }

  if (!peer_endpoints.empty() && !oob_channel->broadcast(sync_msg, peer_endpoints)) {
    PCCL_LOG_ERROR("Failed to broadcast configuration to some peers");
  }
  PCCL_LOG_DEBUG("Configuration broadcasted to {} peers", peer_endpoints.size());

  std::unique_lock<std::mutex> lock(config_mutex);
  if (world_size > 1) {
    auto timeout = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    if (!config_cv.wait_until(lock, timeout, [&]() { return configs_received == world_size; })) {
      PCCL_LOG_ERROR("Configuration synchronization timeout. Received {}/{} configs", 
                    configs_received.load(), world_size);
      return false;
    }
  }

  PCCL_LOG_INFO("Configuration synchronization completed");

  for (int i = 0; i < world_size; i++) {
    if (i != rank) {
      communicator::Endpoint peer_endpoint;
      peer_endpoint.attributes_ = cluster_configs[i].endpoint_configs;
      
      if (peer_endpoint.attributes_.count("pccl.runtime.initialized") == 0) {
        PCCL_LOG_WARN("Rank {} not initialized, skipping channel creation", i);
        continue;
      }

      PCCL_LOG_DEBUG("Channel {} <-> {} establishing", self_config.endpoint_configs.at("pccl.runtime.rank"), peer_endpoint.attributes_.at("pccl.runtime.rank"));
      auto channel = channel_manager->getOrCreateChannel(peer_endpoint);
      if (!channel) {
        PCCL_LOG_WARN("Failed to create channel to rank {}", i);
      } else {
        PCCL_LOG_DEBUG("Channel to rank {} established", i);
      }
    }
  }

  if (!memory_manager->initialize_cluster(cluster_configs)) {
    PCCL_LOG_ERROR("Failed to initialize memory manager cluster");
    return false;
  }

  PCCL_LOG_DEBUG("Memory manager cluster initialized");

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
  
  if (global_verbs_manager) {
    global_verbs_manager.reset();
    PCCL_LOG_DEBUG("Global verbs manager shutdown");
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

bool executeGraph(PrimitiveGrpah& graph, 
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
