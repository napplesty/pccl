#pragma once

#include <array>    // For std::array in command execution
#include <atomic>   // For std::atomic in server
#include <chrono>   // For std::chrono::milliseconds in server
#include <cstdint>  // For fixed-width integer types like uint32_t
#include <cstdio>   // For popen, pclose, fgets
#include <iostream> // For std::cout, std::cerr (mainly for server logging)
#include <map>
#include <memory>    // For std::unique_ptr, std::make_unique
#include <optional>  // C++17, for optional parameters
#include <sstream>   // For std::istringstream, std::ostringstream
#include <stdexcept> // For std::runtime_error
#include <string>
#include <thread> // For std::thread in server
#include <vector>

// Attempt to include cpp-httplib.
// Ensure your build system has 'thirdparty/cpp-httplib' in its include paths
// or adjust the path below accordingly (e.g.,
// "../../../../thirdparty/cpp-httplib/httplib.h")
#include "httplib.h"

// 前向声明，如果需要的话
// class SomeOtherDependency;

namespace pccl {
namespace component {
namespace netconf_model {
namespace ovs {

// 定义流表项匹配字段的结构体
struct MatchFields {
  std::optional<uint32_t> in_port;   // 输入端口
  std::optional<std::string> dl_src; // 源MAC地址
  std::optional<std::string> dl_dst; // 目的MAC地址
  std::optional<uint16_t> dl_type;   // EtherType (例如 0x0800 for IPv4)
  std::optional<uint16_t> vlan_vid;  // VLAN ID
  std::optional<uint8_t> vlan_pcp;   // VLAN PCP
  std::optional<std::string> nw_src; // 源IP地址
  std::optional<std::string> nw_dst; // 目的IP地址
  std::optional<uint8_t> nw_proto;   // IP协议 (例如 6 for TCP, 17 for UDP)
  std::optional<uint8_t> nw_tos;     // IP ToS
  std::optional<uint16_t> tp_src;    // TCP/UDP 源端口
  std::optional<uint16_t> tp_dst;    // TCP/UDP 目的端口
  // 可以根据需要添加更多匹配字段
  // 例如：std::optional<uint32_t> tunnel_id; // 隧道ID
};

// 定义流表项动作的结构体
struct Action {
  enum class Type {
    OUTPUT,     // 转发到指定端口
    CONTROLLER, // 发送到控制器
    DROP,       // 丢弃
    SET_FIELD,  // 修改字段 (例如 set_eth_dst, set_ipv4_src)
    NORMAL,     // 执行正常的L2/L3转发
    FLOOD,      // 洪泛 (不包括输入端口)
    ALL,        // 洪泛 (包括输入端口)
    IN_PORT,    // 从输入端口发送出去
                // 可以添加更多动作类型
  };
  Type type;
  std::map<std::string, std::string>
      params; // 动作参数，例如 {"port": "1"}, {"field": "eth_dst", "value":
              // "00:11:22:33:44:55"}

  Action(Type t, std::map<std::string, std::string> p = {})
      : type(t), params(std::move(p)) {}

  static Action Output(uint32_t port_num) {
    return Action(Type::OUTPUT, {{"port", std::to_string(port_num)}});
  }
  static Action
  Controller(uint32_t controller_id = 0 /* default controller */) {
    return Action(Type::CONTROLLER,
                  {{"controller_id", std::to_string(controller_id)}});
  }
  static Action Drop() { return Action(Type::DROP); }
  // 可以为其他常用动作添加静态工厂方法
};

// 定义流表项的结构体
struct FlowEntry {
  uint32_t flow_id; // 流表项的唯一标识（可选，OVS内部管理）
  uint32_t table_id = 0;                // 流表ID (默认为0)
  uint32_t priority = 32768;            // 优先级 (0-65535, 越大越高)
  MatchFields match;                    // 匹配字段
  std::vector<Action> actions;          // 动作列表
  std::optional<uint16_t> idle_timeout; // 空闲超时时间 (秒)
  std::optional<uint16_t> hard_timeout; // 硬超时时间 (秒)
  std::optional<uint64_t> cookie; // Cookie值，可用于过滤和管理流表项
  // 可以添加其他元数据，例如创建时间等

  FlowEntry(uint32_t id, const MatchFields &m, const std::vector<Action> &act)
      : flow_id(id), match(m), actions(act) {}
};

// 定义端口的结构体
struct Port {
  uint32_t port_no; // OVS内部的端口号 (ofport)
  std::string name; // 端口名称 (例如 "eth0", "veth1")
  std::string type; // 端口类型 (例如 "internal", "system", "patch", "vxlan")
  std::map<std::string, std::string>
      options; // 端口选项 (例如 {"remote_ip": "1.2.3.4"} for vxlan)
  std::optional<std::string> mac_address; // 端口的MAC地址
  // 可以添加其他端口状态信息，例如 link_state, speed 等

  Port(uint32_t no, const std::string &n, const std::string &t = "internal")
      : port_no(no), name(n), type(t) {}
};

// 定义设备间连接的结构体
struct DeviceLink {
  std::string link_id;       // 连接的唯一标识
  std::string src_device_id; // 源设备ID (例如 OVS switch name)
  std::string src_port_name; // 源端口名称
  std::string dst_device_id; // 目的设备ID
  std::string dst_port_name; // 目的端口名称
  enum class LinkType {
    PATCH,  // OVS patch port 连接
    DIRECT, // 物理直连或VETH pair
    TUNNEL  // 例如 VXLAN, GRE
  } type = LinkType::DIRECT;
  std::map<std::string, std::string> properties; // 连接属性，例如隧道参数

  DeviceLink(std::string id, std::string src_dev, std::string src_port,
             std::string dst_dev, std::string dst_port)
      : link_id(std::move(id)), src_device_id(std::move(src_dev)),
        src_port_name(std::move(src_port)), dst_device_id(std::move(dst_dev)),
        dst_port_name(std::move(dst_port)) {}
};

// OVS交换机控制类
class OvsSwitch {
public:
  explicit OvsSwitch(std::string switch_name) : name_(std::move(switch_name)) {}
  virtual ~OvsSwitch() = default;

  // --- 流表操作 ---

  /**
   * @brief 添加一条流表项
   * @param entry 要添加的流表项
   * @return 操作是否成功
   */
  virtual bool addFlow(const FlowEntry &entry) = 0;

  /**
   * @brief 修改一条已存在的流表项 (通常通过删除旧的，添加新的来实现)
   *        或者根据OVS的能力，精确修改某些字段。
   * @param entry 修改后的流表项 (其flow_id或唯一匹配条件应能定位到旧流表)
   * @return 操作是否成功
   */
  virtual bool modifyFlow(const FlowEntry &entry) = 0;

  /**
   * @brief 删除一条流表项
   * @param match 用以匹配要删除流表项的条件 (可以是部分匹配)
   * @param table_id 可选，指定流表ID
   * @param strict 如果为true，则match必须精确匹配
   * @return 操作是否成功
   */
  virtual bool deleteFlow(const MatchFields &match,
                          std::optional<uint32_t> table_id = std::nullopt,
                          bool strict = false) = 0;

  /**
   * @brief 删除交换机上的所有流表项
   * @return 操作是否成功
   */
  virtual bool deleteAllFlows() = 0;

  /**
   * @brief 查询流表项
   * @param match 可选，用以过滤流表项的匹配条件
   * @param table_id 可选，指定流表ID
   * @return 符合条件的流表项列表
   */
  virtual std::vector<FlowEntry>
  dumpFlows(std::optional<MatchFields> match = std::nullopt,
            std::optional<uint32_t> table_id = std::nullopt) = 0;

  // --- 端口操作 ---

  /**
   * @brief 添加一个端口到OVS交换机
   * @param port 要添加的端口信息
   * @return 操作是否成功，如果端口已存在可能返回false或更新成功
   */
  virtual bool addPort(const Port &port) = 0;

  /**
   * @brief 从OVS交换机删除一个端口
   * @param port_name 要删除的端口名称
   * @return 操作是否成功
   */
  virtual bool deletePort(const std::string &port_name) = 0;

  /**
   * @brief 修改端口配置 (例如，设置VLAN tag, trunk等)
   * @param port_name 要修改的端口名
   * @param attributes 要修改的属性和新值
   * @return 操作是否成功
   */
  virtual bool
  modifyPort(const std::string &port_name,
             const std::map<std::string, std::string> &attributes) = 0;

  /**
   * @brief 获取交换机上的所有端口信息
   * @return 端口列表
   */
  virtual std::vector<Port> listPorts() = 0;

  /**
   * @brief 获取指定端口的详细信息
   * @param port_name 端口名称
   * @return 端口信息，如果不存在则返回 std::nullopt
   */
  virtual std::optional<Port> getPortDetails(const std::string &port_name) = 0;

  // --- 设备连接关系管理 (高级抽象，可能依赖于底层的端口和流表操作) ---

  /**
   * @brief 创建两个设备（OVS交换机或外部设备）之间的连接
   *        这可能涉及到创建patch
   * port，或者配置物理接口/隧道接口并添加相应流表。
   * @param link 连接的描述信息
   * @return 操作是否成功
   */
  virtual bool createLink(const DeviceLink &link) = 0;

  /**
   * @brief 删除两个设备之间的连接
   * @param link_id 要删除的连接ID
   * @return 操作是否成功
   */
  virtual bool deleteLink(const std::string &link_id) = 0;

  /**
   * @brief 获取当前交换机相关的所有连接信息
   * @return 连接列表
   */
  virtual std::vector<DeviceLink> listLinks() = 0;

  // --- 其他通用操作 ---

  /**
   * @brief 获取交换机名称
   * @return 交换机名称
   */
  std::string getName() const { return name_; }

  /**
   * @brief 检查交换机是否连接/可达
   * @return 如果可达返回true，否则false
   */
  virtual bool isConnected() = 0;

protected:
  std::string name_; // OVS bridge name
  // 可以在这里添加其他通用成员，例如与OVSDB server的连接句柄等
};

// 如果需要，可以为OvsSwitch定义一个工厂类或方法，用于创建具体实现的实例
// 例如：
// class OvsSwitchFactory {
// public:
//     static std::unique_ptr<OvsSwitch> createOvsdbSwitch(const std::string&
//     bridge_name, const std::string& ovsdb_server_ip, uint16_t port); static
//     std::unique_ptr<OvsSwitch> createOfctlSwitch(const std::string&
//     bridge_name);
// };

/**
 * @brief Executes a shell command and returns its standard output and standard
 * error.
 *
 * @param cmd The command to execute.
 * @return std::string The combined stdout and stderr of the command.
 * @throws std::runtime_error if popen fails.
 */
inline std::string
execute_local_command_util(const char *cmd_with_stderr_redirect) {
  std::array<char, 256> buffer;
  std::string result;
  // std::cout << "Executing command: " << cmd_with_stderr_redirect <<
  // std::endl; // For debugging
  FILE *pipe = popen(cmd_with_stderr_redirect, "r");
  if (!pipe) {
    // std::cerr << "popen() failed for command: " << cmd_with_stderr_redirect
    // << std::endl;
    throw std::runtime_error(std::string("popen() failed for command: ") +
                             cmd_with_stderr_redirect);
  }
  try {
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
      result += buffer.data();
    }
  } catch (...) {
    pclose(pipe);
    // std::cerr << "Exception while reading pipe for command: " <<
    // cmd_with_stderr_redirect << std::endl;
    throw;
  }
  int exit_code = pclose(pipe);
  // You might want to check WIFEXITED(exit_code) and WEXITSTATUS(exit_code)
  // For simplicity, we return the output. The caller might need to inspect it
  // for errors. OVS commands often print errors but still exit with 0.
  // std::cout << "Command output: " << result << std::endl; // For debugging
  return result;
}

/**
 * @brief A simple HTTP server that executes commands received via POST
 * requests. IMPORTANT: This server is a major security risk as it executes
 * arbitrary commands. Use with extreme caution and only in trusted
 * environments. Not suitable for production without significant security
 * enhancements.
 */
class CommandExecutorHttpServer {
public:
  CommandExecutorHttpServer(std::string host = "127.0.0.1", int port = 8080)
      : svr_(), host_(std::move(host)), port_(port), is_running_(false) {
    svr_.Post("/command", [this](const httplib::Request &req,
                                 httplib::Response &res) {
      std::string command_to_execute = req.body;
      // SECURITY WARNING: Executing arbitrary commands from network is
      // dangerous! std::cout << "[Server] Received command: " <<
      // command_to_execute << std::endl;

      // Always redirect stderr to stdout to capture errors
      std::string full_command = command_to_execute + " 2>&1";

      try {
        std::string output = execute_local_command_util(full_command.c_str());
        res.set_content(output, "text/plain; charset=utf-8");
        // std::cout << "[Server] Sent output for command: " <<
        // command_to_execute << std::endl;
      } catch (const std::exception &e) {
        // std::cerr << "[Server] Error executing command '" <<
        // command_to_execute << "': " << e.what() << std::endl;
        res.status = 500; // Internal Server Error
        res.set_content(std::string("Error executing command: ") + e.what(),
                        "text/plain; charset=utf-8");
      }
    });
    // Add a simple health check endpoint
    svr_.Get("/health", [](const httplib::Request &, httplib::Response &res) {
      res.set_content("{\"status\": \"ok\"}", "application/json");
    });
  }

  ~CommandExecutorHttpServer() { stop(); }

  // Delete copy constructor and assignment operator
  CommandExecutorHttpServer(const CommandExecutorHttpServer &) = delete;
  CommandExecutorHttpServer &
  operator=(const CommandExecutorHttpServer &) = delete;

  bool start() {
    if (is_running_) {
      // std::cout << "[Server] Already running on " << host_ << ":" << port_ <<
      // std::endl;
      return true;
    }
    // Run listen in a separate thread so start() is non-blocking
    server_thread_ = std::thread([this]() {
      // std::cout << "[Server] Starting on " << host_ << ":" << port_ <<
      // std::endl;
      if (!svr_.listen(host_.c_str(), port_)) {
        // std::cerr << "[Server] Failed to start on " << host_ << ":" << port_
        // << std::endl; is_running_ remains false or set explicitly
      }
    });
    // Brief pause to allow the server to start, or use a condition variable for
    // robust check A better way would be to wait for svr_.is_running() with a
    // timeout
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    is_running_ = svr_.is_running(); // Update is_running_ after listen attempt
    return is_running_;
  }

  void stop() {
    if (svr_.is_running()) { // Check svr directly
      svr_.stop();
    }
    if (server_thread_.joinable()) {
      server_thread_.join();
    }
    is_running_ = false; // Explicitly set to false after stopping and joining
                         // std::cout << "[Server] Stopped." << std::endl;
  }

  bool isRunning() const { return is_running_ && svr_.is_running(); }

private:
  httplib::Server svr_;
  std::string host_;
  int port_;
  std::thread server_thread_;
  std::atomic<bool> is_running_;
};

/**
 * @brief Implementation of OvsSwitch that controls OVS via a RESTful command
 * executor server.
 */
class RestfulOvsSwitch : public OvsSwitch {
public:
  RestfulOvsSwitch(std::string switch_name, std::string server_host,
                   int server_port)
      : OvsSwitch(std::move(switch_name)),
        http_client_(std::make_unique<httplib::Client>(
            server_host.c_str(),
            server_port)), // Use c_str() for older httplib versions
        server_address_(server_host + ":" + std::to_string(server_port)) {
    http_client_->set_connection_timeout(5, 0); // 5 seconds connection timeout
    http_client_->set_read_timeout(15, 0);      // 15 seconds read timeout
    http_client_->set_write_timeout(15, 0);     // 15 seconds write timeout
  }

  // Helper to send command and get response text. Boolean indicates HTTP
  // success.
  std::pair<bool, std::string> sendCommand(const std::string &ovs_command) {
    // std::cout << "[Client] Sending OVS command to " << server_address_ << ":
    // " << ovs_command << std::endl;
    auto res = http_client_->Post("/command", ovs_command,
                                  "text/plain; charset=utf-8");
    if (res) {
      // std::cout << "[Client] Received HTTP status: " << res->status << " for
      // command: " << ovs_command << std::endl; std::cout << "[Client] Response
      // body: " << res->body << std::endl;
      if (res->status == 200) {
        // OVS commands often return 0 on success but might print errors to
        // stderr (now in body). A very basic check for common error patterns.
        // This is not foolproof. A more robust solution would be for the server
        // to return structured data (stdout, stderr, exit_code).
        bool command_likely_succeeded = true;
        if (res->body.find("ovs-vsctl: ") != std::string::npos &&
            (res->body.find("Error") != std::string::npos ||
             res->body.find("error") != std::string::npos ||
             res->body.find("failed") != std::string::npos)) {
          command_likely_succeeded = false;
        }
        if (res->body.find("ovs-ofctl: ") != std::string::npos &&
            (res->body.find("Error") != std::string::npos ||
             res->body.find("error") != std::string::npos ||
             res->body.find("failed") != std::string::npos)) {
          command_likely_succeeded = false;
        }
        if (res->body.find("No such file or directory") !=
            std::string::npos) { // e.g. ovs-vsctl not found
          command_likely_succeeded = false;
        }

        return {command_likely_succeeded, res->body};
      } else {
        return {false, "HTTP Error: " + std::to_string(res->status) +
                           "\nBody: " + res->body};
      }
    } else {
      auto err_code = res.error();
      std::string err_str = "Unknown error";
      // httplib::to_string is not a standard part of the library for error
      // codes in older versions Using a switch for common cases or simply
      // err_code to string
      switch (err_code) {
      case httplib::Error::Connection:
        err_str = "Connection error";
        break;
      case httplib::Error::Read:
        err_str = "Read error";
        break;
      case httplib::Error::Write:
        err_str = "Write error";
        break;
      // Add other httplib::Error enum values as needed
      default:
        err_str = "HTTP Client Error Code: " +
                  std::to_string(static_cast<int>(err_code));
        break;
      }
      // std::cerr << "[Client] HTTP Request Failed: " << err_str << " for
      // command: " << ovs_command << std::endl;
      return {false, "HTTP Request Failed: " + err_str};
    }
  }

  // --- Helper methods for command string generation ---
private:
  std::string matchFieldsToString(const MatchFields &match,
                                  bool skip_bare_table = false) const {
    std::ostringstream oss;
    bool first = true;
    auto append = [&](const std::string &s) {
      if (!first)
        oss << ",";
      oss << s;
      first = false;
    };
    auto append_val = [&](const std::string &key, const auto &val_opt) {
      if (val_opt.has_value()) {
        std::ostringstream temp_oss; // For potential hex formatting
        if constexpr (std::is_same_v<std::decay_t<decltype(val_opt.value())>,
                                     uint16_t> &&
                      (key == "dl_type" || key == "vlan_vid")) {
          temp_oss << "0x" << std::hex << val_opt.value();
        } else if constexpr (std::is_same_v<
                                 std::decay_t<decltype(val_opt.value())>,
                                 uint8_t> &&
                             (key == "vlan_pcp" || key == "nw_proto" ||
                              key == "nw_tos")) {
          temp_oss << static_cast<int>(
              val_opt.value()); // Print uint8_t as number
        } else {
          temp_oss << val_opt.value();
        }
        append(key + "=" + temp_oss.str());
      }
    };

    append_val("in_port", match.in_port);
    append_val("dl_src", match.dl_src);
    append_val("dl_dst", match.dl_dst);
    append_val("dl_type", match.dl_type);
    append_val("vlan_vid", match.vlan_vid);
    append_val("vlan_pcp", match.vlan_pcp);
    append_val("nw_src", match.nw_src);
    append_val("nw_dst", match.nw_dst);
    append_val("nw_proto", match.nw_proto);
    append_val("nw_tos", match.nw_tos);
    append_val("tp_src", match.tp_src);
    append_val("tp_dst", match.tp_dst);
    return oss.str();
  }

  std::string actionsToString(const std::vector<Action> &actions) const {
    if (actions.empty()) {
      return "drop";
    }
    std::ostringstream oss;
    for (size_t i = 0; i < actions.size(); ++i) {
      const auto &action = actions[i];
      switch (action.type) {
      case Action::Type::OUTPUT:
        oss << "output:" << action.params.at("port");
        break;
      case Action::Type::CONTROLLER:
        oss << "controller";
        if (action.params.count("controller_id"))
          oss << ":" << action.params.at("controller_id");
        break;
      case Action::Type::DROP:
        oss << "drop";
        break;
      case Action::Type::NORMAL:
        oss << "normal";
        break;
      case Action::Type::FLOOD:
        oss << "flood";
        break;
      case Action::Type::ALL:
        oss << "all";
        break;
      case Action::Type::IN_PORT:
        oss << "in_port";
        break;
      case Action::Type::SET_FIELD:
        if (action.params.count("value") && action.params.count("field")) {
          oss << "set_field:" << action.params.at("value") << "->"
              << action.params.at("field");
        } else {
          // Log error or handle missing params
        }
        break;
      default:
        break;
      }
      if (i < actions.size() - 1)
        oss << ",";
    }
    std::string result_str = oss.str();
    if (result_str.empty() &&
        !actions.empty()) { // Should not happen if actions is not empty and
                            // cases are handled
      return "drop";        // Fallback
    }
    return result_str;
  }

public:
  // --- OvsSwitch API Implementations ---

  bool addFlow(const FlowEntry &entry) override {
    std::ostringstream cmd_oss;
    cmd_oss << "ovs-ofctl add-flow " << name_ << " \"";
    cmd_oss << "table=" << entry.table_id;
    cmd_oss << ",priority=" << entry.priority;
    if (entry.cookie.has_value()) {
      cmd_oss << ",cookie=" << entry.cookie.value();
    }

    std::string match_str = matchFieldsToString(entry.match);
    if (!match_str.empty()) {
      cmd_oss << "," << match_str;
    }

    std::string actions_str = actionsToString(entry.actions);
    // An OpenFlow flow rule must have at least one action.
    // If actions_str is empty (e.g. because actions vector was empty, or
    // tostring returned empty) default to drop, or ensure actionsToString
    // always provides a valid action string (like "drop").
    if (actions_str.empty()) {
      cmd_oss << ",actions=drop";
    } else {
      cmd_oss << ",actions=" << actions_str;
    }
    cmd_oss << "\"";

    if (entry.idle_timeout.has_value())
      cmd_oss << " idle_timeout=" << entry.idle_timeout.value();
    if (entry.hard_timeout.has_value())
      cmd_oss << " hard_timeout=" << entry.hard_timeout.value();

    auto result = sendCommand(cmd_oss.str());
    return result.first;
  }

  bool modifyFlow(const FlowEntry &entry) override {
    std::ostringstream cmd_oss;
    cmd_oss << "ovs-ofctl mod-flows " << name_
            << " \""; // Use mod-flows (strict is off by default)
    cmd_oss << "table=" << entry.table_id;
    cmd_oss << ",priority=" << entry.priority;
    if (entry.cookie.has_value()) {
      cmd_oss << ",cookie=" << entry.cookie.value();
    }

    std::string match_str = matchFieldsToString(entry.match);
    if (!match_str.empty()) {
      cmd_oss << "," << match_str;
    }

    std::string actions_str = actionsToString(entry.actions);
    if (actions_str.empty()) { // mod-flows also requires actions
      cmd_oss << ",actions=drop";
    } else {
      cmd_oss << ",actions=" << actions_str;
    }
    cmd_oss << "\"";

    if (entry.idle_timeout.has_value())
      cmd_oss << " idle_timeout=" << entry.idle_timeout.value();
    if (entry.hard_timeout.has_value())
      cmd_oss << " hard_timeout=" << entry.hard_timeout.value();

    auto result = sendCommand(cmd_oss.str());
    return result.first;
  }

  bool deleteFlow(const MatchFields &match,
                  std::optional<uint32_t> table_id = std::nullopt,
                  bool strict = false) override {
    std::ostringstream cmd_oss;
    cmd_oss << "ovs-ofctl del-flows " << name_ << " \"";
    bool first_field = true;
    if (table_id.has_value()) {
      cmd_oss << "table=" << table_id.value();
      first_field = false;
    }

    std::string match_str = matchFieldsToString(match, true);
    if (!match_str.empty()) {
      if (!first_field)
        cmd_oss << ",";
      cmd_oss << match_str;
    } else if (first_field) {
      // Deleting all flows in a specific table (if table_id is set) or all
      // flows (if no table_id and no match) If match_str is empty and table_id
      // is not set, this command becomes "ovs-ofctl del-flows br_name """ which
      // is equivalent to "ovs-ofctl del-flows br_name" - delete all flows. This
      // is fine. If match_str is empty but table_id is set, it means delete all
      // flows in that table.
    }
    cmd_oss << "\"";
    if (strict)
      cmd_oss << " --strict";

    auto result = sendCommand(cmd_oss.str());
    return result.first;
  }

  bool deleteAllFlows() override {
    std::string cmd = "ovs-ofctl del-flows " + name_;
    auto result = sendCommand(cmd);
    return result.first;
  }

  std::vector<FlowEntry>
  dumpFlows(std::optional<MatchFields> match_filter = std::nullopt,
            std::optional<uint32_t> table_id_filter = std::nullopt) override {
    std::string cmd = "ovs-ofctl dump-flows " + name_;
    std::string filter_str;
    if (table_id_filter.has_value()) {
      filter_str += "table=" + std::to_string(table_id_filter.value());
    }
    if (match_filter.has_value()) {
      std::string match_s = matchFieldsToString(match_filter.value());
      if (!match_s.empty()) {
        if (!filter_str.empty())
          filter_str += ",";
        filter_str += match_s;
      }
    }
    if (!filter_str.empty()) {
      cmd += " \"" + filter_str + "\"";
    }

    auto result = sendCommand(cmd);
    std::vector<FlowEntry> flows;
    if (result.first && !result.second.empty()) {
      // TODO: Implement robust parsing of ovs-ofctl dump-flows output.
      // This is a complex task. For now, returning empty.
      // Example line: cookie=0x0, duration=123.45s, table=0, n_packets=10,
      // n_bytes=1000, priority=100,tcp,nw_src=1.2.3.4 actions=output:1
    }
    return flows;
  }

  bool addPort(const Port &port) override {
    std::string cmd =
        "ovs-vsctl --may-exist add-port " + name_ + " " + port.name;
    auto result = sendCommand(cmd);
    // Do not return early on false for add-port, as port might exist.
    // Subsequent set commands are more critical for success if port is new.
    bool overall_success = result.first;

    if (!port.type.empty() || !port.options.empty()) {
      std::ostringstream interface_cmd_oss;
      interface_cmd_oss << "ovs-vsctl --if-exists set interface "
                        << port.name; // Use --if-exists for set too
      if (!port.type.empty()) {
        interface_cmd_oss << " type=" << port.type;
      }
      for (const auto &opt : port.options) {
        interface_cmd_oss << " options:" << opt.first << "=\"" << opt.second
                          << "\"";
      }
      result = sendCommand(interface_cmd_oss.str());
      if (!result.first)
        overall_success = false;
    }
    return overall_success;
  }

  bool deletePort(const std::string &port_name) override {
    std::string cmd =
        "ovs-vsctl --if-exists del-port " + name_ + " " + port_name;
    auto result = sendCommand(cmd);
    return result.first;
  }

  bool
  modifyPort(const std::string &port_name,
             const std::map<std::string, std::string> &attributes) override {
    if (attributes.empty())
      return true;
    std::ostringstream cmd_oss;
    cmd_oss << "ovs-vsctl --if-exists set interface "
            << port_name; // Use --if-exists
    for (const auto &attr : attributes) {
      cmd_oss << " " << attr.first << "=\"" << attr.second << "\"";
    }
    auto result = sendCommand(cmd_oss.str());
    return result.first;
  }

  std::vector<Port> listPorts() override {
    std::string cmd = "ovs-vsctl list-ports " + name_;
    auto result = sendCommand(cmd);
    std::vector<Port> ports_list;
    if (result.first && !result.second.empty()) {
      std::istringstream iss(result.second);
      std::string port_name_line;
      uint32_t temp_port_no = 1;
      while (std::getline(iss, port_name_line)) {
        if (!port_name_line.empty()) {
          ports_list.emplace_back(temp_port_no++, port_name_line);
        }
      }
    }
    return ports_list;
  }

  std::optional<Port> getPortDetails(const std::string &port_name) override {
    std::string cmd = "ovs-vsctl list interface " + port_name;
    auto result = sendCommand(cmd);
    if (result.first && !result.second.empty()) {
      // TODO: Implement robust parsing of 'ovs-vsctl list interface' output.
      // Key-value pairs, map-like structures for options, etc.
      // Example:
      // name                : "port1"
      // ofport              : 1
      // type                : "internal"
      // options             : {map_key="map_val"}
      // For now, returning a basic Port object if name matches
      if (result.second.find("name                : \"" + port_name + "\"") !=
          std::string::npos) {
        uint32_t ofport = 0; // Placeholder
        std::string type;    // Placeholder
        // Try to parse ofport
        size_t ofport_pos = result.second.find("ofport              : ");
        if (ofport_pos != std::string::npos) {
          std::string ofport_str = result.second.substr(
              ofport_pos + sizeof("ofport              : ") - 1);
          ofport = std::stoul(ofport_str);
        }
        // Try to parse type
        size_t type_pos = result.second.find("type                : ");
        if (type_pos != std::string::npos) {
          std::string type_line = result.second.substr(type_pos);
          size_t type_val_pos = type_line.find('"');
          if (type_val_pos != std::string::npos) {
            type_line = type_line.substr(type_val_pos + 1);
            type = type_line.substr(0, type_line.find('"'));
          }
        }
        return Port(ofport, port_name, type);
      }
    }
    return std::nullopt;
  }

  bool createLink(const DeviceLink &link) override {
    if (link.type == DeviceLink::LinkType::PATCH &&
        link.src_device_id == name_ && link.dst_device_id == name_) {

      std::string cmd1 = "ovs-vsctl --may-exist add-port " + name_ + " " +
                         link.src_port_name + " -- --if-exists set interface " +
                         link.src_port_name +
                         " type=patch options:peer=" + link.dst_port_name;
      auto res1 = sendCommand(cmd1);
      if (!res1.first &&
          !(res1.second.find("already exists") != std::string::npos &&
            res1.second.find("may not be set as patch port peer of itself") ==
                std::string::npos)) {
        // if add-port failed for reasons other than "already exists" (and not
        // self-peer error which is specific)
        return false;
      }

      std::string cmd2 = "ovs-vsctl --may-exist add-port " + name_ + " " +
                         link.dst_port_name + " -- --if-exists set interface " +
                         link.dst_port_name +
                         " type=patch options:peer=" + link.src_port_name;
      auto res2 = sendCommand(cmd2);
      if (!res2.first &&
          !(res2.second.find("already exists") != std::string::npos &&
            res2.second.find("may not be set as patch port peer of itself") ==
                std::string::npos)) {
        // if add-port failed for reasons other than "already exists"
        // Potentially try to clean up cmd1 if it was newly created. This is
        // complex.
        return false;
      }
      // If both commands reported success (resX.first is true) or harmless
      // errors ("already exists"), consider it a success. The sendCommand's
      // heuristic for success needs to be good here.
      return true; // Simplified success criteria
    }
    return false;
  }

  bool deleteLink(const std::string &link_id) override {
    // This is highly dependent on how link_id maps to OVS constructs.
    // Assuming link_id might be one of the port names in a patch pair.
    // A more robust way would be to find the port, check if it's a patch port,
    // get its peer, and then delete both ports or clear their patch config.
    // For simplicity, if link_id is a port:
    // 1. Get port details for link_id.
    // 2. If it's a patch port, get its peer.
    // 3. Delete both link_id port and its peer port.
    // This is a placeholder:
    // std::string cmd = "ovs-vsctl --if-exists del-port " + name_ + " " +
    // link_id; auto result = sendCommand(cmd); return result.first;
    return false; // Placeholder
  }

  std::vector<DeviceLink> listLinks() override {
    // TODO: Implement by listing all ports, then for each patch port, find its
    // peer. Reconstruct DeviceLink objects. This requires parsing
    // `options:peer`.
    return {};
  }

  bool isConnected() override {
    if (http_client_) {
      // Check REST server health first
      auto health_res = http_client_->Get("/health");
      if (!health_res || health_res->status != 200) {
        return false; // Server not healthy or not reachable
      }

      // Server is up, now check OVS bridge
      std::string cmd = "ovs-vsctl br-exists " + name_;
      auto ovs_res = sendCommand(cmd);
      // `br-exists` exits with 0 if bridge exists (empty output), 2 if not
      // (prints error). `sendCommand`'s `ovs_res.first` checks for http status
      // 200 and basic errors. If `ovs_res.first` is true, we need to check
      // `ovs_res.second` for content. Empty output means bridge exists.
      // Non-empty usually means an error message.
      if (ovs_res.first) {
        return ovs_res.second.empty() ||
               ovs_res.second.find("is a bridge") !=
                   std::string::npos; // some versions might output "br_name is
                                      // a bridge"
      }
    }
    return false;
  }

private:
  std::unique_ptr<httplib::Client> http_client_;
  std::string server_address_;
};

} // namespace ovs
} // namespace netconf_model
} // namespace component
} // namespace pccl