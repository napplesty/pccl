#pragma once

#include <algorithm>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "runtime.h"

namespace pccl {

struct FlowTableEntry {
  std::string src_ip;
  std::string dst_ip;
  int out_port;
  int priority;

  FlowTableEntry(const std::string& src, const std::string& dst, int port, int prio = 100)
      : src_ip(src), dst_ip(dst), out_port(port), priority(prio) {}
};

struct SwitchPort {
  int port_id;
  bool is_active;

  SwitchPort(int id, bool active = true) : port_id(id), is_active(active) {}
};

struct Switch::Impl {
  int id;
  std::string name;
  std::string ip_address;
  uint16_t port;
  std::vector<SwitchPort> ports;
  std::vector<FlowTableEntry> flow_table;

  Impl(int switch_id, const std::string& switch_name, const std::string& ip, uint16_t port_num)
      : id(switch_id), name(switch_name), ip_address(ip), port(port_num) {}

  void addPort(int port_id, const std::string& port_name, bool is_active = true) {
    ports.emplace_back(port_id, port_name, is_active);
  }

  void addFlowEntry(const std::string& src_ip, const std::string& dst_ip, int out_port,
                    int priority = 100) {
    flow_table.emplace_back(src_ip, dst_ip, out_port, priority);
  }

  std::vector<int> getRouteForIPs(const std::string& src_ip, const std::string& dst_ip) const {
    std::vector<int> out_ports;
    for (const auto& entry : flow_table) {
      if ((entry.src_ip == src_ip || entry.src_ip == "*") &&
          (entry.dst_ip == dst_ip || entry.dst_ip == "*")) {
        out_ports.push_back(entry.out_port);
      }
    }
    return out_ports;
  }

  std::string_view getIpAddress() const { return ip_address; }

  uint16_t getPort() const { return port; }
};

struct Device::Impl {
  int id;
  int rank;
  std::unordered_map<TransportFlags, NetworkAddress, std::hash<TransportFlags>> network_addresses;

  Impl(int device_id, int device_rank,
       std::vector<std::tuple<TransportFlags, NetworkAddress>> endpoint_infos)
      : id(device_id), rank(device_rank) {
    for (const auto& [transport, address] : endpoint_infos) {
      network_addresses[transport] = address;
    }
  }

  NetworkAddress getAddress(TransportFlags transport) const {
    auto it = network_addresses.find(transport);
    if (it != network_addresses.end()) {
      return it->second;
    }

    NetworkAddress empty_addr{};
    return empty_addr;
  }
};

struct OpticalPortConnection {
  int port1;
  int port2;
  bool active;

  OpticalPortConnection(int p1, int p2, bool is_active = true)
      : port1(p1), port2(p2), active(is_active) {}

  bool matches(int p1, int p2) const {
    return (port1 == p1 && port2 == p2) || (port1 == p2 && port2 == p1);
  }
};

struct OpticalSwitch::Impl {
  int id;
  std::string name;
  std::string ip_address;
  uint16_t port;
  std::vector<SwitchPort> ports;
  std::vector<OpticalPortConnection> connections;

  Impl(int switch_id, const std::string& switch_name, const std::string& ip, uint16_t port_num)
      : id(switch_id), name(switch_name), ip_address(ip), port(port_num) {}

  void addPort(int port_id, const std::string& port_name = "", bool is_active = true) {
    ports.emplace_back(port_id, port_name, is_active);
  }

  bool connectPorts(int port1, int port2) {
    bool port1_exists = false;
    bool port2_exists = false;

    for (const auto& port : ports) {
      if (port.port_id == port1) port1_exists = true;
      if (port.port_id == port2) port2_exists = true;
    }

    if (!port1_exists || !port2_exists) {
      return false;
    }

    for (const auto& conn : connections) {
      if (conn.matches(port1, port2)) {
        if (!conn.active) {
          for (auto& c : connections) {
            if (c.matches(port1, port2)) {
              c.active = true;
              break;
            }
          }
          return true;
        }
        return false;
      }
    }

    connections.emplace_back(port1, port2, true);
    return true;
  }

  bool disconnectPorts(int port1, int port2) {
    for (auto& conn : connections) {
      if (conn.matches(port1, port2) && conn.active) {
        conn.active = false;
        return true;
      }
    }
    return false;
  }

  std::vector<std::tuple<int, int>> getActiveConnections() const {
    std::vector<std::tuple<int, int>> active_conns;
    for (const auto& conn : connections) {
      if (conn.active) {
        active_conns.emplace_back(conn.port1, conn.port2);
      }
    }
    return active_conns;
  }

  std::string generateConnectCommand(int port1, int port2) const {
    return "connect " + std::to_string(port1) + " " + std::to_string(port2);
  }

  std::string generateDisconnectCommand(int port1, int port2) const {
    return "disconnect " + std::to_string(port1) + " " + std::to_string(port2);
  }

  std::string_view getIpAddress() const { return ip_address; }

  uint16_t getPort() const { return port; }
};

struct Cluster::Impl {
  std::string topology_file;
  NetworkType network_type;
  std::shared_ptr<Communicator> communicator;
  std::string ip_address;
  uint16_t port;

  std::vector<std::shared_ptr<Device>> devices;
  std::vector<std::shared_ptr<Switch>> switches;
  std::vector<std::shared_ptr<OpticalSwitch>> optical_switches;

  std::unordered_map<int, std::string> route_phase_files;
  std::unordered_map<int, std::string> topology_phase_files;

  int current_route_phase;
  int current_topology_phase;

  Impl(std::string& topo_file, NetworkType net_type, std::shared_ptr<Communicator> comm,
       const std::string& ip = "127.0.0.1", uint16_t port_num = 8080)
      : topology_file(topo_file),
        network_type(net_type),
        communicator(comm),
        ip_address(ip),
        port(port_num),
        current_route_phase(0),
        current_topology_phase(0) {}

  void loadTopology() {
    // 解析拓扑文件并创建网络配置
    // 具体实现取决于文件格式
  }

  void registerDevice(int id, int rank,
                      std::vector<std::tuple<TransportFlags, NetworkAddress>> endpoint_infos) {
    devices.push_back(std::make_shared<Device>(id, rank, endpoint_infos));
  }

  void registerSwitch(int id, const std::string& name, const std::string& ip = "127.0.0.1",
                      uint16_t port_num = 6633) {
    switches.push_back(std::make_shared<Switch>(id, name, ip, port_num));
  }

  void registerOpticalSwitch(int id, const std::string& name, const std::string& ip = "127.0.0.1",
                             uint16_t port_num = 6634) {
    optical_switches.push_back(std::make_shared<OpticalSwitch>(id, name, ip, port_num));
  }

  std::vector<char> serializeDeviceConnectivity(int route_phase, int topology_phase) {
    // 创建网络连接性的二进制表示
    // 基于当前路由和拓扑配置
    std::vector<char> result;
    // 具体实现取决于所需格式
    return result;
  }

  std::string_view getIpAddress() const { return ip_address; }

  uint16_t getPort() const { return port; }
};

}  // namespace pccl