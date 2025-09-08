#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <set>
#include <functional>
#include <mutex>
#include <optional>
#include <span>

namespace pccl {

enum class DeviceType : uint32_t {
  Nic = 1u,
  Cpu = 1u << 1,
  Storage = 1u << 2,
  Accelerator = 1u << 3,
  Switch = 1u << 5
};

enum class BusType {
  Pcie,
  Nvlink,
  Infiniband,
  Ethernet,
  Memory
};

struct Device {
  uint32_t id;
  DeviceType type;
  uint32_t numa_node;
  std::string name;
  std::string vendor;
  std::string model;
  std::string uuid;
  std::string driver_version;

  auto operator<=>(const Device&) const = default;
};

struct LinkMetric {
  float bandwidth_gbps;
  float latency_ns;
  float utilization;
  uint32_t error_count;
  uint64_t timestamp;

  auto operator<=>(const LinkMetric&) const = default;
};

struct Link {
  uint32_t from;
  uint32_t to;
  BusType bus;
  LinkMetric metric;

  auto operator<=>(const Link&) const = default;
};

class AdjacencyMatrix {
public:
  AdjacencyMatrix() = default;
  explicit AdjacencyMatrix(std::span<const Link> links);
  
  bool has(uint32_t from, uint32_t to) const;
  std::optional<Link> get(uint32_t from, uint32_t to) const;
  void set(const Link& link);
  void remove(uint32_t from, uint32_t to);
  
  std::vector<Link> all() const;
  std::vector<Link> from(uint32_t device) const;
  std::vector<Link> to(uint32_t device) const;
  std::vector<Link> around(uint32_t device) const;
  std::vector<Link> by_bus(BusType bus) const;
  
  void merge(const AdjacencyMatrix& other, 
            std::function<bool(const Link&, const Link&)> resolver = nullptr);
  
  std::set<uint32_t> devices() const;
  void clear();
  size_t count() const;
  bool empty() const;

private:
  std::unordered_map<uint32_t, std::unordered_map<uint32_t, Link>> data_;
  mutable std::mutex lock_;
};

class Topology {
public:
  void set_devices(std::span<const Device> devices);
  void set_links(std::span<const Link> links);
  
  std::vector<Device> devices() const;
  std::vector<Link> links() const;
  std::vector<Link> links_from(uint32_t device) const;
  std::vector<Link> links_to(uint32_t device) const;
  std::vector<Link> links_around(uint32_t device) const;
  std::vector<Link> links_by_bus(BusType bus) const;
  
  std::optional<Device> device(uint32_t id) const;
  uint64_t updated() const;
  
  const AdjacencyMatrix& matrix() const;
  void merge(const Topology& other, 
            std::function<bool(const Link&, const Link&)> resolver = nullptr);

private:
  std::vector<Device> device_list_;
  std::unordered_map<uint32_t, Device> device_map_;
  AdjacencyMatrix matrix_;
  mutable std::mutex lock_;
  uint64_t update_time_{0};
};

} // namespace pccl
