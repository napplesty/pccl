#include "pccl_topology.h"
#include <chrono>

namespace pccl {

AdjacencyMatrix::AdjacencyMatrix(std::span<const Link> links) {
  for (const auto& link : links) {
    set(link);
  }
}

bool AdjacencyMatrix::has(uint32_t from, uint32_t to) const {
  std::lock_guard guard(lock_);
  if (auto it = data_.find(from); it != data_.end()) {
    return it->second.contains(to);
  }
  return false;
}

std::optional<Link> AdjacencyMatrix::get(uint32_t from, uint32_t to) const {
  std::lock_guard guard(lock_);
  if (auto it = data_.find(from); it != data_.end()) {
    if (auto link_it = it->second.find(to); link_it != it->second.end()) {
      return link_it->second;
    }
  }
  return std::nullopt;
}

void AdjacencyMatrix::set(const Link& link) {
  std::lock_guard guard(lock_);
  data_[link.from][link.to] = link;
}

void AdjacencyMatrix::remove(uint32_t from, uint32_t to) {
  std::lock_guard guard(lock_);
  if (auto it = data_.find(from); it != data_.end()) {
    it->second.erase(to);
    if (it->second.empty()) {
      data_.erase(it);
    }
  }
}

std::vector<Link> AdjacencyMatrix::all() const {
  std::lock_guard guard(lock_);
  std::vector<Link> result;
  for (const auto& [from, targets] : data_) {
    for (const auto& [to, link] : targets) {
      result.push_back(link);
    }
  }
  return result;
}

std::vector<Link> AdjacencyMatrix::from(uint32_t device) const {
  std::lock_guard guard(lock_);
  std::vector<Link> result;
  if (auto it = data_.find(device); it != data_.end()) {
    for (const auto& [to, link] : it->second) {
      result.push_back(link);
    }
  }
  return result;
}

std::vector<Link> AdjacencyMatrix::to(uint32_t device) const {
  std::lock_guard guard(lock_);
  std::vector<Link> result;
  for (const auto& [from, targets] : data_) {
    for (const auto& [to, link] : targets) {
      if (to == device) {
        result.push_back(link);
      }
    }
  }
  return result;
}

std::vector<Link> AdjacencyMatrix::around(uint32_t device) const {
  auto outgoing = from(device);
  auto incoming = to(device);
  outgoing.insert(outgoing.end(), incoming.begin(), incoming.end());
  return outgoing;
}

std::vector<Link> AdjacencyMatrix::by_bus(BusType bus) const {
  std::lock_guard guard(lock_);
  std::vector<Link> result;
  for (const auto& [from, targets] : data_) {
    for (const auto& [to, link] : targets) {
      if (link.bus == bus) {
        result.push_back(link);
      }
    }
  }
  return result;
}

void AdjacencyMatrix::merge(const AdjacencyMatrix& other, 
                          std::function<bool(const Link&, const Link&)> resolver) {
  std::lock_guard guard(lock_);
  auto other_links = other.all();
  
  for (const auto& link : other_links) {
    if (auto existing = get(link.from, link.to)) {
      if (resolver) {
        if (resolver(*existing, link)) {
          set(link);
        }
      } else if (link.metric.timestamp > existing->metric.timestamp) {
        set(link);
      }
    } else {
      set(link);
    }
  }
}

std::set<uint32_t> AdjacencyMatrix::devices() const {
  std::lock_guard guard(lock_);
  std::set<uint32_t> result;
  for (const auto& [from, targets] : data_) {
    result.insert(from);
    for (const auto& [to, _] : targets) {
      result.insert(to);
    }
  }
  return result;
}

void AdjacencyMatrix::clear() {
  std::lock_guard guard(lock_);
  data_.clear();
}

size_t AdjacencyMatrix::count() const {
  std::lock_guard guard(lock_);
  size_t total = 0;
  for (const auto& [_, targets] : data_) {
    total += targets.size();
  }
  return total;
}

bool AdjacencyMatrix::empty() const {
  std::lock_guard guard(lock_);
  return data_.empty();
}

void Topology::set_devices(std::span<const Device> devices) {
  std::lock_guard guard(lock_);
  device_list_.assign(devices.begin(), devices.end());
  device_map_.clear();
  for (const auto& device : devices) {
    device_map_[device.id] = device;
  }
  update_time_ = std::chrono::system_clock::now().time_since_epoch().count();
}

void Topology::set_links(std::span<const Link> links) {
  std::lock_guard guard(lock_);
  matrix_.clear();
  for (const auto& link : links) {
    matrix_.set(link);
  }
  update_time_ = std::chrono::system_clock::now().time_since_epoch().count();
}

std::vector<Device> Topology::devices() const {
  std::lock_guard guard(lock_);
  return device_list_;
}

std::vector<Link> Topology::links() const {
  std::lock_guard guard(lock_);
  return matrix_.all();
}

std::vector<Link> Topology::links_from(uint32_t device) const {
  std::lock_guard guard(lock_);
  return matrix_.from(device);
}

std::vector<Link> Topology::links_to(uint32_t device) const {
  std::lock_guard guard(lock_);
  return matrix_.to(device);
}

std::vector<Link> Topology::links_around(uint32_t device) const {
  std::lock_guard guard(lock_);
  return matrix_.around(device);
}

std::vector<Link> Topology::links_by_bus(BusType bus) const {
  std::lock_guard guard(lock_);
  return matrix_.by_bus(bus);
}

std::optional<Device> Topology::device(uint32_t id) const {
  std::lock_guard guard(lock_);
  if (auto it = device_map_.find(id); it != device_map_.end()) {
    return it->second;
  }
  return std::nullopt;
}

uint64_t Topology::updated() const {
  std::lock_guard guard(lock_);
  return update_time_;
}

const AdjacencyMatrix& Topology::matrix() const {
  std::lock_guard guard(lock_);
  return matrix_;
}

void Topology::merge(const Topology& other, 
                    std::function<bool(const Link&, const Link&)> resolver) {
  std::lock_guard guard(lock_);
  
  for (const auto& device : other.device_list_) {
    if (!device_map_.contains(device.id)) {
      device_list_.push_back(device);
      device_map_[device.id] = device;
    }
  }
  
  matrix_.merge(other.matrix_, resolver);
  update_time_ = std::chrono::system_clock::now().time_since_epoch().count();
}

} // namespace pccl