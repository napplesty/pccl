#include <c10/util/Exception.h>
#include <plugins/cpu_executor/device.h>
#include <nlohmann/json.hpp>
#include <string_view>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <chrono>
#include <random>
#include <cstring>

namespace engine_c {

struct TMP_STATIC_INITIALIZER {
  TMP_STATIC_INITIALIZER() {
    auto device = TypeRegistry::registerDeviceType("cpu");
    regDev(device, std::make_shared<DeviceBase>(new CpuDevice));
  }
};

static TMP_STATIC_INITIALIZER _____tmp;

struct MemInfo {
  std::string filename;
  long size;
  int fd;

  std::string serialize() {
    nlohmann::json j;
    j["filename"] = filename;
    j["size"] = size;
    return j.dump();
  }

  static MemInfo deserialize(std::string_view mem_info) {
    auto j = nlohmann::json::parse(mem_info);
    MemInfo r;
    r.filename = j["filename"];
    r.size = j["size"];
    r.fd = shm_open(r.filename.c_str(), O_RDWR, 0666);
    TORCH_CHECK(r.fd != -1);

    return r;
  }
};

static std::unordered_map<void*, MemInfo> mem_map;
static std::mutex mem_mutex;

std::string gen_shm_name() {
  auto ts = std::chrono::system_clock::now().time_since_epoch().count();
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(1, 1000000);
  return std::format("/engine_c.{}.{}.txt", \
    std::to_string(ts), 
    std::to_string(dis(gen)));
}

bool CpuDevice::allocatorAvailable() {
  return true;
}

void* CpuDevice::allocate(long nbytes) {
  if (nbytes <= 0) return nullptr;

  std::string filename = gen_shm_name();
  int shm_fd = shm_open(filename.c_str(), O_CREAT | O_RDWR, 0666);
  TORCH_CHECK(shm_fd != -1);

  TORCH_CHECK(ftruncate(shm_fd, nbytes) != -1);

  void* ptr = mmap(nullptr, nbytes, PROT_READ | PROT_WRITE, \
      MAP_SHARED, shm_fd, 0);
  TORCH_CHECK(ptr != MAP_FAILED);

  std::lock_guard<std::mutex> lock(mem_mutex);
  mem_map[ptr] = {filename,nbytes,shm_fd};
  return ptr;
}

void CpuDevice::deallocate(void *ptr) {
  if (!ptr) return;
  std::lock_guard<std::mutex> lock(mem_mutex);
  auto it = mem_map.find(ptr);
  if (it != mem_map.end()) {
    munmap(ptr, it->second.size);
    if (it->second.fd != -1) {
      close(it->second.fd);
    }
    shm_unlink(it->second.filename.c_str());
    mem_map.erase(it);
  }
}

bool CpuDevice::IPCAvailable() {
  return true;
}

std::string CpuDevice::allocateIpcBuffer(void **addr, long size) {
  *addr = allocate(size);
  MemInfo &mem = mem_map[*addr];
  return mem.serialize();
}

long CpuDevice::mapBuffer(std::string &shareable_handle, void **addr) {
  MemInfo mem_info = MemInfo::deserialize(shareable_handle);
  void* ptr = mmap(nullptr, mem_info.size, PROT_READ | PROT_WRITE,
                   MAP_SHARED, mem_info.fd, 0);
  TORCH_CHECK(ptr != MAP_FAILED);

  *addr = ptr;
  std::lock_guard<std::mutex> lock(mem_mutex);
  mem_map[ptr] = mem_info;
  return mem_info.size;
}

}
