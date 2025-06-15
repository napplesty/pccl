#include "utils/defs.h"
#include "utils/logging.h"
#include <chrono>
#include <filesystem>
#include <format>
#include <fstream>
#include <string>

using namespace pccl;

PCCL_API std::string pccl::get_start_timestamp() {
  static auto now = std::chrono::system_clock::now();
  auto current_time = std::chrono::current_zone()->to_local(now);
  return std::format("{:%Y-%m-%d-%H-%M-%S}", current_time);
};

PCCL_API void pccl::create_dir(const std::string &path) {
  if (!std::filesystem::exists(path)) {
    std::filesystem::create_directories(path);
  } else if (!std::filesystem::is_directory(path)) {
    LOG_FATAL << "Path " << path << " is not a directory";
  }
}

PCCL_API void pccl::set_affinity(int cpu_id) {
  cpu_set_t cpuset;
  if (sched_getaffinity(getpid(), sizeof(cpu_set_t), &cpuset) == -1) {
    LOG_WARNING << "Failed to set CPU affinity";
    return;
  }

  int available_cores = 0;
  for (int i = 0; i < CPU_SETSIZE; ++i) {
    if (CPU_ISSET(i, &cpuset)) {
      available_cores++;
    }
  }

  cpu_id %= available_cores;
  CPU_ZERO(&cpuset);
  CPU_SET(cpu_id, &cpuset);
}

static uint64_t get_hash(const char *string, int n) {
  uint64_t result = 4781398151ul;
  for (int c = 0; c < n; c++) {
    result = ((result << 5) + result) ^ string[c];
  }
  return result;
}

std::string get_host_name(int maxlen, const char delim) {
  std::string hostname(maxlen + 1, '\0');
  if (gethostname(const_cast<char *>(hostname.data()), maxlen) != 0) {
    throw std::runtime_error("gethostname failed");
  }
  int i = 0;
  while ((hostname[i] != delim) && (hostname[i] != '\0') && (i < maxlen - 1))
    i++;
  hostname[i] = '\0';
  return hostname.substr(0, i);
}

constexpr char HOSTID_FILE[32] = "/proc/sys/kernel/random/boot_id";

uint64_t compute_host_hash(void) {
  const size_t hashLen = 1024;
  char hostHash[hashLen];

  std::memset(hostHash, 0, hashLen);

  std::string hostName = get_host_name(hashLen, '\0');
  std::strncpy(hostHash, hostName.c_str(), hostName.size());

  if (hostName.size() < hashLen) {
    std::ifstream file(HOSTID_FILE, std::ios::binary);
    if (file.is_open()) {
      file.read(hostHash + hostName.size(), hashLen - hostName.size());
    }
  }

  hostHash[sizeof(hostHash) - 1] = '\0';
  return get_hash(hostHash, strlen(hostHash));
}

PCCL_API uint64_t pccl::host_hash(void) {
  static uint64_t hostHash = compute_host_hash();
  return hostHash;
}

PCCL_API uint64_t pccl::pid_hash(void) {
  pid_t pid = getpid();
  return get_hash(reinterpret_cast<const char *>(&pid), sizeof(pid));
}