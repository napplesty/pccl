#include "utils.h"

#include <sys/syscall.h>
#include <unistd.h>

#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>

namespace pccl {

int bindToCpu() {
  static long long index = 7777;
  cpu_set_t original_mask;
  cpu_set_t target_mask;
  pid_t tid = syscall(SYS_gettid);
  if (sched_getaffinity(tid, sizeof(original_mask), &original_mask) == -1) {
    return -1;
  }

  int cpu_count = CPU_COUNT(&original_mask);
  if (cpu_count <= 0) {
    return -1;
  }

  int target_cpu_index_in_set = (int)(index % (long long)cpu_count);
  int current_cpu_index_in_set = 0;
  int target_cpu_id = -1;

  long max_cpus = sysconf(_SC_NPROCESSORS_CONF);
  if (max_cpus == -1) {
    max_cpus = CPU_SETSIZE;
  }

  for (int i = 0; i < max_cpus; ++i) {
    if (CPU_ISSET(i, &original_mask)) {
      if (current_cpu_index_in_set == target_cpu_index_in_set) {
        target_cpu_id = i;
        break;
      }
      current_cpu_index_in_set++;
    }
  }

  if (target_cpu_id == -1) {
    return -1;
  }

  CPU_ZERO(&target_mask);
  CPU_SET(target_cpu_id, &target_mask);

  if (sched_setaffinity(tid, sizeof(target_mask), &target_mask) == -1) {
    return -1;
  }

  return 0;
}

uint64_t getHash(const char* string, int n) {
  uint64_t result = 5381;
  for (int c = 0; c < n; c++) {
    result = ((result << 5) + result) ^ string[c];
  }
  return result;
}

::std::string getHostName(int maxlen, const char delim) {
  ::std::string hostname(maxlen + 1, '\0');
  if (gethostname(const_cast<char*>(hostname.data()), maxlen) != 0) {
    throw ::std::runtime_error("gethostname failed");
  }
  int i = 0;
  while ((hostname[i] != delim) && (hostname[i] != '\0') && (i < maxlen - 1))
    i++;
  hostname[i] = '\0';
  return hostname.substr(0, i);
}

constexpr char HOSTID_FILE[32] = "/proc/sys/kernel/random/boot_id";

uint64_t computeHostHash(void) {
  const size_t hashLen = 1024;
  char hostHash[hashLen];

  ::std::memset(hostHash, 0, hashLen);

  ::std::string hostName = getHostName(hashLen, '\0');
  ::std::strncpy(hostHash, hostName.c_str(), hostName.size());

  if (hostName.size() < hashLen) {
    ::std::ifstream file(HOSTID_FILE, ::std::ios::binary);
    if (file.is_open()) {
      file.read(hostHash + hostName.size(), hashLen - hostName.size());
    }
  }

  hostHash[sizeof(hostHash) - 1] = '\0';
  return getHash(hostHash, strlen(hostHash));
}

uint64_t getHostHash(void) {
  thread_local ::std::unique_ptr<uint64_t> hostHash =
      ::std::make_unique<uint64_t>(computeHostHash());
  if (hostHash == nullptr) {
    hostHash = ::std::make_unique<uint64_t>(computeHostHash());
  }
  return *hostHash;
}

uint64_t getPidHash(void) {
  pid_t pid = getpid();
  return getHash(reinterpret_cast<const char*>(&pid), sizeof(pid));
}

}  // namespace pccl
