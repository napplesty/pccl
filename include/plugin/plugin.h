#pragma once

#include <memory>
#include <vector>

namespace pccl {

class RegisteredMemory {
 public:
  RegisteredMemory() = default;
  ~RegisteredMemory();
  void *data() const;
  void *originalDataPtr() const;
  size_t size();
  TransportFlags transports();
  std::vector<char> serialize();
  static RegisteredMemory deserialize(const std::vector<char> &data);

 private:
  struct Impl;
  RegisteredMemory(std::shared_ptr<Impl> pimpl);
  std::shared_ptr<Impl> pimpl_;
  friend class Context;
  friend class Connection;
};

class Endpoint {
 public:
  Endpoint() = default;
  Transport transport();
  int maxWriteQueueSize();
  std::vector<char> serialize();
  static Endpoint deserialize(const std::vector<char> &data);

 private:
  struct Impl;
  Endpoint(std::shared_ptr<Impl> pimpl);
  std::shared_ptr<Impl> pimpl_;
  friend class Context;
  friend class Connection;
};

class Plugin {
 public:
  virtual bool registerBuffer(void *ptr, size_t size, int buffer_id) = 0;
  virtual std::vector<char> pack_self() = 0;
  virtual bool unpack_remote(const ::std::vector<char> &data,
                             int remote_rank) = 0;
  virtual void connect_remote(int remote_rank) = 0;
  virtual void disconnect_remote(int remote_rank) = 0;
  virtual uint64_t bandwidth(int remote_rank) = 0;
  virtual uint64_t latency(int remote_rank) = 0;
};

class NetworkPlugin : public Plugin {
 public:
  virtual void put(void *ptr, size_t size, int remote_rank,
                   uint64_t remote_offset, int remote_buffer_id) = 0;
  virtual void get(void *ptr, size_t size, int remote_rank,
                   uint64_t remote_offset, int remote_buffer_id) = 0;
  virtual void signal(int remote_rank, int remote_buffer_id) = 0;
  virtual void
};

class MemoryPlugin : public Plugin {
 public:
  virtual void *get_
};

}  // namespace pccl
