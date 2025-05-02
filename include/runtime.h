#pragma once

#include <bitset>
#include <memory>
#include <string>
#include <vector>

#include "connection.h"
#include "context.h"
#include "device.h"
#include "executor.h"

namespace pccl {

::std::string version();

enum class ChannelType : int {
  NONE,
  MEMORY,
  PORT,
  NVLS,
  INNETWORK,
  ChannelTypeEnd,
};

enum class DeviceType : int {
  HOST,
  CUDA,
  HIP,
  DeviceTypeEnd,
};

enum class ReduceOpType : int {
  SUM,
  ReduceOpEnd,
};

enum class Transport : int {
  Unknown,
  CudaIpc,
  IB0,
  IB1,
  Ethernet,
  TransportEnd,
};

enum class OperationType : int {
  NOP,
  BARRIER,
  PUT,
  GET,
  COPY,
  SIGNAL,
  WAIT,
  FLUSH,
  REDUCE,
  REDUCE_WRITE,
  READ_REDUCE,
  MULTI_READ_REDUCE_STORE,
  NET_CONFIGURE,
  OperationTypeEnd,
};

enum class PacketType {
  LL8,
  LL16,
  Simple,
  PacketTypeEnd,
};

const ::std::string TransportNames[] = {"UNK", "IPC",     "IB0", "IB1",
                                        "ETH", "Netconf", "NUM"};

namespace detail {

constexpr int TransportFlagsSize = static_cast<int>(Transport::TransportEnd);
using TransportFlagsBase = ::std::bitset<TransportFlagsSize>;

}  // namespace detail

class TransportFlags : private detail::TransportFlagsBase {
 public:
  TransportFlags() = default;
  TransportFlags(Transport transport);
  bool has(Transport transport) const;
  bool none() const;
  bool any() const;
  bool all() const;
  size_t count() const;
  TransportFlags &operator|=(TransportFlags other);
  TransportFlags operator|(TransportFlags other) const;
  TransportFlags operator|(Transport transport) const;
  TransportFlags &operator&=(TransportFlags other);
  TransportFlags operator&(TransportFlags other) const;
  TransportFlags operator&(Transport transport) const;
  TransportFlags &operator^=(TransportFlags other);
  TransportFlags operator^(TransportFlags other) const;
  TransportFlags operator^(Transport transport) const;
  TransportFlags operator~() const;
  bool operator==(TransportFlags other) const;
  bool operator!=(TransportFlags other) const;
  detail::TransportFlagsBase toBitset() const;
  void set(size_t pos, bool value = true) {
    detail::TransportFlagsBase::set(pos, value);
  }
  bool test(size_t pos) const { return detail::TransportFlagsBase::test(pos); }

 private:
  TransportFlags(detail::TransportFlagsBase bitset);
};

inline TransportFlags operator|(Transport transport1, Transport transport2) {
  return TransportFlags(transport1) | transport2;
}
inline TransportFlags operator&(Transport transport1, Transport transport2) {
  return TransportFlags(transport1) & transport2;
}
inline TransportFlags operator^(Transport transport1, Transport transport2) {
  return TransportFlags(transport1) ^ transport2;
}

}  // namespace pccl
