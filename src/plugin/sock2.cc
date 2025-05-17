#include "plugin/sock.h"

namespace pccl {

SockMrInfo SockMr::getInfo() const { return SockMrInfo{reinterpret_cast<uint64_t>(buff)}; }

void* SockMr::getBuff() const { return buff; }

bool SockMr::isHost() const { return isHostMemory; }

SockMr::SockMr(void* buff, size_t size, bool isHostMemory)
    : buff(buff), size(size), isHostMemory(isHostMemory) {}

SockMr::~SockMr() {}

}  // namespace pccl