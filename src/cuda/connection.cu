#include "cuda/connection.h"

namespace pccl {

NvlsConnection::NvlsConnection(Endpoint remote, Endpoint local)
    : Connection(remote, local) {}

NvlsConnection::~NvlsConnection() {}

} // namespace pccl