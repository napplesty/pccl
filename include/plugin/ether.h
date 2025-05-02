#pragma once

#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <stddef.h>
#include <sys/socket.h>

#include <functional>
#include <string>

#include "config.h"

namespace pccl {

union SocketAddress {
  struct sockaddr sa;
  struct sockaddr_in sin;
  struct sockaddr_in6 sin6;
};

enum SocketState {
  SocketStateNone = 0,
  SocketStateInitialized = 1,
  SocketStateAccepting = 2,
  SocketStateAccepted = 3,
  SocketStateConnecting = 4,
  SocketStateConnectPolling = 5,
  SocketStateConnected = 6,
  SocketStateBound = 7,
  SocketStateReady = 8,
  SocketStateClosed = 9,
  SocketStateError = 10,
  SocketStateNum = 11
};

enum SocketType {
  SocketTypeUnknown = 0,
  SocketTypeBootstrap = 1,
  SocketTypeProxy = 2,
};

class Socket {
 public:
  Socket(const SocketAddress *addr = nullptr,
         uint64_t magic = MSCCLPP_SOCKET_MAGIC,
         enum SocketType type = SocketTypeUnknown,
         volatile uint32_t *abortFlag = nullptr, int asyncFlag = 0);
  ~Socket();

  void bind();
  void bindAndListen();
  void connect(int64_t timeout = -1);
  void accept(const Socket *listenSocket, int64_t timeout = -1);
  void send(void *ptr, int size);
  void recv(void *ptr, int size);
  void recvUntilEnd(void *ptr, int size, int *closed);
  void close();

  int getFd() const { return fd_; }
  int getAcceptFd() const { return acceptFd_; }
  int getConnectRetries() const { return connectRetries_; }
  int getAcceptRetries() const { return acceptRetries_; }
  volatile uint32_t *getAbortFlag() const { return abortFlag_; }
  int getAsyncFlag() const { return asyncFlag_; }
  enum SocketState getState() const { return state_; }
  uint64_t getMagic() const { return magic_; }
  enum SocketType getType() const { return type_; }
  SocketAddress getAddr() const { return addr_; }
  int getSalen() const { return salen_; }

 private:
  void tryAccept();
  void finalizeAccept();
  void startConnect();
  void pollConnect();
  void finalizeConnect();
  void progressState();

  void socketProgressOpt(int op, void *ptr, int size, int *offset, int block,
                         int *closed);
  void socketProgress(int op, void *ptr, int size, int *offset);
  void socketWait(int op, void *ptr, int size, int *offset);

  int fd_;
  int acceptFd_;
  int connectRetries_;
  int acceptRetries_;
  volatile uint32_t *abortFlag_;
  int asyncFlag_;
  enum SocketState state_;
  uint64_t magic_;
  enum SocketType type_;

  union SocketAddress addr_;
  int salen_;
};

}  // namespace pccl
