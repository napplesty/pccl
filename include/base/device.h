#pragma once

#include <memory>
#include <base/registry.h>

namespace engine_c {

class AllocatorBase {
public:
  virtual bool allocatorAvailable();
  virtual void *allocate(long nbytes);
  virtual void deallocate(void *);
};

class RemoteCommunicatorBase {
public:
  virtual bool remoteCommAvailable();
  virtual std::string activate();
  virtual std::string regBuffer(void *addr, long size);
  virtual void unregBuffer(void *addr);
  virtual void regRemoteHandle(std::string &handle);
  virtual void connect(std::string handle);
  virtual void disconnect(std::string handle);
};

class IpcCommunicatorBase {
public:
  virtual bool IPCAvailable();
  virtual std::string registerBuffer(void *addr, long size);
  virtual std::string allocateIpcBuffer(void **addr, long size);
  virtual long mapBuffer(std::string &shareable_ShareableHandle, void **addr);
};

class DeviceBase : 
  public AllocatorBase, 
  public RemoteCommunicatorBase,
  public IpcCommunicatorBase {
public:

};

std::shared_ptr<DeviceBase> getDev(DeviceType device_type);
void regDev(DeviceType device_type, std::shared_ptr<DeviceBase> device);

}

