#include "pybind11/pybind11.h"
#include "runtime.h"

namespace py = pybind11;

void register_config(py::module &m) {
  py::class_<::pccl::Config>(m, "Config")
      .def_readwrite_static("DEFAULT_KERNEL_THREAD_NUM", &::pccl::Config::DEFAULT_KERNEL_THREAD_NUM)
      .def_readwrite_static("DEFAULT_PROXY_THREAD_NUM", &::pccl::Config::DEFAULT_PROXY_THREAD_NUM)
      .def_readwrite_static("DEFAULT_MEMORY_THREAD_NUM", &::pccl::Config::DEFAULT_MEMORY_THREAD_NUM)
      .def_readwrite_static("DEFAULT_FIFO_SIZE", &::pccl::Config::DEFAULT_FIFO_SIZE)
      .def_readwrite_static("DEFAULT_MAX_THREAD_BLOCK_NUM",
                            &::pccl::Config::DEFAULT_MAX_THREAD_BLOCK_NUM)
      .def_readwrite_static("DEFAULT_NUM_SYNCER", &::pccl::Config::DEFAULT_NUM_SYNCER)
      .def_readonly_static("MAX_CHANNEL", &::pccl::Config::MAX_CHANNEL)
      .def_readonly_static("MAX_CHANNEL_PER_OPERATION", &::pccl::Config::MAX_CHANNEL_PER_OPERATION)
      .def_readonly_static("MAX_OPERATION_PER_THREADBLOCK",
                           &::pccl::Config::MAX_OPERATION_PER_THREADBLOCK)
      .def_readwrite_static("MAX_DEVICE_BUFFER", &::pccl::Config::MAX_DEVICE_BUFFER)
      .def_readwrite_static("MAX_DEVICE_BUFFER_SIZE", &::pccl::Config::MAX_DEVICE_BUFFER_SIZE)
      .def_readwrite_static("MAX_HOST_BUFFER", &::pccl::Config::MAX_HOST_BUFFER)
      .def_readwrite_static("MAX_HOST_BUFFER_SIZE", &::pccl::Config::MAX_HOST_BUFFER_SIZE)
      .def_readwrite_static("DefaultMaxCqSize", &::pccl::Config::DefaultMaxCqSize)
      .def_readwrite_static("DefaultMaxCqPollNum", &::pccl::Config::DefaultMaxCqPollNum)
      .def_readwrite_static("DefaultMaxSendWr", &::pccl::Config::DefaultMaxSendWr)
      .def_readwrite_static("DefaultMaxWrPerSend", &::pccl::Config::DefaultMaxWrPerSend)
      .def_readwrite_static("MaxDataPacketSize", &::pccl::Config::MaxDataPacketSize)
      .def_readwrite_static("MaxControlPacketSize", &::pccl::Config::MaxControlPacketSize)
      .def_readwrite_static("MSCCLPP_SOCKET_MAGIC", &::pccl::Config::MSCCLPP_SOCKET_MAGIC);
}

PYBIND11_MODULE(_pccl, m) {
  m.doc() = "PCCL runtime";
  register_config(m);
}
