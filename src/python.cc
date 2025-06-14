#include "pybind11/operators.h"
#include "pybind11/pybind11.h"
#include "runtime.h"

namespace py = pybind11;

void register_config(py::module &m) {
  py::class_<::pccl::Config>(m, "Config")
      .def_readonly_static("WARP_SIZE", &::pccl::Config::WARP_SIZE)
      .def_readonly_static("WARP_PER_SM", &::pccl::Config::WARP_PER_SM)
      .def_readonly_static("WARP_FOR_SCHEDULE",
                           &::pccl::Config::WARP_FOR_SCHEDULE)
      .def_readonly_static("WARP_FOR_PROXY", &::pccl::Config::WARP_FOR_PROXY)
      .def_readonly_static("WARP_FOR_MEMORY", &::pccl::Config::WARP_FOR_MEMORY)
      .def_readonly_static("MAX_SM_COUNT", &::pccl::Config::MAX_SM_COUNT)
      .def_readonly_static("DEVICE_SYNCER_SIZE",
                           &::pccl::Config::DEVICE_SYNCER_SIZE)
      .def_readonly_static("MAX_OPERATIONS_PER_CAPSULE",
                           &::pccl::Config::MAX_OPERATIONS_PER_CAPSULE)
      .def_readonly_static("FIFO_BUFFER_SIZE",
                           &::pccl::Config::FIFO_BUFFER_SIZE)
      .def_readonly_static("INTER_SM_FIFO_SIZE",
                           &::pccl::Config::INTER_SM_FIFO_SIZE)
      .def_readonly_static("MAX_CHANNEL_PER_OPERATION",
                           &::pccl::Config::MAX_CHANNEL_PER_OPERATION)
      .def_readonly_static("MAX_ACTIVE_CONNECTIONS",
                           &::pccl::Config::MAX_ACTIVE_CONNECTIONS)
      .def_readonly_static("LIB_BUFFER_SIZE", &::pccl::Config::LIB_BUFFER_SIZE)
      .def_readonly_static("DEVICE_BUFFER_SIZE",
                           &::pccl::Config::DEVICE_BUFFER_SIZE)
      .def_readonly_static("HOST_BUFFER_SIZE",
                           &::pccl::Config::HOST_BUFFER_SIZE)
      .def_readonly_static("NUM_SLOT", &::pccl::Config::NUM_SLOT)
      .def_readonly_static("MAX_CQ_SIZE", &::pccl::Config::MAX_CQ_SIZE)
      .def_readonly_static("MAX_CQ_POLL_NUM", &::pccl::Config::MAX_CQ_POLL_NUM)
      .def_readonly_static("MAX_SEND_WR", &::pccl::Config::MAX_SEND_WR)
      .def_readonly_static("MAX_WR_PER_SEND", &::pccl::Config::MAX_WR_PER_SEND)
      .def_readonly_static("PROXY_FLUSH_PERIOD",
                           &::pccl::Config::PROXY_FLUSH_PERIOD)
      .def_readonly_static("PROXY_CHECK_STOP_PERIOD",
                           &::pccl::Config::PROXY_CHECK_STOP_PERIOD);

  py::class_<::pccl::env>(m, "Env")
      .def_readwrite("rank", &::pccl::env::rank)
      .def_readwrite("localRank", &::pccl::env::localRank)
      .def_readwrite("worldSize", &::pccl::env::worldSize)
      .def_readwrite("socketFamily", &::pccl::env::socketFamily)
      .def_readwrite("socketAddr0", &::pccl::env::socketAddr0)
      .def_readwrite("socketPort0", &::pccl::env::socketPort0)
      .def_readwrite("socketAddr1", &::pccl::env::socketAddr1)
      .def_readwrite("socketPort1", &::pccl::env::socketPort1)
      .def_readwrite("ibSocketFamily", &::pccl::env::ibSocketFamily)
      .def_readwrite("ibDevice0", &::pccl::env::ibDevice0)
      .def_readwrite("ibDevice1", &::pccl::env::ibDevice1)
      .def_readwrite("ibPort0", &::pccl::env::ibPort0)
      .def_readwrite("ibPort1", &::pccl::env::ibPort1)
      .def_readwrite("netConfFile", &::pccl::env::netConfFile)
      .def_readwrite("netConfAddr", &::pccl::env::netConfAddr)
      .def_readwrite("netConfPort", &::pccl::env::netConfPort)
      .def_readwrite("netConfModel", &::pccl::env::netConfModel)
      .def_readwrite("profileDir", &::pccl::env::profileDir)
      .def_readwrite("enableTransportList", &::pccl::env::enableTransportList)
      .def("__repr__", [](const ::pccl::env &e) {
        return "<Env rank=" + std::to_string(e.rank) + ">";
      });

  m.def("get_env", &::pccl::getEnv, "Get PCCL environment configuration");
}

void register_runtime(py::module &m) {
  m.def("version", &::pccl::version);

  py::class_<::pccl::TransportFlags>(m, "TransportFlags")
      .def(py::init<::pccl::Transport>())
      .def("has", &::pccl::TransportFlags::has)
      .def("none", &::pccl::TransportFlags::none)
      .def("any", &::pccl::TransportFlags::any)
      .def("all", &::pccl::TransportFlags::all)
      .def("count", &::pccl::TransportFlags::count)
      .def(py::self | py::self)
      .def(py::self | ::pccl::Transport())
      .def(py::self & py::self)
      .def(py::self & ::pccl::Transport())
      .def(py::self ^ py::self)
      .def(py::self ^ ::pccl::Transport())
      .def(~py::self)
      .def("__eq__", &::pccl::TransportFlags::operator==)
      .def("__ne__", &::pccl::TransportFlags::operator!=)
      .def_static("from_string", &::pccl::TransportFlags::fromString)
      .def("to_string", &::pccl::TransportFlags::toString)
      .def("__repr__", [](const ::pccl::TransportFlags &flags) {
        return "<TransportFlags bits=" + flags.toBitset().to_string() + ">";
      });

  // py::class_<::pccl::RegisteredMemory>(m, "RegisteredMemory")
  //     .def("rank_of", &::pccl::RegisteredMemory::rankOf)
  //     .def_property_readonly("host_ptr",
  //                            [](const ::pccl::RegisteredMemory &m) {
  //                              return
  //                              reinterpret_cast<uintptr_t>(m.hostPtr());
  //                            })
  //     .def_property_readonly("device_ptr",
  //                            [](const ::pccl::RegisteredMemory &m) {
  //                              return reinterpret_cast<uintptr_t>(
  //                                  m.devicePtr());
  //                            })
  //     .def_property_readonly("size", &::pccl::RegisteredMemory::size)
  //     .def_property_readonly("type", &::pccl::RegisteredMemory::type)
  //     .def_property_readonly("transports",
  //                            &::pccl::RegisteredMemory::transports)
  //     .def_property_readonly("tag", &::pccl::RegisteredMemory::tag)
  //     .def("serialize", &::pccl::RegisteredMemory::serialize)
  //     .def_static("deserialize", &::pccl::RegisteredMemory::deserialize)
  //     .def("__repr__", [](const ::pccl::RegisteredMemory &m) {
  //       return "<RegisteredMemory rank=" + std::to_string(m.rankOf()) +
  //              " size=" + std::to_string(m.size()) +
  //              " type=" + std::to_string(static_cast<int>(m.type())) + ">";
  //     });
}

void register_enums(py::module &m) {

  py::enum_<::pccl::Transport>(m, "Transport")
      .value("Unknown", ::pccl::Transport::Unknown)
      .value("HostIpc", ::pccl::Transport::HostIpc)
      .value("CudaIpc", ::pccl::Transport::CudaIpc)
      .value("IB", ::pccl::Transport::IB)
      .value("Ethernet", ::pccl::Transport::Ethernet)
      .value("NVLS", ::pccl::Transport::NVLS)
      .export_values();

  py::enum_<::pccl::BufferType>(m, "BufferType")
      .value("LIB", ::pccl::BufferType::LIB)
      .value("DEVICE", ::pccl::BufferType::DEVICE)
      .value("HOST", ::pccl::BufferType::HOST)
      .value("TEMP", ::pccl::BufferType::TEMP)
      .export_values();

  // ChannelType 枚举
  py::enum_<::pccl::ChannelType>(m, "ChannelType")
      .value("NONE", ::pccl::ChannelType::NONE)
      .value("MEMORY", ::pccl::ChannelType::MEMORY)
      .value("PORT", ::pccl::ChannelType::PORT)
      .value("NVLS", ::pccl::ChannelType::NVLS)
      .value("INNETWORK", ::pccl::ChannelType::INNETWORK)
      .export_values();

  // DeviceType 枚举
  py::enum_<::pccl::DeviceType>(m, "DeviceType")
      .value("HOST", ::pccl::DeviceType::HOST)
      .value("CUDA", ::pccl::DeviceType::CUDA)
      .value("HIP", ::pccl::DeviceType::HIP)
      .export_values();

  // NetworkType 枚举
  py::enum_<::pccl::NetworkType>(m, "NetworkType")
      .value("OVS_FLOW", ::pccl::NetworkType::OVS_FLOW)
      .export_values();

  // DataType 枚举
  py::enum_<::pccl::DataType>(m, "DataType")
      .value("I8", ::pccl::DataType::I8)
      .value("I16", ::pccl::DataType::I16)
      .value("I32", ::pccl::DataType::I32)
      .value("I64", ::pccl::DataType::I64)
      .value("U8", ::pccl::DataType::U8)
      .value("U16", ::pccl::DataType::U16)
      .value("U32", ::pccl::DataType::U32)
      .value("U64", ::pccl::DataType::U64)
      .value("FP16", ::pccl::DataType::FP16)
      .value("FP32", ::pccl::DataType::FP32)
      .value("BF16", ::pccl::DataType::BF16)
      .value("FP8_E4M3", ::pccl::DataType::FP8_E4M3)
      .value("FP8_E5M2", ::pccl::DataType::FP8_E5M2)
      .export_values();

  // ReduceOpType 枚举
  py::enum_<::pccl::ReduceOpType>(m, "ReduceOpType")
      .value("SUM", ::pccl::ReduceOpType::SUM)
      .export_values();

  // OperationType 枚举
  py::enum_<::pccl::OperationType>(m, "OperationType")
      .value("NOP", ::pccl::OperationType::NOP)
      .value("BARRIER", ::pccl::OperationType::BARRIER)
      .value("PUT", ::pccl::OperationType::PUT)
      .value("GET", ::pccl::OperationType::GET)
      .value("SIGNAL", ::pccl::OperationType::SIGNAL)
      .value("WAIT", ::pccl::OperationType::WAIT)
      .value("FLUSH", ::pccl::OperationType::FLUSH)
      .value("REDUCE", ::pccl::OperationType::REDUCE)
      .value("REDUCE_WRITE", ::pccl::OperationType::REDUCE_WRITE)
      .value("READ_REDUCE", ::pccl::OperationType::READ_REDUCE)
      .value("MULTI_READ_REDUCE_STORE",
             ::pccl::OperationType::MULTI_READ_REDUCE_STORE)
      .value("NETCONF", ::pccl::OperationType::NETCONF)
      .export_values();

  // PacketType 枚举
  py::enum_<::pccl::PacketType>(m, "PacketType")
      .value("SIMPLE", ::pccl::PacketType::Simple)
      .value("LL16", ::pccl::PacketType::LL16)
      .export_values();
}

PYBIND11_MODULE(_pccl, m) {
  m.doc() = "PCCL runtime";
  register_config(m);
  register_enums(m);
  register_runtime(m);
}
