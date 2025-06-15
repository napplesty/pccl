#include "config.h"
#include "pybind11/pybind11.h"
#include "types.h"
#include "utils.h"
#include <stdexcept>

namespace pccl {

template <size_t N, uint64_t id> class FlagsWrapper {
public:
  constexpr static uint64_t class_id = id;
  std::bitset<N> flags;
  FlagsWrapper() = default;
  FlagsWrapper(const std::string &str) {
    if (str.size() >= 2 && str.compare(0, 2, "0b") == 0) {
      if (str.size() > N + 2) {
        throw std::invalid_argument("string too long");
      }
      flags = std::bitset<N>(str.substr(2));
    } else {
      if (str.size() > N) {
        throw std::invalid_argument("string too long");
      }
      flags = std::bitset<N>(str);
    }
  }

  std::string to_string() const { return "0b" + flags.to_string(); }

  FlagsWrapper operator&(const FlagsWrapper &other) const {
    FlagsWrapper result;
    result.flags = this->flags & other.flags;
    return result;
  }

  FlagsWrapper operator|(const FlagsWrapper &other) const {
    FlagsWrapper result;
    result.flags = this->flags | other.flags;
    return result;
  }

  FlagsWrapper operator^(const FlagsWrapper &other) const {
    FlagsWrapper result;
    result.flags = this->flags ^ other.flags;
    return result;
  }

  FlagsWrapper operator~() const {
    FlagsWrapper result;
    result.flags = ~this->flags;
    return result;
  }

  bool operator==(const FlagsWrapper &other) const {
    return this->flags == other.flags;
  }

  bool operator!=(const FlagsWrapper &other) const {
    return this->flags != other.flags;
  }

  void set(size_t pos, bool val = true) { flags.set(pos, val); }
  bool test(size_t pos) const { return flags.test(pos); }
  constexpr size_t size() const { return N; }
};

} // namespace pccl

namespace py = pybind11;
using namespace pccl;

void register_config(py::module &m) {
  py::class_<Config>(m, "Config")
      .def_readwrite_static("DEVICE_BUFFER_SIZE", &Config::DEVICE_BUFFER_SIZE)
      .def_readwrite_static("HOST_BUFFER_SIZE", &Config::HOST_BUFFER_SIZE)
      .def_readwrite_static("SLOT_GRANULARITY", &Config::SLOT_GRANULARITY)
      .def_readwrite_static("PROXY_FLUSH_PERIOD", &Config::PROXY_FLUSH_PERIOD)
      .def_readwrite_static("PROXY_MAX_FLUSH_SIZE",
                            &Config::PROXY_MAX_FLUSH_SIZE)
      .def_readwrite_static("PROXY_CHECK_STOP_PERIOD",
                            &Config::PROXY_CHECK_STOP_PERIOD);
}

void register_types(py::module &m) {
  py::class_<FlagsWrapper<(size_t)FunctionType::FunctionTypeEnd, 0>>(
      m, "FunctionTypeFlags")
      .def(py::init<>())
      .def(py::init<const std::string &>())
      .def("__and__",
           &FlagsWrapper<(size_t)FunctionType::FunctionTypeEnd, 0>::operator&)
      .def("__or__",
           &FlagsWrapper<(size_t)FunctionType::FunctionTypeEnd, 0>::operator|)
      .def("__xor__",
           &FlagsWrapper<(size_t)FunctionType::FunctionTypeEnd, 0>::operator^)
      .def("__invert__",
           &FlagsWrapper<(size_t)FunctionType::FunctionTypeEnd, 0>::operator~)
      .def(
          "__eq__",
          [](const FlagsWrapper<(size_t)FunctionType::FunctionTypeEnd, 0> &self,
             const FlagsWrapper<(size_t)FunctionType::FunctionTypeEnd, 0>
                 &other) { return self.flags == other.flags; })
      .def(
          "__ne__",
          [](const FlagsWrapper<(size_t)FunctionType::FunctionTypeEnd, 0> &self,
             const FlagsWrapper<(size_t)FunctionType::FunctionTypeEnd, 0>
                 &other) { return self.flags != other.flags; })
      .def("__str__",
           &FlagsWrapper<(size_t)FunctionType::FunctionTypeEnd, 0>::to_string)
      .def("__repr__",
           &FlagsWrapper<(size_t)FunctionType::FunctionTypeEnd, 0>::to_string)
      .def("set", &FlagsWrapper<(size_t)FunctionType::FunctionTypeEnd, 0>::set)
      .def("test",
           &FlagsWrapper<(size_t)FunctionType::FunctionTypeEnd, 0>::test)
      .def("size",
           &FlagsWrapper<(size_t)FunctionType::FunctionTypeEnd, 0>::size);

  py::class_<
      FlagsWrapper<(size_t)BasicOperationType::BasicOperationTypeEnd, 1>>(
      m, "OperationTypeFlags")
      .def(py::init<>())
      .def(py::init<const std::string &>())
      .def("__and__",
           &FlagsWrapper<(size_t)BasicOperationType::BasicOperationTypeEnd,
                         1>::operator&)
      .def("__or__",
           &FlagsWrapper<(size_t)BasicOperationType::BasicOperationTypeEnd,
                         1>::operator|)
      .def("__xor__",
           &FlagsWrapper<(size_t)BasicOperationType::BasicOperationTypeEnd,
                         1>::operator^)
      .def("__invert__",
           &FlagsWrapper<(size_t)BasicOperationType::BasicOperationTypeEnd,
                         1>::operator~)
      .def(
          "__eq__",
          [](const FlagsWrapper<
                 (size_t)BasicOperationType::BasicOperationTypeEnd, 1> &self,
             const FlagsWrapper<
                 (size_t)BasicOperationType::BasicOperationTypeEnd, 1> &other) {
            return self.flags == other.flags;
          })
      .def(
          "__ne__",
          [](const FlagsWrapper<
                 (size_t)BasicOperationType::BasicOperationTypeEnd, 1> &self,
             const FlagsWrapper<
                 (size_t)BasicOperationType::BasicOperationTypeEnd, 1> &other) {
            return self.flags != other.flags;
          })
      .def("__str__",
           &FlagsWrapper<(size_t)BasicOperationType::BasicOperationTypeEnd,
                         1>::to_string)
      .def("__repr__",
           &FlagsWrapper<(size_t)BasicOperationType::BasicOperationTypeEnd,
                         1>::to_string)
      .def("set",
           &FlagsWrapper<(size_t)BasicOperationType::BasicOperationTypeEnd,
                         1>::set)
      .def("test",
           &FlagsWrapper<(size_t)BasicOperationType::BasicOperationTypeEnd,
                         1>::test)
      .def("size",
           &FlagsWrapper<(size_t)BasicOperationType::BasicOperationTypeEnd,
                         1>::size);

  py::class_<FlagsWrapper<(size_t)ComponentType::ComponentTypeEnd, 2>>(
      m, "ComponentTypeFlags")
      .def(py::init<>())
      .def(py::init<const std::string &>())
      .def("__and__",
           &FlagsWrapper<(size_t)ComponentType::ComponentTypeEnd, 2>::operator&)
      .def("__or__",
           &FlagsWrapper<(size_t)ComponentType::ComponentTypeEnd, 2>::operator|)
      .def("__xor__",
           &FlagsWrapper<(size_t)ComponentType::ComponentTypeEnd, 2>::operator^)
      .def("__invert__",
           &FlagsWrapper<(size_t)ComponentType::ComponentTypeEnd, 2>::operator~)
      .def("__eq__",
           [](const FlagsWrapper<(size_t)ComponentType::ComponentTypeEnd, 2>
                  &self,
              const FlagsWrapper<(size_t)ComponentType::ComponentTypeEnd, 2>
                  &other) { return self.flags == other.flags; })
      .def("__ne__",
           [](const FlagsWrapper<(size_t)ComponentType::ComponentTypeEnd, 2>
                  &self,
              const FlagsWrapper<(size_t)ComponentType::ComponentTypeEnd, 2>
                  &other) { return self.flags != other.flags; })
      .def("__str__",
           &FlagsWrapper<(size_t)ComponentType::ComponentTypeEnd, 2>::to_string)
      .def("__repr__",
           &FlagsWrapper<(size_t)ComponentType::ComponentTypeEnd, 2>::to_string)
      .def("set",
           &FlagsWrapper<(size_t)ComponentType::ComponentTypeEnd, 2>::set)
      .def("test",
           &FlagsWrapper<(size_t)ComponentType::ComponentTypeEnd, 2>::test)
      .def("size",
           &FlagsWrapper<(size_t)ComponentType::ComponentTypeEnd, 2>::size);

  py::class_<FlagsWrapper<(size_t)PluginType::PluginTypeEnd, 3>>(
      m, "PluginTypeFlags")
      .def(py::init<>())
      .def(py::init<const std::string &>())
      .def("__and__",
           &FlagsWrapper<(size_t)PluginType::PluginTypeEnd, 3>::operator&)
      .def("__or__",
           &FlagsWrapper<(size_t)PluginType::PluginTypeEnd, 3>::operator|)
      .def("__xor__",
           &FlagsWrapper<(size_t)PluginType::PluginTypeEnd, 3>::operator^)
      .def("__invert__",
           &FlagsWrapper<(size_t)PluginType::PluginTypeEnd, 3>::operator~)
      .def("__eq__",
           [](const FlagsWrapper<(size_t)PluginType::PluginTypeEnd, 3> &self,
              const FlagsWrapper<(size_t)PluginType::PluginTypeEnd, 3> &other) {
             return self.flags == other.flags;
           })
      .def("__ne__",
           [](const FlagsWrapper<(size_t)PluginType::PluginTypeEnd, 3> &self,
              const FlagsWrapper<(size_t)PluginType::PluginTypeEnd, 3> &other) {
             return self.flags != other.flags;
           })
      .def("__str__",
           &FlagsWrapper<(size_t)PluginType::PluginTypeEnd, 3>::to_string)
      .def("__repr__",
           &FlagsWrapper<(size_t)PluginType::PluginTypeEnd, 3>::to_string)
      .def("set", &FlagsWrapper<(size_t)PluginType::PluginTypeEnd, 3>::set)
      .def("test", &FlagsWrapper<(size_t)PluginType::PluginTypeEnd, 3>::test)
      .def("size", &FlagsWrapper<(size_t)PluginType::PluginTypeEnd, 3>::size);
}

void register_utils(py::module &m) {
  auto utils = m.def_submodule("utils", "pccl utils function");
  utils.def("get_start_timestamp", &get_start_timestamp);
  utils.def("create_dir", &create_dir);
  utils.def("set_affinity", &set_affinity);
  utils.def("host_hash", &host_hash);
  utils.def("pid_hash", &pid_hash);
}

PYBIND11_MODULE(_pccl, m) {
  m.doc() = "PCCL runtime";
  register_config(m);
  register_types(m);
  register_utils(m);
}