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

  using PyFlagsFunction = FlagsWrapper<(size_t)FunctionType::FunctionTypeEnd, 0>;
  using PyFlagsOperation = FlagsWrapper<(size_t)BasicOperationType::BasicOperationTypeEnd, 1>;
  using PyFlagsComponent = FlagsWrapper<(size_t)ComponentType::ComponentTypeEnd, 2>;
  using PyFlagsPlugin = FlagsWrapper<(size_t)PluginType::PluginTypeEnd, 3>;

  py::class_<PyFlagsFunction>(
      m, "FunctionTypeFlags")
      .def(py::init<>())
      .def(py::init<const std::string &>())
      .def("__and__", &PyFlagsFunction::operator&)
      .def("__or__", &PyFlagsFunction::operator|)
      .def("__xor__", &PyFlagsFunction::operator^)
      .def("__invert__", &PyFlagsFunction::operator~)
      .def("__eq__", [](const PyFlagsFunction& self, const PyFlagsFunction& other) { return self.flags == other.flags; })
      .def("__ne__", [](const PyFlagsFunction& self, const PyFlagsFunction& other) { return self.flags != other.flags; })
      .def("__str__", &PyFlagsFunction::to_string)
      .def("__repr__", &PyFlagsFunction::to_string)
      .def("set", &PyFlagsFunction::set)
      .def("test", &PyFlagsFunction::test)
      .def("size", &PyFlagsFunction::size);

  py::class_<PyFlagsOperation>(
      m, "OperationTypeFlags")
      .def(py::init<>())
      .def(py::init<const std::string &>())
      .def("__and__", &PyFlagsOperation::operator&)
      .def("__or__", &PyFlagsOperation::operator|)
      .def("__xor__", &PyFlagsOperation::operator^)
      .def("__invert__", &PyFlagsOperation::operator~)
      .def("__eq__", [](const PyFlagsOperation& self, const PyFlagsOperation& other) { return self.flags == other.flags; })
      .def("__ne__", [](const PyFlagsOperation& self, const PyFlagsOperation& other) { return self.flags != other.flags; })
      .def("__str__", &PyFlagsOperation::to_string)
      .def("__repr__", &PyFlagsOperation::to_string)
      .def("set", &PyFlagsOperation::set)
      .def("test", &PyFlagsOperation::test)
      .def("size", &PyFlagsOperation::size);

  py::class_<PyFlagsComponent>(
      m, "ComponentTypeFlags")
      .def(py::init<>())
      .def(py::init<const std::string &>())
      .def("__and__", &PyFlagsComponent::operator&)
      .def("__or__", &PyFlagsComponent::operator|)
      .def("__xor__", &PyFlagsComponent::operator^)
      .def("__invert__", &PyFlagsComponent::operator~)
      .def("__eq__", [](const PyFlagsComponent& self, const PyFlagsComponent& other) { return self.flags == other.flags; })
      .def("__ne__", [](const PyFlagsComponent& self, const PyFlagsComponent& other) { return self.flags != other.flags; })
      .def("__str__", &PyFlagsComponent::to_string)
      .def("__repr__", &PyFlagsComponent::to_string)
      .def("set", &PyFlagsComponent::set)
      .def("test", &PyFlagsComponent::test)
      .def("size", &PyFlagsComponent::size);

  py::class_<PyFlagsPlugin>(
      m, "PluginTypeFlags")
      .def(py::init<>())
      .def(py::init<const std::string &>())
      .def("__and__", &PyFlagsPlugin::operator&)
      .def("__or__", &PyFlagsPlugin::operator|)
      .def("__xor__", &PyFlagsPlugin::operator^)
      .def("__invert__", &PyFlagsPlugin::operator~)
      .def("__eq__", [](const PyFlagsPlugin& self, const PyFlagsPlugin& other) { return self.flags == other.flags; })
      .def("__ne__", [](const PyFlagsPlugin& self, const PyFlagsPlugin& other) { return self.flags != other.flags; })
      .def("__str__", &PyFlagsPlugin::to_string)
      .def("__repr__", &PyFlagsPlugin::to_string)
      .def("set", &PyFlagsPlugin::set)
      .def("test", &PyFlagsPlugin::test)
      .def("size", &PyFlagsPlugin::size);
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