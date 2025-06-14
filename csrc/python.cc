#include "pybind11/pybind11.h"
#include "config.h"
#include "types.h"

namespace pccl {

template <size_t N>
class FlagsWrapper {
public:
  std::bitset<N> flags;
  FlagsWrapper() = default;
  FlagsWrapper(const std::string& str) {
    if (str.size() >= 2 && str.compare(0, 2, "0b") == 0) {
      flags = std::bitset<N>(str.substr(2));
    } 
    else {
      flags = std::bitset<N>(str);
    }
  }

  std::string to_string() const {
    return "0b" + flags.to_string();
  }

  FlagsWrapper operator&(const FlagsWrapper& other) const {
    FlagsWrapper result;
    result.flags = this->flags & other.flags;
    return result;
  }

  FlagsWrapper operator|(const FlagsWrapper& other) const {
    FlagsWrapper result;
    result.flags = this->flags | other.flags;
    return result;
  }

  FlagsWrapper operator^(const FlagsWrapper& other) const {
    FlagsWrapper result;
    result.flags = this->flags ^ other.flags;
    return result;
  }

  FlagsWrapper operator~() const {
    FlagsWrapper result;
    result.flags = ~this->flags;
    return result;
  }

  bool operator==(const FlagsWrapper& other) const {
    return this->flags == other.flags;
  }

  bool operator!=(const FlagsWrapper& other) const {
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
    .def_readwrite_static("PROXY_MAX_FLUSH_SIZE", &Config::PROXY_MAX_FLUSH_SIZE)
    .def_readwrite_static("PROXY_CHECK_STOP_PERIOD", &Config::PROXY_CHECK_STOP_PERIOD);
}

void register_types(py::module &m) {
  py::class_<FlagsWrapper<(size_t)FunctionType::FunctionTypeEnd>>(m, "FunctionTypeFlags")
    .def(py::init<>())
    .def(py::init<const std::string&>())
    .def("__and__", &FlagsWrapper<(size_t)FunctionType::FunctionTypeEnd>::operator&)
    .def("__or__", &FlagsWrapper<(size_t)FunctionType::FunctionTypeEnd>::operator|)
    .def("__xor__", &FlagsWrapper<(size_t)FunctionType::FunctionTypeEnd>::operator^)
    .def("__invert__", &FlagsWrapper<(size_t)FunctionType::FunctionTypeEnd>::operator~)
    .def("__eq__", [](const FlagsWrapper<(size_t)FunctionType::FunctionTypeEnd>& self, const FlagsWrapper<(size_t)FunctionType::FunctionTypeEnd>& other) { return self.flags == other.flags; })
    .def("__ne__", [](const FlagsWrapper<(size_t)FunctionType::FunctionTypeEnd>& self, const FlagsWrapper<(size_t)FunctionType::FunctionTypeEnd>& other) { return self.flags != other.flags; })
    .def("__str__", &FlagsWrapper<(size_t)FunctionType::FunctionTypeEnd>::to_string)
    .def("__repr__", &FlagsWrapper<(size_t)FunctionType::FunctionTypeEnd>::to_string)
    .def("set", &FlagsWrapper<(size_t)FunctionType::FunctionTypeEnd>::set)
    .def("test", &FlagsWrapper<(size_t)FunctionType::FunctionTypeEnd>::test)
    .def("size", &FlagsWrapper<(size_t)FunctionType::FunctionTypeEnd>::size);

  py::class_<FlagsWrapper<(size_t)BasicOperationType::BasicOperationTypeEnd>>(m, "OperationTypeFlags")
    .def(py::init<>())
    .def(py::init<const std::string&>())
    .def("__and__", &FlagsWrapper<(size_t)BasicOperationType::BasicOperationTypeEnd>::operator&)
    .def("__or__", &FlagsWrapper<(size_t)BasicOperationType::BasicOperationTypeEnd>::operator|)
    .def("__xor__", &FlagsWrapper<(size_t)BasicOperationType::BasicOperationTypeEnd>::operator^)
    .def("__invert__", &FlagsWrapper<(size_t)BasicOperationType::BasicOperationTypeEnd>::operator~)
    .def("__eq__", [](const FlagsWrapper<(size_t)BasicOperationType::BasicOperationTypeEnd>& self, const FlagsWrapper<(size_t)BasicOperationType::BasicOperationTypeEnd>& other) { return self.flags == other.flags; })
    .def("__ne__", [](const FlagsWrapper<(size_t)BasicOperationType::BasicOperationTypeEnd>& self, const FlagsWrapper<(size_t)BasicOperationType::BasicOperationTypeEnd>& other) { return self.flags != other.flags; })
    .def("__str__", &FlagsWrapper<(size_t)BasicOperationType::BasicOperationTypeEnd>::to_string)
    .def("__repr__", &FlagsWrapper<(size_t)BasicOperationType::BasicOperationTypeEnd>::to_string)
    .def("set", &FlagsWrapper<(size_t)BasicOperationType::BasicOperationTypeEnd>::set)
    .def("test", &FlagsWrapper<(size_t)BasicOperationType::BasicOperationTypeEnd>::test)
    .def("size", &FlagsWrapper<(size_t)BasicOperationType::BasicOperationTypeEnd>::size);
};

PYBIND11_MODULE(_pccl, m) {
  m.doc() = "PCCL runtime";
  register_config(m);
  register_types(m);
}