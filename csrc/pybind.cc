#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>

#include "runtime/api/configs.h"
#include "runtime/api/repr.h"
#include "runtime/api/runtime.h"

namespace py = pybind11;

using namespace pccl;
using namespace pccl::runtime;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "PCCL (Parallel Collective Communication Library) Python Bindings";

    // === 枚举类型绑定 ===
    
    // PrimitiveType 枚举
    py::enum_<PrimitiveType>(m, "PrimitiveType")
        .value("WRITE", PrimitiveType::WRITE)
        .value("COMPUTE", PrimitiveType::COMPUTE)
        .value("COPY", PrimitiveType::COPY)
        .value("SIGNAL", PrimitiveType::SIGNAL)
        .value("WAITSIGNAL", PrimitiveType::WAITSIGNAL)
        .export_values();

    // DataType 枚举
    py::enum_<DataType>(m, "DataType")
        .value("F32", DataType::F32)
        .value("F16", DataType::F16)
        .value("BF16", DataType::BF16)
        .export_values();

    // ComputeType 枚举
    py::enum_<ComputeType>(m, "ComputeType")
        .value("SUM", ComputeType::SUM)
        .value("MAX", ComputeType::MAX)
        .value("MIN", ComputeType::MIN)
        .value("PROD", ComputeType::PROD)
        .export_values();

    // ExecutorType 枚举
    py::enum_<ExecutorType>(m, "ExecutorType")
        .value("CPU", ExecutorType::CPU)
        .value("CUDA", ExecutorType::CUDA)
        .value("LAST", ExecutorType::LAST)
        .export_values();

    // === 配置结构体绑定 ===
    
    // BufferConfig 结构体
    py::class_<BufferConfig>(m, "BufferConfig")
        .def(py::init<>())
        .def_readwrite("buffer_idx", &BufferConfig::buffer_idx)
        .def_readwrite("dtype", &BufferConfig::dtype)
        .def_readwrite("size", &BufferConfig::size)
        .def_readwrite("executor_type", &BufferConfig::executor_type)
        .def("toJson", &BufferConfig::toJson)
        .def_static("fromJson", &BufferConfig::fromJson);

    // ExecutorConfig 结构体
    py::class_<ExecutorConfig>(m, "ExecutorConfig")
        .def(py::init<>())
        .def_readwrite("executor_type", &ExecutorConfig::executor_type)
        .def_readwrite("num_total_executors", &ExecutorConfig::num_total_executors)
        .def("toJson", &ExecutorConfig::toJson)
        .def_static("fromJson", &ExecutorConfig::fromJson);

    // PrimitiveConfig 结构体
    py::class_<PrimitiveConfig>(m, "PrimitiveConfig")
        .def(py::init<>())
        .def_readwrite("type", &PrimitiveConfig::type)
        .def_readwrite("dtype", &PrimitiveConfig::dtype)
        .def_readwrite("target_rank", &PrimitiveConfig::target_rank)
        .def_readwrite("src_buffer_idx", &PrimitiveConfig::src_buffer_idx)
        .def_readwrite("dst_buffer_idx", &PrimitiveConfig::dst_buffer_idx)
        .def_readwrite("compute_op", &PrimitiveConfig::compute_op)
        .def_readwrite("executor_type", &PrimitiveConfig::executor_type)
        .def_readwrite("num_executors", &PrimitiveConfig::num_executors)
        .def_readwrite("data_size", &PrimitiveConfig::data_size)
        .def_readwrite("signal_value", &PrimitiveConfig::signal_value)
        .def_readwrite("num_dependencies", &PrimitiveConfig::num_dependencies)
        .def_readwrite("num_followers", &PrimitiveConfig::num_followers)
        .def("toJson", &PrimitiveConfig::toJson)
        .def_static("fromJson", &PrimitiveConfig::fromJson);

    // RuntimeConfig 结构体
    py::class_<RuntimeConfig>(m, "RuntimeConfig")
        .def(py::init<>())
        .def_readwrite("local_rank", &RuntimeConfig::local_rank)
        .def_readwrite("world_size", &RuntimeConfig::world_size)
        .def_readwrite("buffers_per_executor", &RuntimeConfig::buffers_per_executor)
        .def_readwrite("default_buffer_sizes", &RuntimeConfig::default_buffer_sizes)
        .def_readwrite("extra_config", &RuntimeConfig::extra_config);

    // === PrimitiveGrpah 类绑定 ===
    py::class_<PrimitiveGrpah>(m, "PrimitiveGraph")
        .def(py::init<int>(), py::arg("rank"))
        .def(py::init<const nlohmann::json&>(), py::arg("json_data"))
        .def("addBuffer", &PrimitiveGrpah::addBuffer, 
             py::arg("idx"), py::arg("dtype"), py::arg("size"))
        .def("addOperator", &PrimitiveGrpah::addOperator, py::arg("op"))
        .def("addDependency", &PrimitiveGrpah::addDependency, 
             py::arg("from_op_id"), py::arg("to_op_id"))
        .def("getRank", &PrimitiveGrpah::getRank)
        .def("getBuffers", &PrimitiveGrpah::getBuffers)
        .def("getOperators", &PrimitiveGrpah::getOperators)
        .def("getExecutors", &PrimitiveGrpah::getExecutors)
        .def_static("loadFromFile", &PrimitiveGrpah::loadFromFile, 
                   py::arg("filename"), py::arg("rank"))
        .def_static("loadFromJson", &PrimitiveGrpah::loadFromJson, py::arg("json_data"));

    // === 全局函数绑定 ===
    
    // 运行时初始化函数
    m.def("initializeRuntime", &initializeRuntime, 
          py::arg("config"), 
          "Initialize the PCCL runtime with the given configuration");

    // 运行时关闭函数
    m.def("shutdownRuntime", &shutdownRuntime, 
          "Shutdown the PCCL runtime");

    // 图执行函数
    m.def("executeGraph", &executeGraph, 
          py::arg("graph"), py::arg("participants"),
          "Execute a primitive graph with the given participants");

    // 全局配置获取函数
    m.def("get_global_config", &get_global_config, 
          "Get the global PCCL configuration");
    // === 异常处理 ===
  
    // === 模块级函数 ===
    
    m.def("get_version", []() -> std::string {
        return "PCCL Python Bindings v1.0.0";
    }, "Get the version of PCCL Python bindings");

    // === 属性绑定 ===
    
    m.attr("__version__") = "1.0.0";
    m.attr("__author__") = "PCCL Team";
}

