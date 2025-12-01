#include <pybind11/pybind11.h>

#include <common.h>
#include <engine.h>

namespace py = pybind11;

using namespace engine_c;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "PCCL Python Bindings";

  py::class_<Engine>(m, "Engine")
    .def(py::init<int, int>())
    .def("initEngine", &Engine::initEngine)
    .def("regOp", &Engine::regOp)
    .def("exeOp", &Engine::exeOp)
    .def("exportEndpoint", &Engine::exportEndpoint)
    .def("joinCluster", &Engine::joinCluster)
    .def("exitCluster", &Engine::exitCluster);

  auto utils_submodule = m.def_submodule("utils");

  py::class_<utils::LaunchEnvironments>(utils_submodule, "LaunchEnvironments")
    .def_static("listOpt", &utils::LaunchEnvironments::listOpt)
    .def_static("getEnv", &utils::LaunchEnvironments::getEnv)
    .def_static("registerOpt", &utils::LaunchEnvironments::registerOpt);
}



