#include <filesystem>
#include "kernel_jit/handle.h"

namespace pccl {

class CUDACompiler {
public:
  static std::filesystem::path library_root_path;
  static std::filesystem::path library_include_path;
  static std::filesystem::path cuda_home;
  static std::string library_version;


};


} // namespace pccl