#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <string_view>
#include <tuple>
#include "utils/hash.hpp"

namespace pccl {

class CxxFuser {
public:
  static std::filesystem::path library_root_path;
  static std::filesystem::path library_include_path;
  static std::filesystem::path cache_path;
  static uint64_t              library_version;

  CxxFuser();
  ~CxxFuser();

  CxxFuser(const CxxFuser&) = delete;
  CxxFuser& operator=(const CxxFuser&) = delete;

  CxxFuser(CxxFuser&& other) noexcept;
  CxxFuser& operator=(CxxFuser&& other) noexcept;

  std::function<void(char**)> &compile(std::string_view operators);

private:
  std::string includes;
  std::map<uint64_t, std::tuple<std::string, std::function<void(char**)>>> fused_op_cache;
};


} // namespace pccl