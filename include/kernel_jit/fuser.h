#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <string_view>
#include <tuple>
#include "utils/hash.hpp"

namespace pccl {

class CxxOpFuser {
public:
  static std::filesystem::path library_root_path;
  static std::filesystem::path library_include_path;
  static std::filesystem::path cache_path;
  static uint64_t              library_version;

  CxxOpFuser();
  ~CxxOpFuser();

  CxxOpFuser(const CxxOpFuser&) = delete;
  CxxOpFuser& operator=(const CxxOpFuser&) = delete;

  CxxOpFuser(CxxOpFuser&& other) noexcept;
  CxxOpFuser& operator=(CxxOpFuser&& other) noexcept;

  std::function<void(char**)> &compile(std::string_view operators);

private:
  std::string includes;
  std::map<uint64_t, std::tuple<std::string, std::function<void(char**)>>> fused_op_cache;
};


} // namespace pccl