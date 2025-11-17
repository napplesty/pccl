#include <string>
#include <string_view>

namespace engine_c::utils {

std::string serialize(void *ptr, size_t nbyte);
void deserialize(void *ptr, std::string_view str);

}
