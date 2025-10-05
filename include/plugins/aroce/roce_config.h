#include <string>
namespace pccl::communicator {
class VerbsConfig {
 public:
  VerbsConfig();
  static VerbsConfig& getInstance() {
    static VerbsConfig instance;
    return instance;
  }
  const std::string& getLibPath() const { return lib_path; }
  const std::string& getDeviceName() const { return device_name; }
  int getPortNum() const { return port_num; }
  int getGidIndex() const { return gid_index; }
  int getQkey() const { return qkey; }
  static constexpr int max_send_wr = 1024;
  static constexpr int max_recv_wr = 1024;
  static constexpr int max_send_sge = 16;
  static constexpr int max_recv_sge = 16;
  static constexpr int max_inline_data = 256;
 public:
  std::string lib_path;
  std::string device_name;
  int port_num;
  int gid_index;
  int qkey;
 private:
  VerbsConfig(const VerbsConfig&) = delete;
  VerbsConfig& operator=(const VerbsConfig&) = delete;
};

} // namespace pccl::communicator
