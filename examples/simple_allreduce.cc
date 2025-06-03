#include "component/logging.h"
#include "runtime.h"
#include <cstdlib>
#include <iostream>
#include <memory>
#include <vector>

using namespace pccl;

int main(int argc, char *argv[]) {
  // 初始化环境
  auto env = getEnv();
  std::cout << "Process rank " << env->rank << " of " << env->worldSize
            << " starting..." << std::endl;

  // 创建通信器
  auto communicator = std::make_shared<Communicator>();

  // 注册远程端点（在实际应用中，这需要通过某种发现机制完成）
  for (int i = 0; i < env->worldSize; i++) {
    if (i != env->rank) {
      auto endpoint = std::make_shared<Endpoint>(communicator);
      communicator->registerEndpoint(endpoint, i);
    }
  }

  // 准备数据
  const size_t count = 1024;
  std::vector<float> sendbuf(count);
  std::vector<float> recvbuf(count);

  // 初始化发送缓冲区
  for (size_t i = 0; i < count; i++) {
    sendbuf[i] = static_cast<float>(env->rank + 1);
  }

  std::cout << "Rank " << env->rank << " initialized data with value "
            << sendbuf[0] << std::endl;

  // 注册操作符
  std::string op_path = "allreduce_ring.op";
  auto allreduce_op = communicator->registerOperator(op_path);

  if (!allreduce_op) {
    LOG_ERROR << "Failed to register AllReduce operator";
    return 1;
  }

  // 执行AllReduce
  Event event;
  event.flush = []() {};
  event.wait = []() {};
  event.record = []() {};

  std::cout << "Rank " << env->rank << " executing AllReduce..." << std::endl;

  Event result = allreduce_op->execute(
      env->rank, sendbuf.data(), recvbuf.data(), DataType::FP32,
      count * sizeof(float), count * sizeof(float), event,
      true, // flush
      0     // tag
  );

  // 等待完成
  result.wait();

  // 验证结果
  float expected = 0.0f;
  for (int i = 0; i < env->worldSize; i++) {
    expected += static_cast<float>(i + 1);
  }

  bool correct = true;
  for (size_t i = 0; i < count; i++) {
    if (std::abs(recvbuf[i] - expected) > 1e-6) {
      correct = false;
      break;
    }
  }

  if (correct) {
    std::cout << "Rank " << env->rank << " AllReduce completed successfully. "
              << "Result = " << recvbuf[0] << " (expected " << expected << ")"
              << std::endl;
  } else {
    std::cout << "Rank " << env->rank << " AllReduce FAILED! "
              << "Result = " << recvbuf[0] << " (expected " << expected << ")"
              << std::endl;
  }

  return correct ? 0 : 1;
}