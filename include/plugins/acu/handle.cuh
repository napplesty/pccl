#pragma once

#include <type_traits>
#include <tuple>
#include <cuda.h>
#include <cuda_runtime.h>

namespace pccl {

enum class OperatorType { 
  SYNC_COMPUTE,
  ASYNC_COMPUTE,
  SYNC_IO, 
  ASYNC_IO 
};

// 元编程辅助：检查类型是否为Operator
template<typename T>
struct is_operator {
private:
    template<typename U>
    static std::true_type test(decltype(&U::type), decltype(&U::execute_impl));
    
    template<typename U>
    static std::false_type test(...);
    
public:
    static constexpr bool value = decltype(test<T>(nullptr, nullptr))::value;
};

// CUDACompoundOperator 模板类
template<typename... Ops>
class CUDACompoundOperator {
private:
    static_assert(sizeof...(Ops) > 0, "At least one operator is required");
    static_assert((is_operator<Ops>::value && ...), "All template parameters must be Operators");
    
    // 内部执行函数 - 分发到具体的操作
    template<size_t Index>
    static __device__ void execute_op(char* arg) {
        using Op = typename std::tuple_element<Index, std::tuple<Ops...>>::type;
        Op::execute_impl(arg);
    }
    
    // 递归模板展开执行所有操作
    template<size_t... Indices>
    static __device__ void execute_ops_impl(char** args, std::index_sequence<Indices...>) {
        // 使用折叠表达式按顺序执行所有操作
        (execute_op<Indices>(args[Indices]), ...);
    }

public:
    // 执行所有操作的入口点
    static __global__ void execute(char** args) {
        // 按顺序执行所有操作
        execute_ops_impl(args, std::index_sequence_for<Ops...>{});
    }
    
    // 获取启动配置
    static dim3 get_grid_dim() {
        return dim3(1, 1, 1); // 使用单个block
    }
    
    static dim3 get_block_dim() {
        return dim3(1, 1, 1); // 使用单个线程
    }
    
    // 获取操作数量
    static constexpr size_t num_operators() {
        return sizeof...(Ops);
    }
};

}