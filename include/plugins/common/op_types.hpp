#pragma once

#include <tuple>
#include <concepts>
#include <type_traits>

namespace pccl {

template<typename... Args>
constexpr bool all_pod_v = (std::is_standard_layout_v<Args> && ...) && (std::is_trivial_v<Args> && ...);

template<typename T>
concept HasExecuteImpl = requires(T obj) {
  &T::execute_impl;
};

template<typename T>
struct ExecuteImplTraits;

template<typename T, typename Ret, typename... Args>
struct ExecuteImplTraits<Ret(T::*)(Args...)> {
  using return_type = Ret;
  using argument_types = std::tuple<Args...>;
  static constexpr bool all_args_pod = all_pod_v<Args...>;
};

template<typename T, typename Ret, typename... Args>
struct ExecuteImplTraits<Ret(T::*)(Args...) const> {
  using return_type = Ret;
  using argument_types = std::tuple<Args...>;
  static constexpr bool all_args_pod = all_pod_v<Args...>;
};

template<typename T>
concept ValidExecuteImpl = requires {
  requires HasExecuteImpl<T>;
  typename ExecuteImplTraits<decltype(&T::execute_impl)>;
  requires std::same_as<
    typename ExecuteImplTraits<decltype(&T::execute_impl)>::return_type, 
    void
  >;
  requires ExecuteImplTraits<decltype(&T::execute_impl)>::all_args_pod;
};

#define ASSERT_VALID_EXECUTE_IMPL(Class) \
  static_assert(ValidExecuteImpl<Class>, \
                #Class " must have execute_impl() member function with void return type and all POD arguments")

} // namespace pccl

