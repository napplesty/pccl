#pragma once

#include <functional>
#include <memory>
#include <string>
#include <type_traits>
#include <tuple>
#include <utility>
#include <typeindex>
#include <concepts>
#include <unordered_map>
#include "runtime/meta.h"

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
concept HasValidExecuteImpl = requires {
    requires HasExecuteImpl<T>;
    typename ExecuteImplTraits<decltype(&T::execute_impl)>;
    requires std::same_as<
        typename ExecuteImplTraits<decltype(&T::execute_impl)>::return_type, 
        void
    >;
    requires ExecuteImplTraits<decltype(&T::execute_impl)>::all_args_pod;
};

#define ASSERT_HAS_EXECUTE_IMPL(Class) \
  static_assert(HasValidExecuteImpl<Class>, \
                #Class " must have execute_impl() member function with void return type and all POD arguments")
  
template<typename T>
concept ExecutableOperator = requires(T& op, char* input) {
    { op.execute(input) } -> std::same_as<void>;
};

template<typename T>
struct function_traits;

template<typename R, typename... Args>
struct function_traits<R(Args...)> {
    using return_type = R;
    using args_tuple = std::tuple<Args...>;
    static constexpr size_t arity = sizeof...(Args);
};

template<typename R, typename... Args>
struct function_traits<R(*)(Args...)> : function_traits<R(Args...)> {};

template<typename T, typename R, typename... Args>
struct function_traits<R(T::*)(Args...)> : function_traits<R(Args...)> {};

template<typename T, typename R, typename... Args>
struct function_traits<R(T::*)(Args...) const> : function_traits<R(Args...)> {};

template<typename Derived>
class Operator {
  ASSERT_HAS_EXECUTE_IMPL(Derived);
public:
  void execute(char* input) {
    using execute_impl_t = decltype(&Derived::execute_impl);
    using traits = function_traits<execute_impl_t>;
    
    [this, input]<typename... Args>(std::type_identity<std::tuple<Args...>>) {
        unpack_and_call<Args...>(input);
    }(std::type_identity<typename traits::args_tuple>{});
  }

  static auto input_types() -> std::vector<std::type_index> {
    using execute_impl_t = decltype(&Derived::execute_impl);
    using traits = function_traits<execute_impl_t>;
    
    return []<typename... Args>(std::tuple<Args...>) {
      std::vector<std::type_index> types;
      (types.push_back(typeid(Args)), ...);
      return types;
    }(typename traits::args_tuple{});
  }

  static constexpr OperatorType type() {
    if constexpr (requires { Derived::operator_type; }) {
      return Derived::operator_type;
    }
    return OperatorType::COMPUTE;
  }

private:
  template<typename... Args>
  void unpack_and_call(char* input) {
    size_t offset = 0;
    
    std::tuple<Args...> args = [&]<size_t... Is>(std::index_sequence<Is...>) {
      return std::make_tuple(
        *reinterpret_cast<std::tuple_element_t<Is, std::tuple<Args...>>*>(
          input + (offset = (Is * sizeof(Args)), offset)
        )...
      );
    }(std::index_sequence_for<Args...>{});
    
    std::apply([this](auto&&... args) {
      static_cast<Derived*>(this)->execute_impl(std::forward<decltype(args)>(args)...);
    }, args);
  }
};

template<ExecutableOperator... Ops>
class CompoundOperator {
  std::tuple<Ops...> operators;

public:
  void execute(char** inputs) {
    [this, inputs]<size_t... Is>(std::index_sequence<Is...>) {
      (std::get<Is>(operators).execute(inputs[Is]), ...);
    }(std::index_sequence_for<Ops...>{});
  }

  static auto input_types() -> std::vector<std::vector<std::type_index>> {
    return []<size_t... Is>(std::index_sequence<Is...>) {
      return std::vector<std::vector<std::type_index>>{
        std::tuple_element_t<Is, std::tuple<Ops...>>::input_types()...
      };
    }(std::index_sequence_for<Ops...>{});
  }

  static constexpr OperatorType type() {
    if constexpr (sizeof...(Ops) > 0) {
      return std::tuple_element_t<0, std::tuple<Ops...>>::type();
    }
    return OperatorType::COMPUTE;
  }
};

class OperatorInterface {
public:
  virtual ~OperatorInterface() = default;
  virtual void execute(char** inputs) = 0;
  virtual auto get_input_types() const -> std::vector<std::vector<std::type_index>> = 0;
  virtual auto get_type() const -> OperatorType = 0;
};

template<ExecutableOperator Op>
class SingleOperatorWrapper : public OperatorInterface {
  Op op;

public:
  void execute(char** inputs) override { 
    op.execute(inputs[0]);
  }
    
  auto get_input_types() const -> std::vector<std::vector<std::type_index>> override {
    return {Op::input_types()};
  }
    
  auto get_type() const -> OperatorType override { return Op::type(); }
};

template<typename... Ops>
class CompoundOperatorWrapper : public OperatorInterface {
  CompoundOperator<Ops...> op;

public:
  void execute(char** inputs) override { 
    op.execute(inputs);
  }
    
  auto get_input_types() const -> std::vector<std::vector<std::type_index>> override {
    return op.input_types();
  }
    
  auto get_type() const -> OperatorType override { return op.type(); }
};

class OperatorRegistry {
  std::unordered_map<std::string, std::function<std::unique_ptr<OperatorInterface>()>> registry;

public:
  template<ExecutableOperator Op>
  void register_op(std::string name) {
    registry[std::move(name)] = [] {
      return std::make_unique<SingleOperatorWrapper<Op>>();
    };
  }

  template<typename... Ops>
  void register_compound_op(std::string name) {
    registry[std::move(name)] = [] {
      return std::make_unique<CompoundOperatorWrapper<Ops...>>();
    };
  }

  auto create(const std::string& name) const -> std::unique_ptr<OperatorInterface> {
    if (auto it = registry.find(name); it != registry.end()) {
      return it->second();
    }
    throw std::runtime_error("Unknown operator: " + name);
  }

  auto contains(const std::string& name) const -> bool {
    return registry.contains(name);
  }
};

} // namespace pccl

