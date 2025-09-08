#pragma once

namespace pccl {

#define ASSERT_DEVICE(__cond, __msg)                         \
  do {                                                               \
    if (!(__cond)) {                                                 \
      __assert_fail(__msg, __FILE__, __LINE__, __PRETTY_FUNCTION__); \
    }                                                                \
  } while (0)

#define POLL_MAYBE_JAILBREAK(__cond, __max_spin_cnt)                                          \
  do {                                                                                        \
    [[maybe_unused]] int64_t __spin_cnt = 0;                                                  \
    while (__cond) {                                                                          \
      ASSERT_DEVICE((__max_spin_cnt < 0 || __spin_cnt++ != __max_spin_cnt), #__cond); \
    }                                                                                         \
  } while (0);

#define OR_POLL_MAYBE_JAILBREAK(__cond1, __cond2, __max_spin_cnt)                                       \
  do {                                                                                                  \
    [[maybe_unused]] int64_t __spin_cnt = 0;                                                            \
    while (true) {                                                                                      \
      if (!(__cond1)) {                                                                                 \
        break;                                                                                          \
      } else if (!(__cond2)) {                                                                          \
        break;                                                                                          \
      }                                                                                                 \
      ASSERT_DEVICE((__max_spin_cnt < 0 || __spin_cnt++ != __max_spin_cnt), #__cond1 #__cond2); \
    }                                                                                                   \
  } while (0);

#define CUDA_CHECK(status) \
  do { \
    if (status != cudaSuccess) \
      throw; \
  } while (0)

} // namespace pccl
