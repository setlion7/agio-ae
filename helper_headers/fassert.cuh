#pragma once

#include <cstdio>
#include <cstdlib>
#include <cuda.h>

#ifdef __CUDA_ARCH__

#define force_assert(eval)                                              \
    if ((eval) == false) {                                              \
        printf("%s: %s:%d  %s\n", __func__, __FILE__, __LINE__, #eval); \
        __nanosleep(1024 * 512);                                        \
        __trap();                                                       \
    }

#define force_assert_printf(eval, fmt, ...)                                                \
    if ((eval) == false) {                                                                 \
        printf("%s: %s:%d  %s, " fmt, __func__, __FILE__, __LINE__, #eval, ##__VA_ARGS__); \
        __nanosleep(1024 * 512);                                                           \
        __trap();                                                                          \
    }

#else

#define force_assert(eval)                                                       \
    if ((eval) == false) {                                                       \
        fprintf(stderr, "%s: %s:%d  %s\n", __func__, __FILE__, __LINE__, #eval); \
        std::abort();                                                            \
    }

#define force_assert_printf(eval, fmt, ...)                                                         \
    if ((eval) == false) {                                                                          \
        fprintf(stderr, "%s: %s:%d  %s, " fmt, __func__, __FILE__, __LINE__, #eval, ##__VA_ARGS__); \
        std::abort();                                                                               \
    }

#endif
