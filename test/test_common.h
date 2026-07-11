//
// Created by jan on 17.11.2024.
//

#ifndef TEST_COMMON_H
#define TEST_COMMON_H

#include "../src/core/common.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef NDEBUG
#ifdef __GNUC__
#define DBG_BREAK __builtin_trap()
#endif
#endif

#ifndef DBG_BREAK
#define DBG_BREAK (void)0
#endif

#define TEST_ASSERT(expr, msg, ...)                                                                                    \
    ((expr) ? 1                                                                                                        \
            : ((fprintf(stderr, "Failed assertion \"" #expr "\" at %s:%d in function %s: " msg "\n", __FILE__,         \
                        __LINE__, __func__ __VA_OPT__(, ) __VA_ARGS__),                                                \
                DBG_BREAK, exit(EXIT_FAILURE), 0)))

extern const allocator_t TEST_ALLOCATOR;

char *read_file_to_string(const char *path, size_t chunk_size);

static inline uint64_t xorshift64(uint64_t *state)
{
    uint64_t x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    return x;
}

static inline real_t xorshift_uniform01(uint64_t *state)
{
    // 53-bit precision uniform [0, 1)
    return (real_t)(xorshift64(state) >> 11) / (real_t)((uint64_t)1 << 53);
}

static inline real_t xorshift_uniform_range(uint64_t *state, real_t lo, real_t hi)
{
    return lo + xorshift_uniform01(state) * (hi - lo);
}

#endif // TEST_COMMON_H
