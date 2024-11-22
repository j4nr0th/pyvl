//
// Created by jan on 17.11.2024.
//

#ifndef TEST_COMMON_H
#define TEST_COMMON_H

#include "../src/common.h"
#include <stdio.h>
#include <stdlib.h>

#ifndef NDEBUG
#   ifdef __GNUC__
#       define DBG_BREAK __builtin_trap()
#   endif
#endif

#ifndef DBG_BREAK
#   define DBG_BREAK (void)0
#endif

#define TEST_ASSERT(expr, msg, ...) ((expr) ? 1 : ((fprintf(stderr, "Failed assertion \"" #expr "\" at %s:%d in function %s: " msg "\n", __FILE__, __LINE__, __func__ __VA_OPT__(,) __VA_ARGS__ ), DBG_BREAK, exit(EXIT_FAILURE), 0)))

extern const allocator_t TEST_ALLOCATOR;

char *read_file_to_string(const char *path, size_t chunk_size);

#endif //TEST_COMMON_H
