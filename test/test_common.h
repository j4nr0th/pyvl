//
// Created by jan on 17.11.2024.
//

#ifndef TEST_COMMON_H
#define TEST_COMMON_H

#include "../src/common.h"
#include <stdio.h>
#include <stdlib.h>

#define TEST_ASSERT(expr, msg, ...) ((expr) ? 1 : ((fprintf(stderr, "Failed assertion \"" #expr "\" at %s:%d in function %s: " msg, __FILE__, __LINE__, __func__ __VA_OPT__(,) __VA_ARGS__ ), exit(EXIT_FAILURE), 0)))

extern const allocator_t TEST_ALLOCATOR;

#endif //TEST_COMMON_H
