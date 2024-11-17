//
// Created by jan on 17.11.2024.
//

#include "test_common.h"


enum { ALLOCATOR_MAGIC_NUMBER = 0xB16B00B1E5 };

static void *test_allocate(void *state, size_t sz)
{
    TEST_ASSERT(state == (void *)ALLOCATOR_MAGIC_NUMBER, "Allocator magic number does not match");
    return malloc(sz);
}

static void test_deallocate(void *state, void *ptr)
{
    TEST_ASSERT(state == (void *)ALLOCATOR_MAGIC_NUMBER, "Allocator magic number does not match");
    free(ptr);
}

static void *test_reallocate(void *state, void *ptr, size_t new_size)
{
    TEST_ASSERT(state == (void *)ALLOCATOR_MAGIC_NUMBER, "Allocator magic number does not match");
    return realloc(ptr, new_size);
}

const allocator_t TEST_ALLOCATOR = {
    .allocate = test_allocate,
    .deallocate = test_deallocate,
    .reallocate = test_reallocate,
    .state = (void *)ALLOCATOR_MAGIC_NUMBER
};
