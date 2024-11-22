//
// Created by jan on 17.11.2024.
//

#include "test_common.h"

#include <errno.h>
#include <string.h>


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

char* read_file_to_string(const char* path, size_t chunk_size)
{
    FILE *f_in = fopen(path, "r");
    TEST_ASSERT(f_in, "Could not open file %s, %s", path, strerror(errno));
    size_t buffer_size = chunk_size;
    char *buffer = malloc((sizeof *buffer) * buffer_size);
    TEST_ASSERT(buffer, "Buffer not allocate");
    size_t read_sz;
    while ((read_sz = fread(buffer + buffer_size - chunk_size, 1, chunk_size, f_in)) == chunk_size)
    {
        buffer_size += chunk_size;
        char *new_buffer = realloc(buffer, buffer_size * sizeof(*buffer));
        TEST_ASSERT(new_buffer, "Buffer not reallocated");
        buffer = new_buffer;
    }
    fclose(f_in);
    buffer[buffer_size - chunk_size + read_sz] = 0;
    return buffer;
}
