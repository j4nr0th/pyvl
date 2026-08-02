//
// Created by jan on 15.11.2024.
//

#include "common.h"

#include <stdlib.h>

static void *cvl_default_allocate(void *state, size_t size)
{
    (void)state;
    return malloc(size);
}

static void cvl_default_deallocate(void *state, void *ptr)
{
    (void)state;
    free(ptr);
}

static void *cvl_default_reallocate(void *state, void *ptr, size_t new_size)
{
    (void)state;
    return realloc(ptr, new_size);
}

/** @brief Default allocator backed by malloc/free/realloc. */
const allocator_t CVL_DEFAULT_ALLOCATOR = {
    .allocate = cvl_default_allocate,
    .deallocate = cvl_default_deallocate,
    .reallocate = cvl_default_reallocate,
    .state = NULL,
};
