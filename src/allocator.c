//
// Created by jan on 24.11.2024.
//

#include "allocator.h"

enum
{
    MEM_ALLOCATOR_MAGIC = 0xB00B1E5,
    OBJ_ALLOCATOR_MAGIC = 0xB00B5
};

static void *mem_allocate(void *state, size_t size)
{
    (void)state;
    return PyMem_Malloc(size);
}

static void *mem_reallocate(void *state, void *ptr, size_t new_size)
{
    (void)state;
    return PyMem_Realloc(ptr, new_size);
}

static void mem_deallocate(void *state, void *ptr)
{
    (void)state;
    PyMem_Free(ptr);
}

CVL_INTERNAL
const allocator_t CVL_MEM_ALLOCATOR = {
    .allocate = mem_allocate,
    .deallocate = mem_deallocate,
    .reallocate = mem_reallocate,
    .state = (void *)MEM_ALLOCATOR_MAGIC,
};

static void *obj_allocate(void *state, size_t size)
{
    (void)state;
    return PyObject_Malloc(size);
}

static void *obj_reallocate(void *state, void *ptr, size_t new_size)
{
    (void)state;
    return PyObject_Realloc(ptr, new_size);
}

static void obj_deallocate(void *state, void *ptr)
{
    (void)state;
    PyObject_Free(ptr);
}

CVL_INTERNAL
const allocator_t CVL_OBJ_ALLOCATOR = {
    .allocate = obj_allocate,
    .deallocate = obj_deallocate,
    .reallocate = obj_reallocate,
    .state = (void *)OBJ_ALLOCATOR_MAGIC,
};
