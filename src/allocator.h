// Statically available allocators that have no internal state.

#pragma once

#include "core/common.h"
#include "module.h"

CVL_INTERNAL
extern const allocator_t CVL_MEM_ALLOCATOR;

CVL_INTERNAL
extern const allocator_t CVL_OBJ_ALLOCATOR;
