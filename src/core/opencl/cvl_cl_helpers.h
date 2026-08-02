#pragma once
/*
 * Shared inline helpers for the OpenCL wrapper and cl_compute layers.
 *
 * Provides:
 *   - Allocator wrappers (cl_resolve_allocator, cl_alloc, cl_free, cl_realloc)
 *   - cl_ensure_buffer - grow-or-create a device buffer
 *   - zero_device_buffer - zero-fill a device buffer via host write
 *   - cl_prepare_wait_list - build a raw cl_event array from typed events
 *
 * All functions are static inline - the build uses INTERPROCEDURAL_OPTIMIZATION
 * so the compiler inlines and optimises as if the code were local.
 */

#include "../common.h"
#include "cvl_cl_buffer.h"
#include "cvl_cl_command.h"
#include "cvl_cl_common.h"
#include "cvl_cl_ctx.h"

#include <CL/cl.h>
#include <stdlib.h>
#include <string.h>

/* ================================================================ */
/* Allocator helpers                                                */
/* ================================================================ */

/**
 * @brief Resolve an allocator pointer, defaulting to CVL_DEFAULT_ALLOCATOR.
 *
 * @param allocator  Allocator to use, or NULL for the default.
 * @return Pointer to the resolved allocator (never NULL).
 */
static inline const allocator_t *cl_resolve_allocator(const allocator_t *allocator)
{
    return allocator ? allocator : &CVL_DEFAULT_ALLOCATOR;
}

/**
 * @brief Allocate @p size bytes via the given allocator.
 *
 * @param allocator  Allocator (NULL resolves to default).
 * @param size       Number of bytes to allocate.
 * @return Pointer to allocated memory, or NULL on failure.
 */
static inline void *cl_alloc(const allocator_t *allocator, size_t size)
{
    const allocator_t *a = cl_resolve_allocator(allocator);
    return a->allocate(a->state, size);
}

/**
 * @brief Free a pointer via the given allocator.
 *
 * @param allocator  Allocator (NULL resolves to default).
 * @param ptr        Pointer to free (NULL is a no-op).
 */
static inline void cl_free(const allocator_t *allocator, void *ptr)
{
    if (ptr == NULL)
        return;
    const allocator_t *a = cl_resolve_allocator(allocator);
    a->deallocate(a->state, ptr);
}

/**
 * @brief Reallocate a pointer via the given allocator.
 *
 * @param allocator  Allocator (NULL resolves to default).
 * @param ptr        Pointer to reallocate (NULL acts as malloc).
 * @param new_size   New size in bytes.
 * @return Pointer to reallocated memory, or NULL on failure.
 */
static inline void *cl_realloc(const allocator_t *allocator, void *ptr, size_t new_size)
{
    const allocator_t *a = cl_resolve_allocator(allocator);
    return a->reallocate(a->state, ptr, new_size);
}

/* ================================================================ */
/* Buffer helpers                                                   */
/* ================================================================ */

/**
 * @brief Maximum number of events in a wait list.
 */
enum
{
    CL_MAX_WAIT_EVENTS = 16
};

/**
 * @brief Grow or create a device buffer to at least @p size_bytes.
 *
 * If the buffer does not exist yet, it is created via cvl_cl_buffer_create.
 * If it exists but capacity is insufficient, it is grown via
 * cvl_cl_buffer_reserve (preserving existing content).  Otherwise only
 * the logical size is updated.
 *
 * @param buf        Buffer handle (may have mem == NULL on first call).
 * @param ctx        OpenCL context (borrowed).
 * @param queue      Command queue for reserve (may be NULL if buf is large enough).
 * @param size_bytes Minimum required capacity in bytes.
 * @return CVL_CL_SUCCESS or error.
 */
static inline cvl_cl_status_t cl_ensure_buffer(cvl_cl_buffer_t *buf, const cvl_cl_ctx_t *ctx, cvl_cl_queue_t *queue,
                                               size_t size_bytes)
{
    if (!buf->mem)
    {
        const cvl_cl_buffer_desc_t desc = {
            .access = CVL_CL_BUF_READ_WRITE,
            .size_bytes = size_bytes,
            .host_ptr = NULL,
            .use_host_ptr = false,
        };
        return cvl_cl_buffer_create(ctx, &desc, buf);
    }

    if (buf->capacity >= size_bytes)
    {
        buf->size = size_bytes;
        return CVL_CL_SUCCESS;
    }

    cvl_cl_status_t st = cvl_cl_buffer_reserve(buf, ctx, queue, size_bytes);
    if (st == CVL_CL_SUCCESS)
        buf->size = size_bytes;
    return st;
}

/**
 * @brief Zero-fill a device buffer via a host-side write.
 *
 * Allocates a temporary zeroed host buffer using the given allocator,
 * writes it to the device, and finishes.  Suitable for infrequent resets.
 *
 * @param allocator  Allocator for the temporary host buffer (NULL = default).
 * @param queue      Command queue.
 * @param buf        Device buffer to zero.
 * @param size_bytes Number of bytes to zero.
 * @return CVL_CL_SUCCESS or error.
 */
static inline cvl_cl_status_t zero_device_buffer(const allocator_t *allocator, cvl_cl_queue_t *queue,
                                                 cvl_cl_buffer_t *buf, size_t size_bytes)
{
    if (size_bytes == 0)
        return CVL_CL_SUCCESS;

    void *zeros = cl_alloc(allocator, size_bytes);
    if (!zeros)
        return CVL_CL_ERR_MEMORY;
    memset(zeros, 0, size_bytes);

    /* The write is non-blocking (CL_FALSE) - the driver may still be copying
     * from @p zeros after this call returns, so the buffer must stay alive
     * until the queue is finished.  Freeing it earlier is a use-after-free
     * that crashes NVIDIA's async copy path (observed as a segfault in the
     * driver's event-handler thread). */
    cvl_cl_status_t st = cvl_cl_write_buffer(queue, buf, 0, size_bytes, zeros, 0, NULL, NULL);
    if (st == CVL_CL_SUCCESS)
        st = cvl_cl_finish(queue);
    cl_free(allocator, zeros);
    return st;
}

/* ================================================================ */
/* Event wait-list helpers                                          */
/* ================================================================ */

/**
 * @brief Prepare a raw cl_event wait-list and output-event pointer.
 *
 * Builds a stack-local `cl_event` array from the typed wait events and
 * sets `*out_ptr` to point at it (or NULL if no waits).  Also initialises
 * `*raw_out` to NULL and sets `*out_ptr_event` to point at it (or NULL
 * if the caller does not want an output event).
 *
 * @param n_wait          Number of wait events (0 … CL_MAX_WAIT_EVENTS).
 * @param wait_events     Array of typed wait events (may be NULL if n_wait == 0).
 * @param wait_list_raw   Stack buffer for raw cl_event values (must hold CL_MAX_WAIT_EVENTS).
 * @param out_wait_ptr    [out] Pointer to the raw list, or NULL.
 * @param raw_out         [in,out] Storage for the output event handle.
 * @param out_event       [out] Non-NULL if the caller wants an output event.
 * @param out_ptr_event   [out] Pointer to raw_out, or NULL.
 * @return CVL_CL_SUCCESS or CVL_CL_ERR_INVALID_PARAM if n_wait > CL_MAX_WAIT_EVENTS.
 */
static inline cvl_cl_status_t cl_prepare_wait_list(unsigned n_wait, const cvl_cl_event_t *wait_events,
                                                   cl_event wait_list_raw[CL_MAX_WAIT_EVENTS], cl_event **out_wait_ptr,
                                                   cl_event *raw_out, const cvl_cl_event_t *out_event,
                                                   cl_event **out_ptr_event)
{
    *out_wait_ptr = NULL;
    if (n_wait > 0)
    {
        if (n_wait > CL_MAX_WAIT_EVENTS)
            return CVL_CL_ERR_INVALID_PARAM;
        if (n_wait == 1 && wait_events)
        {
            wait_list_raw[0] = wait_events[0].event;
        }
        else
        {
            for (unsigned i = 0; i < n_wait; ++i)
                wait_list_raw[i] = wait_events[i].event;
        }
        *out_wait_ptr = wait_list_raw;
    }

    *raw_out = NULL;
    *out_ptr_event = out_event ? raw_out : NULL;
    return CVL_CL_SUCCESS;
}
