#pragma once
/*
 * Shared inline helpers for the OpenCL wrapper and cl_compute layers.
 *
 * Provides:
 *   - cl_ensure_buffer_chained   - grow-or-create a device buffer via a chain
 *   - zero_device_buffer_chained - zero-fill a device buffer via a chain
 *   - cl_prepare_wait_list - build a raw cl_event array from typed events
 *
 * All functions are static inline - the build uses INTERPROCEDURAL_OPTIMIZATION
 * so the compiler inlines and optimises as if the code were local.
 */

#include "cvl_cl_buffer.h"
#include "cvl_cl_chain.h"
#include "cvl_cl_command.h"
#include "cvl_cl_common.h"

#include <CL/cl.h>
#include <assert.h>
#include <string.h>

/* ================================================================ */
/* Buffer helpers                                                   */
/* ================================================================ */

/**
 * @brief Grow or create a device buffer to at least @p size_bytes.
 *
 * If the buffer does not exist yet it is created via cvl_cl_buffer_create
 * (a synchronous host-side call - no device work to chain).  If it exists
 * but capacity is insufficient it is grown asynchronously through
 * @p chain via cvl_cl_chain_grow_buffer, so the content copy is ordered
 * after the chain's pending operations and downstream chained ops wait
 * for the growth.  Otherwise only the logical size is updated.
 *
 * @param buf         Buffer handle (may have mem == NULL on first call).
 * @param ctx         OpenCL context.
 * @param chain       Chain to order and track the growth on.
 * @param size_bytes  Minimum required capacity in bytes.
 * @return CVL_CL_SUCCESS or error.
 */
static inline cvl_cl_status_t cl_ensure_buffer_chained(cvl_cl_buffer_t *buf, cl_context ctx, cvl_cl_chain_t *chain,
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
        cvl_cl_status_t st = cvl_cl_buffer_create(ctx, &desc, buf);
        if (st == CVL_CL_SUCCESS)
            buf->size = size_bytes;
        return st;
    }

    if (buf->capacity >= size_bytes)
    {
        buf->size = size_bytes;
        return CVL_CL_SUCCESS;
    }

    cvl_cl_status_t st = cvl_cl_chain_grow_buffer(chain, buf, ctx, size_bytes);
    if (st == CVL_CL_SUCCESS)
        buf->size = size_bytes;
    return st;
}

/**
 * @brief Zero-fill a device buffer via a chain-tracked host write.
 *
 * @p zeros must be a caller-provided zeroed buffer of at least
 * @p size_bytes (this wrapper performs no allocation) and must stay
 * valid until the enqueued write completes (the write is async).
 *
 * @param chain      Chain to order and track the zero write on.
 * @param buf        Device buffer to zero.
 * @param zeros      Caller-provided zeroed host buffer.
 * @param size_bytes Number of bytes to zero.
 * @return CVL_CL_SUCCESS or error.
 */
static inline cvl_cl_status_t zero_device_buffer_chained(cvl_cl_chain_t *chain, cvl_cl_buffer_t *buf, const void *zeros,
                                                         size_t size_bytes)
{
    if (size_bytes == 0)
        return CVL_CL_SUCCESS;

    return cvl_cl_chain_write_buffer(chain, buf, 0, size_bytes, zeros, 0, NULL, NULL);
}

/* ================================================================ */
/* Event wait-list helpers                                          */
/* ================================================================ */

/**
 * @brief Build a raw cl_event wait-list from typed events.
 *
 * @param n_wait         Number of wait events (≤ CL_MAX_WAIT_EVENTS; asserted).
 * @param wait_events    Array of typed wait events (may be NULL if n_wait == 0).
 * @param wait_list_raw  Stack buffer for raw cl_event values (must hold CL_MAX_WAIT_EVENTS).
 * @param out_n          [out] Number of raw events written.
 */
static inline void cl_prepare_wait_list(unsigned n_wait, const cvl_cl_event_t *wait_events,
                                        cl_event wait_list_raw[CL_MAX_WAIT_EVENTS], unsigned *out_n)
{
    assert(n_wait <= CL_MAX_WAIT_EVENTS);
    for (unsigned i = 0; i < n_wait; ++i)
        wait_list_raw[i] = wait_events[i].event;
    *out_n = n_wait;
}
