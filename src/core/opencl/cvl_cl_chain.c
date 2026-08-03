/*
 * Command dependency chaining.
 *
 * Implements cvl_cl_chain_t: a stream of dependent OpenCL commands on a
 * single queue.  Every chained operation waits on all of the chain's
 * pending events (plus any caller-supplied extra wait events, pending
 * first) and records its own completion event as the new pending event.
 * When the pending list would overflow CL_MAX_WAIT_EVENTS the pending
 * events are compacted into a single marker event
 * (clEnqueueMarkerWithWaitList, OpenCL 1.2+) that completes when all
 * of them do.
 *
 * The queue is borrowed and must outlive the chain.  This module never
 * allocates: pending events live in a fixed-size array inside the chain.
 */

#include "cvl_cl_chain.h"
#include "cvl_cl_command.h"
#include "cvl_cl_kernel.h"

#include <assert.h>
#include <string.h>

#include <CL/cl.h>

/* ------------------------------------------------------------------ */
/* File-static helpers                                                */
/* ------------------------------------------------------------------ */

/**
 * @brief Assemble the raw wait list for a chained operation.
 *
 * The chain's pending events come first, followed by the caller's extra
 * wait events.  If the combined list would exceed CL_MAX_WAIT_EVENTS
 * (or the pending list is already full), the pending events are first
 * compacted into a single marker event that completes when all of them
 * do; the old pending events are released and the marker becomes the
 * sole pending event.
 *
 * @param chain       Chain whose pending events form the start of the list.
 * @param n_wait      Number of caller-supplied extra wait events.
 * @param wait_events Caller-supplied wait events (may be NULL if n_wait == 0).
 * @param raw         Output buffer (must hold CL_MAX_WAIT_EVENTS entries).
 * @param out_n_raw   [out] Number of raw events written.
 * @return CVL_CL_SUCCESS or error.
 */
static cvl_cl_status_t chain_build_wait_list(cvl_cl_chain_t *chain, unsigned n_wait, const cvl_cl_event_t *wait_events,
                                             cl_event raw[CL_MAX_WAIT_EVENTS], unsigned *out_n_raw)
{
    assert(chain && chain->queue);
    assert(n_wait == 0 || wait_events != NULL);

    if (chain->n_pending + n_wait > CL_MAX_WAIT_EVENTS || chain->n_pending == CL_MAX_WAIT_EVENTS)
    {
        cl_event raw_pending[CL_MAX_WAIT_EVENTS];
        for (unsigned i = 0; i < chain->n_pending; ++i)
            raw_pending[i] = chain->pending[i].event;

        cl_event marker = NULL;
        const cl_int err = clEnqueueMarkerWithWaitList(chain->queue, chain->n_pending, raw_pending, &marker);
        if (err != CL_SUCCESS)
            return cvl_cl_status_from_cl_int(err);

        for (unsigned i = 0; i < chain->n_pending; ++i)
            cvl_cl_event_release(&chain->pending[i]);
        chain->pending[0] = cvl_cl_event_take(marker);
        chain->n_pending = 1;

        /* After compaction the wait list is 1 (the marker) plus the extras. */
        if (1 + n_wait > CL_MAX_WAIT_EVENTS)
            return CVL_CL_ERR_INVALID_PARAM;
    }

    unsigned n_raw = 0;
    for (unsigned i = 0; i < chain->n_pending; ++i)
        raw[n_raw++] = chain->pending[i].event;
    for (unsigned i = 0; i < n_wait; ++i)
        raw[n_raw++] = wait_events[i].event;
    *out_n_raw = n_raw;
    return CVL_CL_SUCCESS;
}

/**
 * @brief Record the completion event of a successful enqueue in the chain.
 *
 * The chain takes ownership of @p raw_out and appends it to its pending
 * list.  If @p out_event is non-NULL the event is additionally retained
 * so the caller receives an owned copy (the caller must release it).
 *
 * @param chain     Chain to extend.
 * @param raw_out   Raw cl_event returned by the enqueue (owned).
 * @param out_event Optional caller-owned output event.
 */
static void chain_capture_event(cvl_cl_chain_t *chain, cl_event raw_out, cvl_cl_event_t *out_event)
{
    assert(chain);
    assert(chain->n_pending < CL_MAX_WAIT_EVENTS);
    chain->pending[chain->n_pending++] = cvl_cl_event_take(raw_out);
    if (out_event != NULL)
    {
        clRetainEvent(raw_out);
        *out_event = cvl_cl_event_take(raw_out);
    }
}

/* ------------------------------------------------------------------ */
/* Lifecycle                                                          */
/* ------------------------------------------------------------------ */

void cvl_cl_chain_init(cvl_cl_chain_t *chain, cl_command_queue queue)
{
    assert(chain);
    *chain = (cvl_cl_chain_t){.queue = queue};
}

void cvl_cl_chain_destroy(cvl_cl_chain_t *chain)
{
    if (!chain)
        return;
    for (unsigned i = 0; i < chain->n_pending; ++i)
        cvl_cl_event_release(&chain->pending[i]);
    memset(chain, 0, sizeof(*chain));
}

/* ------------------------------------------------------------------ */
/* Synchronisation                                                   */
/* ------------------------------------------------------------------ */

cvl_cl_status_t cvl_cl_chain_flush(cvl_cl_chain_t *chain)
{
    assert(chain && chain->queue);
    const cl_int err = clFlush(chain->queue);
    if (err != CL_SUCCESS)
        return cvl_cl_status_from_cl_int(err);
    return CVL_CL_SUCCESS;
}

cvl_cl_status_t cvl_cl_chain_finish(cvl_cl_chain_t *chain)
{
    assert(chain && chain->queue);
    if (chain->n_pending == 0)
        return CVL_CL_SUCCESS;

    cl_event raw[CL_MAX_WAIT_EVENTS];
    for (unsigned i = 0; i < chain->n_pending; ++i)
        raw[i] = chain->pending[i].event;

    const cl_int err = clWaitForEvents(chain->n_pending, raw);

    /* Release regardless of the wait result; the chain is left empty. */
    for (unsigned i = 0; i < chain->n_pending; ++i)
        cvl_cl_event_release(&chain->pending[i]);
    chain->n_pending = 0;

    if (err != CL_SUCCESS)
        return cvl_cl_status_from_cl_int(err);
    return CVL_CL_SUCCESS;
}

bool cvl_cl_chain_is_ready(const cvl_cl_chain_t *chain)
{
    assert(chain);
    for (unsigned i = 0; i < chain->n_pending; ++i)
    {
        if (!cvl_cl_event_is_ready(&chain->pending[i]))
            return false;
    }
    return true;
}

/* ------------------------------------------------------------------ */
/* Chained operations                                                 */
/* ------------------------------------------------------------------ */

cvl_cl_status_t cvl_cl_chain_write_buffer(cvl_cl_chain_t *chain, cvl_cl_buffer_t *buffer, size_t offset, size_t size,
                                          const void *host_ptr, unsigned n_wait, const cvl_cl_event_t *wait_events,
                                          cvl_cl_event_t *out_event)
{
    assert(chain && chain->queue);
    assert(buffer && buffer->mem);
    assert(size == 0 || host_ptr != NULL);

    cl_event raw[CL_MAX_WAIT_EVENTS];
    unsigned n_raw = 0;
    {
        cvl_cl_status_t s = chain_build_wait_list(chain, n_wait, wait_events, raw, &n_raw);
        if (s != CVL_CL_SUCCESS)
            return s;
    }

    cl_event raw_out = NULL;
    const cl_int err = clEnqueueWriteBuffer(chain->queue, buffer->mem, CL_FALSE, offset, size, host_ptr, n_raw,
                                            (n_raw > 0) ? raw : NULL, &raw_out);
    if (err != CL_SUCCESS)
        return cvl_cl_status_from_cl_int(err);

    chain_capture_event(chain, raw_out, out_event);
    return CVL_CL_SUCCESS;
}

cvl_cl_status_t cvl_cl_chain_read_buffer(cvl_cl_chain_t *chain, const cvl_cl_buffer_t *buffer, size_t offset,
                                         size_t size, void *host_ptr, unsigned n_wait,
                                         const cvl_cl_event_t *wait_events, cvl_cl_event_t *out_event)
{
    assert(chain && chain->queue);
    assert(buffer && buffer->mem);
    assert(size == 0 || host_ptr != NULL);

    cl_event raw[CL_MAX_WAIT_EVENTS];
    unsigned n_raw = 0;
    {
        cvl_cl_status_t s = chain_build_wait_list(chain, n_wait, wait_events, raw, &n_raw);
        if (s != CVL_CL_SUCCESS)
            return s;
    }

    cl_event raw_out = NULL;
    const cl_int err = clEnqueueReadBuffer(chain->queue, buffer->mem, CL_FALSE, offset, size, host_ptr, n_raw,
                                           (n_raw > 0) ? raw : NULL, &raw_out);
    if (err != CL_SUCCESS)
        return cvl_cl_status_from_cl_int(err);

    chain_capture_event(chain, raw_out, out_event);
    return CVL_CL_SUCCESS;
}

cvl_cl_status_t cvl_cl_chain_copy_buffer(cvl_cl_chain_t *chain, const cvl_cl_buffer_t *src, cvl_cl_buffer_t *dst,
                                         size_t src_offset, size_t dst_offset, size_t size, unsigned n_wait,
                                         const cvl_cl_event_t *wait_events, cvl_cl_event_t *out_event)
{
    assert(chain && chain->queue);
    assert(src && src->mem);
    assert(dst && dst->mem);

    cl_event raw[CL_MAX_WAIT_EVENTS];
    unsigned n_raw = 0;
    {
        cvl_cl_status_t s = chain_build_wait_list(chain, n_wait, wait_events, raw, &n_raw);
        if (s != CVL_CL_SUCCESS)
            return s;
    }

    cl_event raw_out = NULL;
    const cl_int err = clEnqueueCopyBuffer(chain->queue, src->mem, dst->mem, src_offset, dst_offset, size, n_raw,
                                           (n_raw > 0) ? raw : NULL, &raw_out);
    if (err != CL_SUCCESS)
        return cvl_cl_status_from_cl_int(err);

    chain_capture_event(chain, raw_out, out_event);
    return CVL_CL_SUCCESS;
}

cvl_cl_status_t cvl_cl_chain_ndrange(cvl_cl_chain_t *chain, cl_kernel kernel, unsigned dims, const size_t global_work[],
                                     const size_t local_work[], const cvl_cl_karg_t kargs[], unsigned n_wait,
                                     const cvl_cl_event_t *wait_events, cvl_cl_event_t *out_event)
{
    assert(chain && chain->queue);
    assert(kernel && global_work);
    assert(dims >= 1 && dims <= 3);

    if (kargs != NULL)
    {
        cvl_cl_status_t s = cvl_cl_kernel_set_args(kernel, kargs);
        if (s != CVL_CL_SUCCESS)
            return s;
    }

    cl_event raw[CL_MAX_WAIT_EVENTS];
    unsigned n_raw = 0;
    {
        cvl_cl_status_t s = chain_build_wait_list(chain, n_wait, wait_events, raw, &n_raw);
        if (s != CVL_CL_SUCCESS)
            return s;
    }

    cl_event raw_out = NULL;
    const cl_int err = clEnqueueNDRangeKernel(chain->queue, kernel, dims, NULL, global_work, local_work, n_raw,
                                              (n_raw > 0) ? raw : NULL, &raw_out);
    if (err != CL_SUCCESS)
        return cvl_cl_status_from_cl_int(err);

    chain_capture_event(chain, raw_out, out_event);
    return CVL_CL_SUCCESS;
}

cvl_cl_status_t cvl_cl_chain_grow_buffer(cvl_cl_chain_t *chain, cvl_cl_buffer_t *buf, cl_context ctx,
                                         size_t new_capacity)
{
    assert(chain && chain->queue);
    assert(buf && buf->mem);
    assert(new_capacity > buf->capacity);

    /* Create the new buffer (synchronous host-side API call). */
    cl_int err;
    cl_mem new_mem = clCreateBuffer(ctx, cvl_cl_buffer_access_to_flags(buf->access), new_capacity, NULL, &err);
    if (err != CL_SUCCESS)
        return cvl_cl_status_from_cl_int(err);

    if (buf->size > 0)
    {
        /* Chain-tracked copy of the old contents into the new buffer.
         * Waits on the chain's pending events; recorded as pending so
         * downstream ops wait for the growth to finish. */
        cl_event raw[CL_MAX_WAIT_EVENTS];
        unsigned n_raw = 0;
        cvl_cl_status_t s = chain_build_wait_list(chain, 0, NULL, raw, &n_raw);
        if (s != CVL_CL_SUCCESS)
        {
            clReleaseMemObject(new_mem);
            return s;
        }

        cl_event raw_out = NULL;
        err = clEnqueueCopyBuffer(chain->queue, buf->mem, new_mem, 0, 0, buf->size, n_raw, (n_raw > 0) ? raw : NULL,
                                  &raw_out);
        if (err != CL_SUCCESS)
        {
            clReleaseMemObject(new_mem);
            return cvl_cl_status_from_cl_int(err);
        }

        chain_capture_event(chain, raw_out, NULL);
    }

    /* Release the old buffer (the copy command retains it until done). */
    clReleaseMemObject(buf->mem);
    buf->mem = new_mem;
    buf->capacity = new_capacity;
    return CVL_CL_SUCCESS;
}
