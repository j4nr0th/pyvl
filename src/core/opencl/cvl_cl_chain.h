#pragma once
/*
 * Command dependency chaining.
 *
 * A chain is a stream of dependent OpenCL commands on a single queue:
 * every operation enqueued through the chain automatically waits on
 * all previously chained operations, and its own completion event is
 * recorded for the next operation to wait on.  Multi-stage pipelines
 * (write → kernel → read) become a simple sequence of enqueues with
 * no explicit wait-list bookkeeping.
 *
 * Pending events accumulate up to @ref CL_MAX_WAIT_EVENTS.  When the
 * next operation would overflow the wait-list limit the pending events
 * are compacted into a single marker event (clEnqueueMarkerWithWaitList,
 * OpenCL 1.2+) that completes when all of them do.
 *
 * The queue is borrowed and must outlive the chain.  Do not mix raw
 * enqueues on the same queue between chain operations without
 * accounting for the chain's pending dependencies.
 *
 * Callers that need to synchronise on a specific operation can request
 * an owned output event (out_event); the chain keeps its own internal
 * copy either way.
 */

#include "cvl_cl_buffer.h"
#include "cvl_cl_command.h"
#include "cvl_cl_common.h"
#include "cvl_cl_kernel.h"

#include <CL/cl.h>

/* ------------------------------------------------------------------ */
/* Chain handle                                                       */
/* ------------------------------------------------------------------ */

typedef struct
{
    cl_command_queue queue;                     /**< Borrowed. */
    cvl_cl_event_t pending[CL_MAX_WAIT_EVENTS]; /**< Owned events for in-flight operations. */
    unsigned n_pending;                         /**< Live entries in pending[]. */
} cvl_cl_chain_t;

/**
 * @brief Initialise an empty chain on a queue.
 *
 * @param chain Chain to initialise.
 * @param queue Command queue (borrowed - must outlive the chain).
 */
void cvl_cl_chain_init(cvl_cl_chain_t *chain, cl_command_queue queue);

/**
 * @brief Release all pending events.  Does not wait.
 *
 * @param chain Chain to destroy (may be NULL).
 */
void cvl_cl_chain_destroy(cvl_cl_chain_t *chain);

/**
 * @brief Flush the queue (push pending commands to the device).
 */
cvl_cl_status_t cvl_cl_chain_flush(cvl_cl_chain_t *chain);

/**
 * @brief Block until all chained operations complete.
 *
 * Waits on all pending events and releases them (the chain is left
 * empty and ready for reuse).
 *
 * @return CVL_CL_SUCCESS or error.
 */
cvl_cl_status_t cvl_cl_chain_finish(cvl_cl_chain_t *chain);

/**
 * @brief Non-blocking check whether all chained operations completed.
 *
 * @return true if the chain has no pending work or all pending events
 *         are ready, false otherwise.
 */
bool cvl_cl_chain_is_ready(const cvl_cl_chain_t *chain);

/* ------------------------------------------------------------------ */
/* Chained operations                                                 */
/*                                                                     */
/* Each op waits on the chain's pending events (plus any extra         */
/* caller-supplied wait events) and records its own completion event.  */
/* If out_event is non-NULL it receives an owned event for the op      */
/* (the caller must release it); the chain keeps its own copy.         */
/* ------------------------------------------------------------------ */

cvl_cl_status_t cvl_cl_chain_write_buffer(cvl_cl_chain_t *chain, cvl_cl_buffer_t *buffer, size_t offset, size_t size,
                                          const void *host_ptr, unsigned n_wait, const cvl_cl_event_t *wait_events,
                                          cvl_cl_event_t *out_event);

cvl_cl_status_t cvl_cl_chain_read_buffer(cvl_cl_chain_t *chain, const cvl_cl_buffer_t *buffer, size_t offset,
                                         size_t size, void *host_ptr, unsigned n_wait,
                                         const cvl_cl_event_t *wait_events, cvl_cl_event_t *out_event);

cvl_cl_status_t cvl_cl_chain_copy_buffer(cvl_cl_chain_t *chain, const cvl_cl_buffer_t *src, cvl_cl_buffer_t *dst,
                                         size_t src_offset, size_t dst_offset, size_t size, unsigned n_wait,
                                         const cvl_cl_event_t *wait_events, cvl_cl_event_t *out_event);

cvl_cl_status_t cvl_cl_chain_ndrange(cvl_cl_chain_t *chain, cl_kernel kernel, unsigned dims, const size_t global_work[],
                                     const size_t local_work[], const cvl_cl_karg_t kargs[], unsigned n_wait,
                                     const cvl_cl_event_t *wait_events, cvl_cl_event_t *out_event);

/**
 * @brief Grow a device buffer asynchronously through the chain.
 *
 * Allocates a new cl_mem of @p new_capacity bytes and, when the buffer
 * has content, enqueues a chain-tracked copy of the old contents into
 * it.  The copy waits on the chain's pending events and is itself
 * recorded as a pending event, so downstream chained operations
 * automatically run after the growth completes.  The old cl_mem is
 * released after the copy is enqueued (the command retains it).
 *
 * Buffer creation itself (clCreateBuffer) is a synchronous host-side
 * API call - only the content copy is asynchronous.
 *
 * @param chain         Chain to order and track the copy on.
 * @param buf           Buffer to grow (must already have a cl_mem).
 * @param ctx           Context for creating the new cl_mem.
 * @param new_capacity  Minimum capacity in bytes (> buf->capacity).
 * @return CVL_CL_SUCCESS or error.
 */
cvl_cl_status_t cvl_cl_chain_grow_buffer(cvl_cl_chain_t *chain, cvl_cl_buffer_t *buf, cl_context ctx,
                                         size_t new_capacity);
