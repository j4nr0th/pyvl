#pragma once
/*
 * Opaque asynchronous operation handle.
 *
 * Wraps a single cl_event with ownership semantics.  The caller
 * does not touch the raw cl_event — they wait, poll, or release.
 *
 * A future that has not been initialised (zero-initialised or after
 * release) is considered "empty" and its operations are no-ops.
 *
 * Future takes ownership of the underlying cl_event: clReleaseEvent
 * is called on release/wait.  The raw event may be extracted for
 * wait-list chaining via @ref cvl_cl_future_event().
 */

#include "cvl_cl_common.h"

#include <CL/cl.h>

/* ------------------------------------------------------------------ */
/* Future handle                                                      */
/* ------------------------------------------------------------------ */

typedef struct
{
    cl_event event;
} cvl_cl_future_t;

/**
 * @brief Initialise an empty future (no pending operation).
 *
 * Safe to call on a zero-initialised struct.
 */
static inline void cvl_cl_future_init(cvl_cl_future_t *f)
{
    f->event = NULL;
}

/**
 * @brief Take ownership of a raw cl_event.
 *
 * The future will call clReleaseEvent on the event when it is
 * waited, released, or replaced.
 *
 * @param f    Future to initialise.
 * @param ev   Raw cl_event (may be NULL for an empty future).
 */
static inline void cvl_cl_future_init_from_event(cvl_cl_future_t *f, cl_event ev)
{
    f->event = ev;
}

/**
 * @brief Block until the operation completes.
 *
 * Calls clWaitForEvents.  After waiting, the event is released
 * and the future becomes empty.
 *
 * Safe to call on an empty future (no-op).
 *
 * @return CVL_CL_SUCCESS, CVL_CL_ERR_EVENT, or CVL_CL_ERR_INTERNAL.
 */
cvl_cl_status_t cvl_cl_future_wait(cvl_cl_future_t *f);

/**
 * @brief Non-blocking check whether the operation has completed.
 *
 * Uses clGetEventInfo with CL_COMMAND_EXECUTION_STATUS.
 * Returns true if status >= CL_COMPLETE or if the future is empty.
 *
 * @param f   Future to check.
 * @return true if complete (or empty), false if still running.
 */
bool cvl_cl_future_is_ready(const cvl_cl_future_t *f);

/**
 * @brief Release the underlying event and reset the future to empty.
 *
 * Safe to call on an empty future (no-op).
 */
void cvl_cl_future_release(cvl_cl_future_t *f);

/**
 * @brief Return the raw cl_event (may be NULL).
 *
 * Use this when building OpenCL wait-lists for chained operations.
 * The caller must NOT release the event — the future owns it.
 */
static inline cl_event cvl_cl_future_event(const cvl_cl_future_t *f)
{
    return f ? f->event : NULL;
}
