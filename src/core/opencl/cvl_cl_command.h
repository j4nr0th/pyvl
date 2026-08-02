#pragma once
/*
 * Command submission and event management.
 *
 * Provides typed wrappers for the common enqueue operations:
 *   - cvl_cl_ndrange        - kernel launch with typed args
 *   - cvl_cl_write_buffer   - host → device
 *   - cvl_cl_read_buffer    - device → host
 *   - cvl_cl_copy_buffer    - device → device
 *   - cvl_cl_finish / flush - synchronisation
 *
 * All operations accept optional event wait lists and produce an
 * optional output event, following OpenCL's async model.
 */

#include "cvl_cl_buffer.h"
#include "cvl_cl_common.h"
#include "cvl_cl_ctx.h"
#include "cvl_cl_kernel.h"

#include <CL/cl.h>

/* ------------------------------------------------------------------ */
/* Event management                                                   */
/* ------------------------------------------------------------------ */

struct cvl_cl_event_t
{
    cl_event event;
    bool owns; /**< If true, clReleaseEvent will be called on destroy. */
};

/**
 * @brief Wrap a raw cl_event without taking ownership.
 *
 * The caller must keep the cl_event valid while this wrapper is used.
 * No clReleaseEvent is called on destroy.
 */
cvl_cl_event_t cvl_cl_event_wrap(cl_event raw);

/**
 * @brief Take ownership of a raw cl_event.
 *
 * clReleaseEvent WILL be called on @ref cvl_cl_event_release.
 */
cvl_cl_event_t cvl_cl_event_take(cl_event raw);

/**
 * @brief Wait for an event to complete.
 *
 * @param event Event to wait on (may be NULL / empty).
 * @return CVL_CL_SUCCESS or error.
 */
cvl_cl_status_t cvl_cl_event_wait(const cvl_cl_event_t *event);

/**
 * @brief Release an event (only if ownership was taken).
 *
 * Safe to call on an event created via @ref cvl_cl_event_wrap as well
 * (no-op for non-owned events).
 *
 * @param event Event to release (may be NULL).
 */
void cvl_cl_event_release(cvl_cl_event_t *event);

/** @brief Return the raw cl_event. */
static inline cl_event cvl_cl_event_event(const cvl_cl_event_t *ev)
{
    return ev ? ev->event : NULL;
}

/* ------------------------------------------------------------------ */
/* NDRange kernel launch                                              */
/* ------------------------------------------------------------------ */

/**
 * @brief Enqueue a kernel with typed arguments (via cpyutl-style descriptors).
 *
 * Example:
 * @code
 *   cvl_cl_status_t st = cvl_cl_ndrange(queue, kernel,
 *       1, (size_t[]){1024}, (size_t[]){256},
 *       (cvl_cl_karg_t[]){
 *           {.type = CVL_CL_KARG_BUFFER, .index = 0, .mem = cvl_cl_buffer_mem(&buf)},
 *           {.type = CVL_CL_KARG_SCALAR_UINT, .index = 1, .scalar_uint = n},
 *           {},
 *       },
 *       0, NULL, NULL);
 * @endcode
 *
 * @param queue      Target queue.
 * @param kernel     Kernel to execute.
 * @param dims       Work dimensions (1, 2, or 3).
 * @param global     Global work size [dims].
 * @param local      Local work size [dims], or NULL to let the runtime choose.
 * @param kargs      NULL-terminated typed argument descriptor array (may be NULL for no args).
 * @param n_wait     Number of events to wait on before executing.
 * @param wait_events Wait-list (length @p n_wait).
 * @param out_event  Optional output event (caller must release).
 * @return CVL_CL_SUCCESS or error.
 */
cvl_cl_status_t cvl_cl_ndrange(cvl_cl_queue_t *queue, cvl_cl_kernel_t *kernel, unsigned dims,
                               const size_t global_work[], const size_t local_work[], const cvl_cl_karg_t kargs[],
                               unsigned n_wait, const cvl_cl_event_t *wait_events, cvl_cl_event_t *out_event);

/* ------------------------------------------------------------------ */
/* Buffer transfers                                                   */
/* ------------------------------------------------------------------ */

/**
 * @brief Enqueue a host → device write (async, non-blocking).
 *
 * @param queue      Target queue.
 * @param buffer     Device buffer.
 * @param offset     Byte offset into the device buffer.
 * @param size       Number of bytes to write.
 * @param host_ptr   Source data on the host.
 * @param n_wait     Wait-list size.
 * @param wait_events Optional events to wait on.
 * @param out_event  Optional output event.
 * @return CVL_CL_SUCCESS or error.
 */
cvl_cl_status_t cvl_cl_write_buffer(cvl_cl_queue_t *queue, cvl_cl_buffer_t *buffer, size_t offset, size_t size,
                                    const void *host_ptr, unsigned n_wait, const cvl_cl_event_t *wait_events,
                                    cvl_cl_event_t *out_event);

/**
 * @brief Enqueue a device → host read (async, non-blocking).
 *
 * @param queue      Target queue.
 * @param buffer     Device buffer.
 * @param offset     Byte offset into the device buffer.
 * @param size       Number of bytes to read.
 * @param host_ptr   Destination buffer on the host (must be at least @p size bytes).
 * @param n_wait     Wait-list size.
 * @param wait_events Optional events to wait on.
 * @param out_event  Optional output event.
 * @return CVL_CL_SUCCESS or error.
 */
cvl_cl_status_t cvl_cl_read_buffer(cvl_cl_queue_t *queue, const cvl_cl_buffer_t *buffer, size_t offset, size_t size,
                                   void *host_ptr, unsigned n_wait, const cvl_cl_event_t *wait_events,
                                   cvl_cl_event_t *out_event);

/**
 * @brief Enqueue a device → device copy (async, non-blocking).
 *
 * @param queue      Target queue.
 * @param src        Source buffer.
 * @param dst        Destination buffer.
 * @param src_offset Source byte offset.
 * @param dst_offset Destination byte offset.
 * @param size       Number of bytes to copy.
 * @param n_wait     Wait-list size.
 * @param wait_events Optional events to wait on.
 * @param out_event  Optional output event.
 * @return CVL_CL_SUCCESS or error.
 */
cvl_cl_status_t cvl_cl_copy_buffer(cvl_cl_queue_t *queue, const cvl_cl_buffer_t *src, cvl_cl_buffer_t *dst,
                                   size_t src_offset, size_t dst_offset, size_t size, unsigned n_wait,
                                   const cvl_cl_event_t *wait_events, cvl_cl_event_t *out_event);

/* ------------------------------------------------------------------ */
/* Synchronisation                                                   */
/* ------------------------------------------------------------------ */

/**
 * @brief Block until all previously enqueued commands complete.
 *
 * Wrapper around clFinish.
 */
cvl_cl_status_t cvl_cl_finish(cvl_cl_queue_t *queue);

/**
 * @brief Flush the queue (push pending commands to the device).
 *
 * Wrapper around clFlush.
 */
cvl_cl_status_t cvl_cl_flush(cvl_cl_queue_t *queue);

/** @brief
 *  Convenience: wait for N events from a raw cl_event array (host-side sync).
 *
 *  Calls clWaitForEvents.
 */
cvl_cl_status_t cvl_cl_wait_for_events(unsigned n_events, const cl_event event_list[]);
