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
 * All operations operate on raw cl_command_queue / cl_kernel handles
 * and accept optional event wait lists, producing an optional output
 * event - following OpenCL's async model.
 *
 * The cvl_cl_event_t wrapper tracks ownership of a cl_event.  For
 * dependency chaining across multiple operations see cvl_cl_chain.h.
 */

#include "cvl_cl_buffer.h"
#include "cvl_cl_common.h"
#include "cvl_cl_kernel.h"

#include <CL/cl.h>

/* ------------------------------------------------------------------ */
/* Event management                                                   */
/* ------------------------------------------------------------------ */

typedef struct
{
    cl_event event;
    bool owns; /**< If true, clReleaseEvent will be called on release. */
} cvl_cl_event_t;

/**
 * @brief Wrap a raw cl_event without taking ownership.
 *
 * The caller must keep the cl_event valid while this wrapper is used.
 * No clReleaseEvent is called on release.
 */
cvl_cl_event_t cvl_cl_event_wrap(cl_event raw);

/**
 * @brief Take ownership of a raw cl_event.
 *
 * clReleaseEvent WILL be called on @ref cvl_cl_event_release.
 */
cvl_cl_event_t cvl_cl_event_take(cl_event raw);

/**
 * @brief Block until the event completes.
 *
 * @param event Event to wait on (may be NULL / empty - no-op).
 * @return CVL_CL_SUCCESS or error.
 */
cvl_cl_status_t cvl_cl_event_wait(const cvl_cl_event_t *event);

/**
 * @brief Non-blocking completion check.
 *
 * Returns true if the event has completed (CL_COMPLETE or later), if
 * the event is empty, or if the status cannot be determined (the error
 * will surface on wait).
 *
 * @param event Event to check (may be NULL / empty).
 * @return true if complete (or empty), false if still running.
 */
bool cvl_cl_event_is_ready(const cvl_cl_event_t *event);

/**
 * @brief Release the underlying cl_event if ownership was taken.
 *
 * Safe to call on an event created via @ref cvl_cl_event_wrap as well
 * (no-op for non-owned events), and on NULL / empty events.
 *
 * @param event Event to release (may be NULL).
 */
void cvl_cl_event_release(cvl_cl_event_t *event);

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
 *           {.type = CVL_CL_KARG_BUFFER, .index = 0, .mem = buf.mem},
 *           {.type = CVL_CL_KARG_SCALAR_UINT, .index = 1, .scalar_uint = n},
 *           {},
 *       },
 *       0, NULL, NULL);
 * @endcode
 *
 * @param queue       Target queue.
 * @param kernel      Kernel to execute.
 * @param dims        Work dimensions (1, 2, or 3).
 * @param global_work Global work size [dims].
 * @param local_work  Local work size [dims], or NULL to let the runtime choose.
 * @param kargs       NULL-terminated typed argument descriptor array (may be NULL for no args).
 * @param n_wait      Number of events to wait on before executing.
 * @param wait_events Wait-list (length @p n_wait).
 * @param out_event   Optional output event (filled with an owned event; caller must release).
 * @return CVL_CL_SUCCESS or error.
 */
cvl_cl_status_t cvl_cl_ndrange(cl_command_queue queue, cl_kernel kernel, unsigned dims, const size_t global_work[],
                               const size_t local_work[], const cvl_cl_karg_t kargs[], unsigned n_wait,
                               const cvl_cl_event_t *wait_events, cvl_cl_event_t *out_event);

/* ------------------------------------------------------------------ */
/* Buffer transfers                                                   */
/* ------------------------------------------------------------------ */

/**
 * @brief Enqueue a host → device write (async, non-blocking).
 *
 * @param queue       Target queue.
 * @param buffer      Device buffer.
 * @param offset      Byte offset into the device buffer.
 * @param size        Number of bytes to write.
 * @param host_ptr    Source data on the host.
 * @param n_wait      Wait-list size.
 * @param wait_events Optional events to wait on.
 * @param out_event   Optional output event (owned; caller must release).
 * @return CVL_CL_SUCCESS or error.
 */
cvl_cl_status_t cvl_cl_write_buffer(cl_command_queue queue, cvl_cl_buffer_t *buffer, size_t offset, size_t size,
                                    const void *host_ptr, unsigned n_wait, const cvl_cl_event_t *wait_events,
                                    cvl_cl_event_t *out_event);

/**
 * @brief Enqueue a device → host read (async, non-blocking).
 *
 * @param queue       Target queue.
 * @param buffer      Device buffer.
 * @param offset      Byte offset into the device buffer.
 * @param size        Number of bytes to read.
 * @param host_ptr    Destination buffer on the host (must be at least @p size bytes).
 * @param n_wait      Wait-list size.
 * @param wait_events Optional events to wait on.
 * @param out_event   Optional output event (owned; caller must release).
 * @return CVL_CL_SUCCESS or error.
 */
cvl_cl_status_t cvl_cl_read_buffer(cl_command_queue queue, const cvl_cl_buffer_t *buffer, size_t offset, size_t size,
                                   void *host_ptr, unsigned n_wait, const cvl_cl_event_t *wait_events,
                                   cvl_cl_event_t *out_event);

/**
 * @brief Enqueue a device → device copy (async, non-blocking).
 *
 * @param queue       Target queue.
 * @param src         Source buffer.
 * @param dst         Destination buffer.
 * @param src_offset  Source byte offset.
 * @param dst_offset  Destination byte offset.
 * @param size        Number of bytes to copy.
 * @param n_wait      Wait-list size.
 * @param wait_events Optional events to wait on.
 * @param out_event   Optional output event (owned; caller must release).
 * @return CVL_CL_SUCCESS or error.
 */
cvl_cl_status_t cvl_cl_copy_buffer(cl_command_queue queue, const cvl_cl_buffer_t *src, cvl_cl_buffer_t *dst,
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
cvl_cl_status_t cvl_cl_finish(cl_command_queue queue);

/**
 * @brief Flush the queue (push pending commands to the device).
 *
 * Wrapper around clFlush.
 */
cvl_cl_status_t cvl_cl_flush(cl_command_queue queue);

/**
 * @brief Convenience: wait for N raw cl_event handles (host-side sync).
 *
 * Calls clWaitForEvents.
 */
cvl_cl_status_t cvl_cl_wait_for_events(unsigned n_events, const cl_event event_list[]);
