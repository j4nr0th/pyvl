#pragma once
/*
 * Typed device buffer with optional FP32 host-side conversion.
 *
 * Manages a device-side cl_mem buffer sized in "elements" (one real3_t
 * per element) and a persistent host staging area for FP32 ↔ FP64
 * conversion.
 *
 * FP64 mode (default / CVL_CL_PRECISION_FP64):
 *   Device stores sizeof(real3_t) = 24 bytes per element.
 *   write_async copies directly from the user's real3_t[].
 *   read_async copies directly to the user's real3_t[].
 *
 * FP32 mode (CVL_CL_PRECISION_FP32):
 *   Device stores 3 × sizeof(float) = 12 bytes per element.
 *   write_async converts host doubles → staging floats, then enqueues write.
 *   read_async enqueues read into staging floats; conversion back to
 *   doubles happens in cvl_cl_staging_buffer_read_finish() after the
 *   future completes.
 *
 * Buffer grows on demand via reserve(); never shrinks.
 */

#include "../common.h"
#include "cvl_cl_buffer.h"
#include "cvl_cl_common.h"
#include "cvl_cl_ctx.h"
#include "cvl_cl_future.h"

#include <CL/cl.h>

/* ------------------------------------------------------------------ */
/* Staging buffer handle                                               */
/* ------------------------------------------------------------------ */

typedef struct
{
    cvl_cl_buffer_t device;    /**< Device-side buffer. */
    float *host_fp32;          /**< Host staging for FP32 (NULL in FP64 mode). */
    size_t capacity_elements;  /**< Current capacity in real3_t elements. */
    size_t element_size_bytes; /**< 24 for FP64, 12 for FP32. */
    cvl_cl_precision_t precision;
    bool unified_memory;
} cvl_cl_staging_buffer_t;

/**
 * @brief Initialise a staging buffer.
 *
 * Does not allocate anything.  First allocation happens on reserve().
 *
 * @param buf             Uninitialised buffer struct.
 * @param ctx             Context (borrowed).
 * @param precision       FP32 or FP64.
 * @param unified_memory  Hint from CL_DEVICE_HOST_UNIFIED_MEMORY.
 * @return CVL_CL_SUCCESS.
 */
cvl_cl_status_t cvl_cl_staging_buffer_init(cvl_cl_staging_buffer_t *buf, const cvl_cl_ctx_t *ctx,
                                           cvl_cl_precision_t precision, bool unified_memory);

/**
 * @brief Grow the buffer to at least @p n_elements capacity.
 *
 * Grows the device buffer (and host staging, if FP32) on demand.
 * Existing content is preserved via clEnqueueCopyBuffer.
 *
 * @param buf         Staging buffer.
 * @param ctx         Context.
 * @param queue       Queue for copy (may be NULL if not growing).
 * @param n_elements  Minimum number of real3_t elements.
 * @return CVL_CL_SUCCESS or error.
 */
cvl_cl_status_t cvl_cl_staging_buffer_reserve(cvl_cl_staging_buffer_t *buf, const cvl_cl_ctx_t *ctx,
                                              cvl_cl_queue_t *queue, size_t n_elements);

/**
 * @brief Enqueue an asynchronous host → device write.
 *
 * For FP64: writes directly from @p host_data.
 * For FP32: converts to float staging then enqueues write.
 *
 * @param buf              Staging buffer.
 * @param queue            Queue.
 * @param host_data        Host real3_t array (n_elements).
 * @param n_elements       Number of elements to write.
 * @param dst_offset_el    Offset into device buffer (in elements).
 * @param out_future       Filled with future for the write event (caller may wait/release).
 * @return CVL_CL_SUCCESS or error.
 */
cvl_cl_status_t cvl_cl_staging_buffer_write_async(cvl_cl_staging_buffer_t *buf, cvl_cl_queue_t *queue,
                                                  const real3_t *host_data, size_t n_elements, size_t dst_offset_el,
                                                  cvl_cl_future_t *out_future);

/**
 * @brief Enqueue an asynchronous device → host read.
 *
 * For FP64: reads directly into @p host_data.
 * For FP32: reads into internal float staging; call
 *           cvl_cl_staging_buffer_read_finish() AFTER waiting
 *           on @p out_future to convert to double.
 *
 * @param buf              Staging buffer.
 * @param queue            Queue.
 * @param host_data        Host output array (n_elements).
 * @param n_elements       Number of elements to read.
 * @param src_offset_el    Offset into device buffer (in elements).
 * @param out_future       Filled with future for the read event.
 * @return CVL_CL_SUCCESS or error.
 */
cvl_cl_status_t cvl_cl_staging_buffer_read_async(cvl_cl_staging_buffer_t *buf, cvl_cl_queue_t *queue,
                                                 real3_t *host_data, size_t n_elements, size_t src_offset_el,
                                                 cvl_cl_future_t *out_future);

/**
 * @brief Complete a pending FP32 read: convert internal float staging
 *        back to the user's real3_t array.
 *
 * Must be called AFTER the future returned by read_async() has
 * completed (cvl_cl_future_wait).  Safe no-op in FP64 mode.
 *
 * @param buf            Staging buffer.
 * @param host_data      Host output array (same pointer passed to read_async).
 * @param n_elements     Number of elements.
 * @param src_offset_el  Same offset used in read_async.
 * @return CVL_CL_SUCCESS.
 */
cvl_cl_status_t cvl_cl_staging_buffer_read_finish(cvl_cl_staging_buffer_t *buf, real3_t *host_data, size_t n_elements,
                                                  size_t src_offset_el);

/**
 * @brief Convenience: asynchronous read, then wait + finish.
 *
 * Combines read_async(), cvl_cl_future_wait(), and read_finish().
 *
 * @param buf             Staging buffer.
 * @param queue           Queue.
 * @param host_data       Host output array.
 * @param n_elements      Number of elements.
 * @param src_offset_el   Offset into device buffer.
 * @return CVL_CL_SUCCESS or error.
 */
cvl_cl_status_t cvl_cl_staging_buffer_read_and_wait(cvl_cl_staging_buffer_t *buf, cvl_cl_queue_t *queue,
                                                    real3_t *host_data, size_t n_elements, size_t src_offset_el);

/**
 * @brief Destroy the staging buffer, releasing device memory and host staging.
 */
void cvl_cl_staging_buffer_destroy(cvl_cl_staging_buffer_t *buf);
