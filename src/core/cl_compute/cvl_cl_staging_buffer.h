#pragma once
/*
 * Typed device buffer with optional FP32 host-side conversion.
 *
 * Manages a device-side cl_mem buffer sized in "elements" (one real3_t
 * per element).  All enqueues go through a cvl_cl_chain_t so the
 * caller controls dependency ordering; the buffer itself performs no
 * host allocation.
 *
 * FP64 mode (default / CVL_CL_PRECISION_FP64):
 *   Device stores sizeof(real3_t) = 24 bytes per element.
 *   write_async copies directly from the user's real3_t[].
 *   read_async copies directly to the user's real3_t[].
 *
 * FP32 mode (CVL_CL_PRECISION_FP32):
 *   Device stores 3 × sizeof(float) = 12 bytes per element.
 *   write_async converts host doubles → caller's float scratch, then
 *   enqueues the write.  read_async enqueues the read into the float
 *   scratch; conversion back to doubles happens in
 *   cvl_cl_staging_buffer_read_finish() AFTER the chain is finished.
 *
 * The float scratch is caller-provided (typically carved from a work
 * buffer) and must hold at least 3 * n_elements floats for the
 * operation in flight.
 *
 * The device buffer grows on demand via reserve(); never shrinks.
 */

#include "../common.h"
#include "../opencl/cvl_cl_buffer.h"
#include "../opencl/cvl_cl_chain.h"
#include "../opencl/cvl_cl_common.h"

#include <CL/cl.h>

/* ------------------------------------------------------------------ */
/* Staging buffer handle                                               */
/* ------------------------------------------------------------------ */

typedef struct
{
    cvl_cl_buffer_t device;    /**< Device-side buffer. */
    size_t capacity_elements;  /**< Current capacity in real3_t elements. */
    size_t element_size_bytes; /**< 24 for FP64, 12 for FP32. */
    cvl_cl_precision_t precision;
} cvl_cl_staging_buffer_t;

/**
 * @brief Initialise a staging buffer (zero state, no allocation).
 *
 * @param buf       Uninitialised buffer struct.
 * @param precision FP32 or FP64.
 * @return CVL_CL_SUCCESS.
 */
cvl_cl_status_t cvl_cl_staging_buffer_init(cvl_cl_staging_buffer_t *buf, cvl_cl_precision_t precision);

/**
 * @brief Grow the buffer to at least @p n_elements capacity.
 *
 * Grows the device buffer on demand; existing content is preserved
 * via clEnqueueCopyBuffer.
 *
 * @param buf         Staging buffer.
 * @param ctx         Context.
 * @param queue       Queue for the grow-copy (may be NULL when growing an empty buffer).
 * @param n_elements  Minimum number of real3_t elements.
 * @return CVL_CL_SUCCESS or error.
 */
cvl_cl_status_t cvl_cl_staging_buffer_reserve(cvl_cl_staging_buffer_t *buf, cl_context ctx, cl_command_queue queue,
                                              size_t n_elements);

/**
 * @brief Enqueue an asynchronous host → device write through a chain.
 *
 * FP64: writes directly from @p host_data.  FP32: converts to
 * @p scratch_f32 (≥ 3·n_elements floats, NULL in FP64 mode) then
 * enqueues the write.
 *
 * @param buf            Staging buffer.
 * @param chain          Chain to enqueue the write on (records the op's event).
 * @param host_data      Host real3_t array (n_elements).
 * @param scratch_f32    Caller-provided float scratch (FP32 only).
 * @param n_elements     Number of elements to write.
 * @param dst_offset_el  Offset into device buffer (in elements).
 * @param out_event      Optional owned event for this op (caller must release), may be NULL.
 * @return CVL_CL_SUCCESS or error.
 */
cvl_cl_status_t cvl_cl_staging_buffer_write_async(cvl_cl_staging_buffer_t *buf, cvl_cl_chain_t *chain,
                                                  const real3_t *host_data, float *scratch_f32, size_t n_elements,
                                                  size_t dst_offset_el, cvl_cl_event_t *out_event);

/**
 * @brief Enqueue an asynchronous device → host read through a chain.
 *
 * FP64: reads directly into @p host_data.  FP32: reads into
 * @p scratch_f32; call cvl_cl_staging_buffer_read_finish() AFTER the
 * chain completes to convert back to doubles.
 *
 * @param buf            Staging buffer.
 * @param chain          Chain to enqueue the read on.
 * @param host_data      Host output array (n_elements).
 * @param scratch_f32    Caller-provided float scratch (FP32 only).
 * @param n_elements     Number of elements to read.
 * @param src_offset_el  Offset into device buffer (in elements).
 * @param out_event      Optional owned event for this op (caller must release), may be NULL.
 * @return CVL_CL_SUCCESS or error.
 */
cvl_cl_status_t cvl_cl_staging_buffer_read_async(cvl_cl_staging_buffer_t *buf, cvl_cl_chain_t *chain,
                                                 real3_t *host_data, float *scratch_f32, size_t n_elements,
                                                 size_t src_offset_el, cvl_cl_event_t *out_event);

/**
 * @brief Complete a pending FP32 read: convert the float scratch back
 *        to the user's real3_t array.
 *
 * Must be called AFTER the chain used for read_async() has finished.
 * Safe no-op in FP64 mode.
 *
 * @param buf            Staging buffer.
 * @param host_data      Host output array (same pointer passed to read_async).
 * @param scratch_f32    Float scratch used by read_async.
 * @param n_elements     Number of elements.
 * @return CVL_CL_SUCCESS.
 */
cvl_cl_status_t cvl_cl_staging_buffer_read_finish(cvl_cl_staging_buffer_t *buf, real3_t *host_data,
                                                  const float *scratch_f32, size_t n_elements);

/**
 * @brief Convenience: read through a chain, finish the chain, convert.
 *
 * Combines read_async(), cvl_cl_chain_finish(), and read_finish().
 *
 * @param buf            Staging buffer.
 * @param chain          Chain to enqueue the read on (finished afterwards).
 * @param host_data      Host output array.
 * @param scratch_f32    Float scratch (FP32 only).
 * @param n_elements     Number of elements.
 * @param src_offset_el  Offset into device buffer.
 * @return CVL_CL_SUCCESS or error.
 */
cvl_cl_status_t cvl_cl_staging_buffer_read_and_wait(cvl_cl_staging_buffer_t *buf, cvl_cl_chain_t *chain,
                                                    real3_t *host_data, float *scratch_f32, size_t n_elements,
                                                    size_t src_offset_el);

/**
 * @brief Destroy the staging buffer, releasing the device buffer.
 */
void cvl_cl_staging_buffer_destroy(cvl_cl_staging_buffer_t *buf);
