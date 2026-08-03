#include "cvl_cl_staging_buffer.h"
#include "../common.h" /* real3_t / real_t - cvl_cl_staging_buffer.h uses real3_t */

#include <assert.h>
#include <string.h>

cvl_cl_status_t cvl_cl_staging_buffer_init(cvl_cl_staging_buffer_t *buf, cvl_cl_precision_t precision)
{
    assert(buf);

    memset(buf, 0, sizeof(*buf));
    buf->precision = precision;
    buf->element_size_bytes = (precision == CVL_CL_PRECISION_FP32) ? (3u * sizeof(float)) : (3u * sizeof(double));
    return CVL_CL_SUCCESS;
}

cvl_cl_status_t cvl_cl_staging_buffer_reserve(cvl_cl_staging_buffer_t *buf, cl_context ctx, cl_command_queue queue,
                                              size_t n_elements)
{
    assert(buf && ctx);

    if (n_elements <= buf->capacity_elements)
        return CVL_CL_SUCCESS;

    /* Grow device buffer. */
    const size_t needed_bytes = n_elements * buf->element_size_bytes;
    cvl_cl_status_t st = cvl_cl_buffer_reserve(&buf->device, ctx, queue, needed_bytes);
    if (st != CVL_CL_SUCCESS)
        return st;

    buf->capacity_elements = n_elements;
    return CVL_CL_SUCCESS;
}

cvl_cl_status_t cvl_cl_staging_buffer_write_async(cvl_cl_staging_buffer_t *buf, cvl_cl_chain_t *chain,
                                                  const real3_t *host_data, float *scratch_f32, size_t n_elements,
                                                  size_t dst_offset_el, cvl_cl_event_t *out_event)
{
    assert(buf && chain && host_data);
    assert(dst_offset_el + n_elements <= buf->capacity_elements);

    const size_t offset_bytes = dst_offset_el * buf->element_size_bytes;

    if (buf->precision == CVL_CL_PRECISION_FP32)
    {
        /* Convert host doubles → float scratch (3 floats per element). */
        assert(scratch_f32 != NULL);
        const double *src = (const double *)host_data;
#pragma omp simd
        for (size_t i = 0; i < 3u * n_elements; ++i)
            scratch_f32[i] = (float)src[i];

        return cvl_cl_chain_write_buffer(chain, &buf->device, offset_bytes, n_elements * 3u * sizeof(float),
                                         scratch_f32, 0, NULL, out_event);
    }

    /* FP64: write straight from the user's real3_t array. */
    return cvl_cl_chain_write_buffer(chain, &buf->device, offset_bytes, n_elements * buf->element_size_bytes, host_data,
                                     0, NULL, out_event);
}

cvl_cl_status_t cvl_cl_staging_buffer_read_async(cvl_cl_staging_buffer_t *buf, cvl_cl_chain_t *chain,
                                                 real3_t *host_data, float *scratch_f32, size_t n_elements,
                                                 size_t src_offset_el, cvl_cl_event_t *out_event)
{
    assert(buf && chain && host_data);
    assert(src_offset_el + n_elements <= buf->capacity_elements);

    const size_t offset_bytes = src_offset_el * buf->element_size_bytes;

    if (buf->precision == CVL_CL_PRECISION_FP32)
    {
        /* Read into the float scratch; conversion deferred to read_finish. */
        assert(scratch_f32 != NULL);
        return cvl_cl_chain_read_buffer(chain, &buf->device, offset_bytes, n_elements * 3u * sizeof(float), scratch_f32,
                                        0, NULL, out_event);
    }

    /* FP64: read straight into the user's real3_t array. */
    return cvl_cl_chain_read_buffer(chain, &buf->device, offset_bytes, n_elements * buf->element_size_bytes, host_data,
                                    0, NULL, out_event);
}

cvl_cl_status_t cvl_cl_staging_buffer_read_finish(cvl_cl_staging_buffer_t *buf, real3_t *host_data,
                                                  const float *scratch_f32, size_t n_elements)
{
    assert(buf && host_data);

    /* FP64 has no conversion - the data is already in host_data. */
    if (buf->precision != CVL_CL_PRECISION_FP32)
        return CVL_CL_SUCCESS;

    assert(scratch_f32 != NULL);

    /* Convert float scratch → host doubles (3 doubles per element). */
    double *dst = (double *)host_data;
#pragma omp simd
    for (size_t i = 0; i < 3u * n_elements; ++i)
        dst[i] = (double)scratch_f32[i];

    return CVL_CL_SUCCESS;
}

cvl_cl_status_t cvl_cl_staging_buffer_read_and_wait(cvl_cl_staging_buffer_t *buf, cvl_cl_chain_t *chain,
                                                    real3_t *host_data, float *scratch_f32, size_t n_elements,
                                                    size_t src_offset_el)
{
    assert(buf && chain && host_data);

    cvl_cl_status_t st =
        cvl_cl_staging_buffer_read_async(buf, chain, host_data, scratch_f32, n_elements, src_offset_el, NULL);
    if (st != CVL_CL_SUCCESS)
        return st;

    st = cvl_cl_chain_finish(chain);
    if (st != CVL_CL_SUCCESS)
        return st;

    return cvl_cl_staging_buffer_read_finish(buf, host_data, scratch_f32, n_elements);
}

void cvl_cl_staging_buffer_destroy(cvl_cl_staging_buffer_t *buf)
{
    assert(buf);
    cvl_cl_buffer_destroy(&buf->device);
    memset(buf, 0, sizeof(*buf));
}
