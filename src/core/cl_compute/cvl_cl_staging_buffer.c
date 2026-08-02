#include "cvl_cl_staging_buffer.h"
#include "../opencl/cvl_cl_helpers.h"

#include <string.h>

cvl_cl_status_t cvl_cl_staging_buffer_init(cvl_cl_staging_buffer_t *buf, cvl_cl_precision_t precision,
                                           bool unified_memory, size_t max_elements, const allocator_t *allocator)
{
    memset(buf, 0, sizeof(*buf));
    buf->precision = precision;
    buf->unified_memory = unified_memory;
    buf->allocator = cl_resolve_allocator(allocator);
    buf->element_size_bytes = (precision == CVL_CL_PRECISION_FP32) ? (3u * sizeof(float)) : sizeof(real3_t);

    /* Pre-allocate FP32 host staging to avoid realloc during reserve(). */
    if (precision == CVL_CL_PRECISION_FP32 && max_elements > 0)
    {
        buf->host_fp32 = (float *)cl_alloc(buf->allocator, max_elements * 3u * sizeof(float));
        if (!buf->host_fp32)
            return CVL_CL_ERR_MEMORY;
    }

    return CVL_CL_SUCCESS;
}

cvl_cl_status_t cvl_cl_staging_buffer_reserve(cvl_cl_staging_buffer_t *buf, const cvl_cl_ctx_t *ctx,
                                              cvl_cl_queue_t *queue, size_t n_elements)
{
    if (!buf || !ctx)
        return CVL_CL_ERR_INVALID_PARAM;

    if (n_elements <= buf->capacity_elements)
        return CVL_CL_SUCCESS;

    /* Grow device buffer. */
    const size_t needed_bytes = n_elements * buf->element_size_bytes;
    cvl_cl_status_t st = cvl_cl_buffer_reserve(&buf->device, ctx, queue, needed_bytes);
    if (st != CVL_CL_SUCCESS)
        return st;

    /* FP32 host staging is pre-allocated at init - no realloc needed. */

    buf->capacity_elements = n_elements;
    return CVL_CL_SUCCESS;
}

cvl_cl_status_t cvl_cl_staging_buffer_write_async(cvl_cl_staging_buffer_t *buf, cvl_cl_queue_t *queue,
                                                  const real3_t *host_data, size_t n_elements, size_t dst_offset_el,
                                                  cvl_cl_future_t *out_future)
{
    if (!buf || !queue || !host_data || !out_future)
        return CVL_CL_ERR_INVALID_PARAM;
    if (dst_offset_el + n_elements > buf->capacity_elements)
        return CVL_CL_ERR_BUFFER_SIZE;

    cvl_cl_future_init(out_future);

    const size_t offset_bytes = dst_offset_el * buf->element_size_bytes;
    const size_t data_bytes = n_elements * buf->element_size_bytes;

    if (buf->precision == CVL_CL_PRECISION_FP32)
    {
        /* Convert double → float into host staging. */
        float *staging = buf->host_fp32 + dst_offset_el * 3u;
#pragma omp simd
        for (size_t i = 0; i < n_elements; ++i)
        {
            staging[3u * i + 0u] = (float)host_data[i].x;
            staging[3u * i + 1u] = (float)host_data[i].y;
            staging[3u * i + 2u] = (float)host_data[i].z;
        }
        /* Enqueue write of the float staging buffer. */
        cl_event evt = NULL;
        cl_int err = clEnqueueWriteBuffer(cvl_cl_queue_queue(queue), buf->device.mem, CL_FALSE, offset_bytes,
                                          data_bytes, staging, 0, NULL, &evt);
        if (err != CL_SUCCESS)
            return cvl_cl_status_from_cl_int(err);
        out_future->event = evt;
    }
    else
    {
        /* FP64: write directly from host data. */
        cl_event evt = NULL;
        cl_int err = clEnqueueWriteBuffer(queue->queue, buf->device.mem, CL_FALSE, offset_bytes, data_bytes, host_data,
                                          0, NULL, &evt);
        if (err != CL_SUCCESS)
            return cvl_cl_status_from_cl_int(err);
        out_future->event = evt;
    }

    return CVL_CL_SUCCESS;
}

cvl_cl_status_t cvl_cl_staging_buffer_read_async(cvl_cl_staging_buffer_t *buf, cvl_cl_queue_t *queue,
                                                 real3_t *host_data, size_t n_elements, size_t src_offset_el,
                                                 cvl_cl_future_t *out_future)
{
    if (!buf || !queue || !host_data || !out_future)
        return CVL_CL_ERR_INVALID_PARAM;
    if (src_offset_el + n_elements > buf->capacity_elements)
        return CVL_CL_ERR_BUFFER_SIZE;

    cvl_cl_future_init(out_future);

    const size_t offset_bytes = src_offset_el * buf->element_size_bytes;
    const size_t data_bytes = n_elements * buf->element_size_bytes;

    if (buf->precision == CVL_CL_PRECISION_FP32)
    {
        /* Read into float staging (conversion happens in read_finish). */
        float *staging = buf->host_fp32 + src_offset_el * 3u;
        cl_event evt = NULL;
        cl_int err = clEnqueueReadBuffer(queue->queue, buf->device.mem, CL_FALSE, offset_bytes, data_bytes, staging, 0,
                                         NULL, &evt);
        if (err != CL_SUCCESS)
            return cvl_cl_status_from_cl_int(err);
        out_future->event = evt;
    }
    else
    {
        /* FP64: read directly into host output. */
        cl_event evt = NULL;
        cl_int err = clEnqueueReadBuffer(queue->queue, buf->device.mem, CL_FALSE, offset_bytes, data_bytes, host_data,
                                         0, NULL, &evt);
        if (err != CL_SUCCESS)
            return cvl_cl_status_from_cl_int(err);
        out_future->event = evt;
    }

    return CVL_CL_SUCCESS;
}

cvl_cl_status_t cvl_cl_staging_buffer_read_finish(cvl_cl_staging_buffer_t *buf, real3_t *host_data, size_t n_elements,
                                                  size_t src_offset_el)
{
    if (!buf || !host_data)
        return CVL_CL_ERR_INVALID_PARAM;

    /* FP64 mode has no conversion - data is already in host_data. */
    if (buf->precision != CVL_CL_PRECISION_FP32)
        return CVL_CL_SUCCESS;

    if (src_offset_el + n_elements > buf->capacity_elements)
        return CVL_CL_ERR_BUFFER_SIZE;

    /* Convert float staging → double host_data. */
    const float *staging = buf->host_fp32 + src_offset_el * 3u;
#pragma omp simd
    for (size_t i = 0; i < n_elements; ++i)
    {
        host_data[i].x = (real_t)staging[3u * i + 0u];
        host_data[i].y = (real_t)staging[3u * i + 1u];
        host_data[i].z = (real_t)staging[3u * i + 2u];
    }

    return CVL_CL_SUCCESS;
}

cvl_cl_status_t cvl_cl_staging_buffer_read_and_wait(cvl_cl_staging_buffer_t *buf, cvl_cl_queue_t *queue,
                                                    real3_t *host_data, size_t n_elements, size_t src_offset_el)
{
    cvl_cl_future_t f;
    cvl_cl_status_t st = cvl_cl_staging_buffer_read_async(buf, queue, host_data, n_elements, src_offset_el, &f);
    if (st != CVL_CL_SUCCESS)
        return st;

    st = cvl_cl_future_wait(&f);
    if (st != CVL_CL_SUCCESS)
        return st;

    return cvl_cl_staging_buffer_read_finish(buf, host_data, n_elements, src_offset_el);
}

void cvl_cl_staging_buffer_destroy(cvl_cl_staging_buffer_t *buf)
{
    if (!buf)
        return;
    cl_free(buf->allocator, buf->host_fp32);
    buf->host_fp32 = NULL;
    cvl_cl_buffer_destroy(&buf->device);
    memset(buf, 0, sizeof(*buf));
}
