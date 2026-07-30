#include "cvl_cl_command.h"

#include <string.h>

/* ------------------------------------------------------------------ */
/* Event management                                                   */
/* ------------------------------------------------------------------ */

cvl_cl_event_t cvl_cl_event_wrap(cl_event raw)
{
    cvl_cl_event_t ev;
    ev.event = raw;
    ev.owns = false;
    return ev;
}

cvl_cl_event_t cvl_cl_event_take(cl_event raw)
{
    cvl_cl_event_t ev;
    ev.event = raw;
    ev.owns = true;
    return ev;
}

cvl_cl_status_t cvl_cl_event_wait(const cvl_cl_event_t *event)
{
    if (!event || !event->event)
        return CVL_CL_SUCCESS; /* Nothing to wait on. */
    const cl_int err = clWaitForEvents(1, &event->event);
    if (err != CL_SUCCESS)
        return cvl_cl_status_from_cl_int(err);
    return CVL_CL_SUCCESS;
}

void cvl_cl_event_release(cvl_cl_event_t *event)
{
    if (!event)
        return;
    if (event->owns && event->event)
    {
        clReleaseEvent(event->event);
    }
    event->event = NULL;
    event->owns = false;
}

/* ------------------------------------------------------------------ */
/* Internal: build a raw cl_event* list from wait_events              */
/* ------------------------------------------------------------------ */

static cl_event *build_event_list(unsigned n_wait, const cvl_cl_event_t *wait_events, cl_event *scratch)
{
    if (n_wait == 0 || !wait_events)
    {
        scratch[0] = NULL;
        return NULL;
    }
    for (unsigned i = 0; i < n_wait; ++i)
        scratch[i] = wait_events[i].event;
    return scratch;
}

/* ------------------------------------------------------------------ */
/* NDRange kernel launch                                              */
/* ------------------------------------------------------------------ */

cvl_cl_status_t cvl_cl_ndrange(cvl_cl_queue_t *queue, cvl_cl_kernel_t *kernel, unsigned dims,
                               const size_t global_work[], const size_t local_work[], const cvl_cl_karg_t kargs[],
                               unsigned n_wait, const cvl_cl_event_t *wait_events, cvl_cl_event_t *out_event)
{
    if (!queue || !kernel || !queue->queue || !kernel->kernel)
        return CVL_CL_ERR_INVALID_PARAM;
    if (dims < 1 || dims > 3 || !global_work)
        return CVL_CL_ERR_INVALID_PARAM;

    cvl_cl_status_t status = CVL_CL_SUCCESS;

    /* Set kernel arguments if provided. */
    if (kargs)
    {
        status = cvl_cl_kernel_set_args(kernel, kargs);
        if (status != CVL_CL_SUCCESS)
            return status;
    }

    /* Build event wait list. */
    cl_event wait_list_raw[16];
    cl_event *wait_ptr = NULL;
    if (n_wait > 0)
    {
        if (n_wait > 16)
        {
            /* For large wait lists, caller should manage events externally.
             * This is a soft limit — in practice wait lists are small. */
            return CVL_CL_ERR_INVALID_PARAM;
        }
        wait_ptr = build_event_list(n_wait, wait_events, wait_list_raw);
    }

    /* Output event. */
    cl_event raw_out = NULL;
    cl_event *p_out = out_event ? &raw_out : NULL;

    cl_int err = clEnqueueNDRangeKernel(queue->queue, kernel->kernel, dims, NULL, global_work, local_work, n_wait,
                                        wait_ptr, p_out);
    if (err != CL_SUCCESS)
        return cvl_cl_status_from_cl_int(err);

    if (out_event)
        *out_event = cvl_cl_event_take(raw_out);

    return CVL_CL_SUCCESS;
}

/* ------------------------------------------------------------------ */
/* Buffer transfers                                                   */
/* ------------------------------------------------------------------ */

cvl_cl_status_t cvl_cl_write_buffer(cvl_cl_queue_t *queue, cvl_cl_buffer_t *buffer, size_t offset, size_t size,
                                    const void *host_ptr, unsigned n_wait, const cvl_cl_event_t *wait_events,
                                    cvl_cl_event_t *out_event)
{
    if (!queue || !buffer || !queue->queue || !buffer->mem)
        return CVL_CL_ERR_INVALID_PARAM;
    if (!host_ptr || size == 0)
        return CVL_CL_ERR_INVALID_PARAM;
    if (offset + size > buffer->capacity)
        return CVL_CL_ERR_BUFFER_SIZE;

    cl_event wait_list_raw[16];
    cl_event *wait_ptr = NULL;
    if (n_wait > 0)
    {
        if (n_wait > 16)
            return CVL_CL_ERR_INVALID_PARAM;
        wait_ptr = build_event_list(n_wait, wait_events, wait_list_raw);
    }

    cl_event raw_out = NULL;
    cl_event *p_out = out_event ? &raw_out : NULL;

    cl_int err =
        clEnqueueWriteBuffer(queue->queue, buffer->mem, CL_FALSE, offset, size, host_ptr, n_wait, wait_ptr, p_out);
    if (err != CL_SUCCESS)
        return cvl_cl_status_from_cl_int(err);

    if (out_event)
        *out_event = cvl_cl_event_take(raw_out);

    return CVL_CL_SUCCESS;
}

cvl_cl_status_t cvl_cl_read_buffer(cvl_cl_queue_t *queue, const cvl_cl_buffer_t *buffer, size_t offset, size_t size,
                                   void *host_ptr, unsigned n_wait, const cvl_cl_event_t *wait_events,
                                   cvl_cl_event_t *out_event)
{
    if (!queue || !buffer || !queue->queue || !buffer->mem)
        return CVL_CL_ERR_INVALID_PARAM;
    if (!host_ptr || size == 0)
        return CVL_CL_ERR_INVALID_PARAM;
    if (offset + size > buffer->capacity)
        return CVL_CL_ERR_BUFFER_SIZE;

    cl_event wait_list_raw[16];
    cl_event *wait_ptr = NULL;
    if (n_wait > 0)
    {
        if (n_wait > 16)
            return CVL_CL_ERR_INVALID_PARAM;
        wait_ptr = build_event_list(n_wait, wait_events, wait_list_raw);
    }

    cl_event raw_out = NULL;
    cl_event *p_out = out_event ? &raw_out : NULL;

    cl_int err =
        clEnqueueReadBuffer(queue->queue, buffer->mem, CL_FALSE, offset, size, host_ptr, n_wait, wait_ptr, p_out);
    if (err != CL_SUCCESS)
        return cvl_cl_status_from_cl_int(err);

    if (out_event)
        *out_event = cvl_cl_event_take(raw_out);

    return CVL_CL_SUCCESS;
}

cvl_cl_status_t cvl_cl_copy_buffer(cvl_cl_queue_t *queue, const cvl_cl_buffer_t *src, cvl_cl_buffer_t *dst,
                                   size_t src_offset, size_t dst_offset, size_t size, unsigned n_wait,
                                   const cvl_cl_event_t *wait_events, cvl_cl_event_t *out_event)
{
    if (!queue || !src || !dst || !queue->queue || !src->mem || !dst->mem)
        return CVL_CL_ERR_INVALID_PARAM;
    if (size == 0)
        return CVL_CL_SUCCESS;
    if (src_offset + size > src->capacity || dst_offset + size > dst->capacity)
        return CVL_CL_ERR_BUFFER_SIZE;

    cl_event wait_list_raw[16];
    cl_event *wait_ptr = NULL;
    if (n_wait > 0)
    {
        if (n_wait > 16)
            return CVL_CL_ERR_INVALID_PARAM;
        wait_ptr = build_event_list(n_wait, wait_events, wait_list_raw);
    }

    cl_event raw_out = NULL;
    cl_event *p_out = out_event ? &raw_out : NULL;

    cl_int err =
        clEnqueueCopyBuffer(queue->queue, src->mem, dst->mem, src_offset, dst_offset, size, n_wait, wait_ptr, p_out);
    if (err != CL_SUCCESS)
        return cvl_cl_status_from_cl_int(err);

    if (out_event)
        *out_event = cvl_cl_event_take(raw_out);

    return CVL_CL_SUCCESS;
}

/* ------------------------------------------------------------------ */
/* Synchronisation                                                   */
/* ------------------------------------------------------------------ */

cvl_cl_status_t cvl_cl_finish(cvl_cl_queue_t *queue)
{
    if (!queue || !queue->queue)
        return CVL_CL_ERR_INVALID_PARAM;
    const cl_int err = clFinish(queue->queue);
    if (err != CL_SUCCESS)
        return cvl_cl_status_from_cl_int(err);
    return CVL_CL_SUCCESS;
}

cvl_cl_status_t cvl_cl_flush(cvl_cl_queue_t *queue)
{
    if (!queue || !queue->queue)
        return CVL_CL_ERR_INVALID_PARAM;
    const cl_int err = clFlush(queue->queue);
    if (err != CL_SUCCESS)
        return cvl_cl_status_from_cl_int(err);
    return CVL_CL_SUCCESS;
}

cvl_cl_status_t cvl_cl_wait_for_events(unsigned n_events, const cl_event event_list[])
{
    if (n_events == 0)
        return CVL_CL_SUCCESS;
    if (!event_list)
        return CVL_CL_ERR_INVALID_PARAM;
    const cl_int err = clWaitForEvents(n_events, event_list);
    if (err != CL_SUCCESS)
        return cvl_cl_status_from_cl_int(err);
    return CVL_CL_SUCCESS;
}
