#include "cvl_cl_command.h"
#include "cvl_cl_helpers.h"

#include <assert.h>

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

bool cvl_cl_event_is_ready(const cvl_cl_event_t *event)
{
    if (!event || !event->event)
        return true;

    cl_int status = 0;
    const cl_int err = clGetEventInfo(event->event, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(status), &status, NULL);
    if (err != CL_SUCCESS)
        return true; /* Status unknown - treat as ready; the error surfaces on wait. */

    /* CL_QUEUED(-3) < CL_SUBMITTED(-2) < CL_RUNNING(-1) < CL_COMPLETE(0);
     * positive values are error terminations, which are also "done". */
    return status >= CL_COMPLETE;
}

void cvl_cl_event_release(cvl_cl_event_t *event)
{
    if (!event)
        return;
    if (event->owns && event->event)
        clReleaseEvent(event->event);
    event->event = NULL;
    event->owns = false;
}

/* ------------------------------------------------------------------ */
/* NDRange kernel launch                                              */
/* ------------------------------------------------------------------ */

cvl_cl_status_t cvl_cl_ndrange(cl_command_queue queue, cl_kernel kernel, unsigned dims, const size_t global_work[],
                               const size_t local_work[], const cvl_cl_karg_t kargs[], unsigned n_wait,
                               const cvl_cl_event_t *wait_events, cvl_cl_event_t *out_event)
{
    assert(queue && kernel && global_work);
    assert(dims >= 1 && dims <= 3);

    /* Set kernel arguments if provided. */
    if (kargs != NULL)
    {
        const cvl_cl_status_t s = cvl_cl_kernel_set_args(kernel, kargs);
        if (s != CVL_CL_SUCCESS)
            return s;
    }

    cl_event wait_list_raw[CL_MAX_WAIT_EVENTS];
    unsigned n_raw = 0;
    cl_prepare_wait_list(n_wait, wait_events, wait_list_raw, &n_raw);

    cl_event raw_out = NULL;
    const cl_int err = clEnqueueNDRangeKernel(queue, kernel, dims, NULL, global_work, local_work, n_raw,
                                              n_raw > 0 ? wait_list_raw : NULL, &raw_out);
    if (err != CL_SUCCESS)
        return cvl_cl_status_from_cl_int(err);

    if (out_event != NULL)
        *out_event = cvl_cl_event_take(raw_out);
    else if (raw_out != NULL)
        clReleaseEvent(raw_out);

    return CVL_CL_SUCCESS;
}

/* ------------------------------------------------------------------ */
/* Buffer transfers                                                   */
/* ------------------------------------------------------------------ */

cvl_cl_status_t cvl_cl_write_buffer(cl_command_queue queue, cvl_cl_buffer_t *buffer, size_t offset, size_t size,
                                    const void *host_ptr, unsigned n_wait, const cvl_cl_event_t *wait_events,
                                    cvl_cl_event_t *out_event)
{
    assert(queue && buffer && (size == 0 || host_ptr));

    cl_event wait_list_raw[CL_MAX_WAIT_EVENTS];
    unsigned n_raw = 0;
    cl_prepare_wait_list(n_wait, wait_events, wait_list_raw, &n_raw);

    cl_event raw_out = NULL;
    const cl_int err = clEnqueueWriteBuffer(queue, buffer->mem, CL_FALSE, offset, size, host_ptr, n_raw,
                                            n_raw > 0 ? wait_list_raw : NULL, &raw_out);
    if (err != CL_SUCCESS)
        return cvl_cl_status_from_cl_int(err);

    if (out_event != NULL)
        *out_event = cvl_cl_event_take(raw_out);
    else if (raw_out != NULL)
        clReleaseEvent(raw_out);

    return CVL_CL_SUCCESS;
}

cvl_cl_status_t cvl_cl_read_buffer(cl_command_queue queue, const cvl_cl_buffer_t *buffer, size_t offset, size_t size,
                                   void *host_ptr, unsigned n_wait, const cvl_cl_event_t *wait_events,
                                   cvl_cl_event_t *out_event)
{
    assert(queue && buffer && (size == 0 || host_ptr));

    cl_event wait_list_raw[CL_MAX_WAIT_EVENTS];
    unsigned n_raw = 0;
    cl_prepare_wait_list(n_wait, wait_events, wait_list_raw, &n_raw);

    cl_event raw_out = NULL;
    const cl_int err = clEnqueueReadBuffer(queue, buffer->mem, CL_FALSE, offset, size, host_ptr, n_raw,
                                           n_raw > 0 ? wait_list_raw : NULL, &raw_out);
    if (err != CL_SUCCESS)
        return cvl_cl_status_from_cl_int(err);

    if (out_event != NULL)
        *out_event = cvl_cl_event_take(raw_out);
    else if (raw_out != NULL)
        clReleaseEvent(raw_out);

    return CVL_CL_SUCCESS;
}

cvl_cl_status_t cvl_cl_copy_buffer(cl_command_queue queue, const cvl_cl_buffer_t *src, cvl_cl_buffer_t *dst,
                                   size_t src_offset, size_t dst_offset, size_t size, unsigned n_wait,
                                   const cvl_cl_event_t *wait_events, cvl_cl_event_t *out_event)
{
    assert(queue && src && dst);

    cl_event wait_list_raw[CL_MAX_WAIT_EVENTS];
    unsigned n_raw = 0;
    cl_prepare_wait_list(n_wait, wait_events, wait_list_raw, &n_raw);

    cl_event raw_out = NULL;
    const cl_int err = clEnqueueCopyBuffer(queue, src->mem, dst->mem, src_offset, dst_offset, size, n_raw,
                                           n_raw > 0 ? wait_list_raw : NULL, &raw_out);
    if (err != CL_SUCCESS)
        return cvl_cl_status_from_cl_int(err);

    if (out_event != NULL)
        *out_event = cvl_cl_event_take(raw_out);
    else if (raw_out != NULL)
        clReleaseEvent(raw_out);

    return CVL_CL_SUCCESS;
}

/* ------------------------------------------------------------------ */
/* Synchronisation                                                   */
/* ------------------------------------------------------------------ */

cvl_cl_status_t cvl_cl_finish(cl_command_queue queue)
{
    assert(queue);
    const cl_int err = clFinish(queue);
    if (err != CL_SUCCESS)
        return cvl_cl_status_from_cl_int(err);
    return CVL_CL_SUCCESS;
}

cvl_cl_status_t cvl_cl_flush(cl_command_queue queue)
{
    assert(queue);
    const cl_int err = clFlush(queue);
    if (err != CL_SUCCESS)
        return cvl_cl_status_from_cl_int(err);
    return CVL_CL_SUCCESS;
}

cvl_cl_status_t cvl_cl_wait_for_events(unsigned n_events, const cl_event event_list[])
{
    if (n_events == 0)
        return CVL_CL_SUCCESS;
    const cl_int err = clWaitForEvents(n_events, event_list);
    if (err != CL_SUCCESS)
        return cvl_cl_status_from_cl_int(err);
    return CVL_CL_SUCCESS;
}
