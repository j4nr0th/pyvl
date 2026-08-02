#include "cvl_cl_future.h"

cvl_cl_status_t cvl_cl_future_wait(cvl_cl_future_t *f)
{
    if (!f || !f->event)
        return CVL_CL_SUCCESS;

    const cl_int err = clWaitForEvents(1, &f->event);
    /* Release the event after waiting - the future expires. */
    clReleaseEvent(f->event);
    f->event = NULL;

    if (err != CL_SUCCESS)
        return cvl_cl_status_from_cl_int(err);
    return CVL_CL_SUCCESS;
}

bool cvl_cl_future_is_ready(const cvl_cl_future_t *f)
{
    if (!f || !f->event)
        return true; /* empty future = trivially ready */

    cl_int status;
    const cl_int err = clGetEventInfo(f->event, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(status), &status, NULL);
    if (err != CL_SUCCESS)
        return true; /* can't determine → assume ready (error will surface on wait) */

    return status <= CL_COMPLETE;
}

void cvl_cl_future_release(cvl_cl_future_t *f)
{
    if (!f || !f->event)
        return;
    clReleaseEvent(f->event);
    f->event = NULL;
}
