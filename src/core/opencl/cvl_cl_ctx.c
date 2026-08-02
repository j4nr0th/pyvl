#include "cvl_cl_ctx.h"
#include "cvl_cl_device.h"

#include <stdlib.h>

cvl_cl_status_t cvl_cl_ctx_create(const cvl_cl_device_t *device, cvl_cl_ctx_t *out_ctx)
{
    if (!device || !out_ctx)
        return CVL_CL_ERR_INVALID_PARAM;

    out_ctx->context = NULL;
    out_ctx->device = NULL;

    cl_int err;
    cl_device_id dev_id = device->id;
    cl_context ctx = clCreateContext(NULL, 1, &dev_id, NULL, NULL, &err);
    if (err != CL_SUCCESS)
        return cvl_cl_status_from_cl_int(err);

    out_ctx->context = ctx;
    out_ctx->device = device;
    return CVL_CL_SUCCESS;
}

void cvl_cl_ctx_destroy(cvl_cl_ctx_t *ctx)
{
    if (!ctx)
        return;
    if (ctx->context)
    {
        clReleaseContext(ctx->context);
        ctx->context = NULL;
    }
    ctx->device = NULL;
}

cvl_cl_status_t cvl_cl_queue_create(const cvl_cl_ctx_t *ctx, const cvl_cl_queue_props_t *props, cvl_cl_queue_t *out_q)
{
    if (!ctx || !out_q || !ctx->context)
        return CVL_CL_ERR_INVALID_PARAM;

    out_q->queue = NULL;
    out_q->ctx = NULL;

    /* Build properties for clCreateCommandQueueWithProperties (OpenCL 2.0+).
     * For maximum compatibility, fall back to clCreateCommandQueue (OpenCL 1.2) if
     * we can't use the properties version. */
    cl_command_queue_properties qprops = 0;
    if (props)
    {
        if (props->profiling)
            qprops |= CL_QUEUE_PROFILING_ENABLE;
    }

    cl_int err;
#ifdef CL_VERSION_2_0
    cl_queue_properties prop_list[3] = {0};
    unsigned np = 0;
    if (props && props->out_of_order)
    {
        prop_list[np++] = CL_QUEUE_PROPERTIES;
        prop_list[np++] = (cl_queue_properties)(CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | qprops);
    }
    else if (qprops)
    {
        prop_list[np++] = CL_QUEUE_PROPERTIES;
        prop_list[np++] = (cl_queue_properties)qprops;
    }
    prop_list[np] = 0;
    cl_command_queue q = clCreateCommandQueueWithProperties(ctx->context, ctx->device->id, prop_list, &err);
#else
    cl_command_queue q = clCreateCommandQueue(ctx->context, ctx->device->id, qprops, &err);
#endif
    if (err != CL_SUCCESS)
        return cvl_cl_status_from_cl_int(err);

    out_q->queue = q;
    out_q->ctx = ctx;
    return CVL_CL_SUCCESS;
}

void cvl_cl_queue_destroy(cvl_cl_queue_t *q)
{
    if (!q)
        return;
    if (q->queue)
    {
        clReleaseCommandQueue(q->queue);
        q->queue = NULL;
    }
    q->ctx = NULL;
}
