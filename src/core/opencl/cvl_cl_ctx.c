#include "cvl_cl_ctx.h"

#include <assert.h>

cvl_cl_status_t cvl_cl_ctx_create(const cvl_cl_device_t *device, cl_context *out_ctx)
{
    /* Internal module: NULL device / out pointer are contract violations. */
    assert(device != NULL);
    assert(out_ctx != NULL);

    *out_ctx = NULL;

    /* Single-device context; the platform property pins the device's platform. */
    const cl_context_properties props[] = {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)device->platform_id,
        0,
    };

    cl_int err = 0;
    cl_context ctx = clCreateContext(props, 1, &device->id, NULL, NULL, &err);
    if (err != CL_SUCCESS)
        return cvl_cl_status_from_cl_int(err);

    *out_ctx = ctx;
    return CVL_CL_SUCCESS;
}

void cvl_cl_ctx_destroy(cl_context *ctx)
{
    assert(ctx != NULL);
    if (*ctx != NULL)
    {
        clReleaseContext(*ctx);
        *ctx = NULL;
    }
}

cvl_cl_status_t cvl_cl_queue_create(cl_context ctx, cl_device_id device_id, const cvl_cl_queue_props_t *props,
                                    cl_command_queue *out_q)
{
    /* Internal module: NULL ctx / device / out pointer are contract violations. */
    assert(ctx != NULL);
    assert(device_id != NULL);
    assert(out_q != NULL);

    *out_q = NULL;

    cl_command_queue_properties flags = 0;
    if (props)
    {
        if (props->out_of_order)
            flags |= CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
        if (props->profiling)
            flags |= CL_QUEUE_PROFILING_ENABLE;
    }

    cl_int err = 0;
#ifdef CL_VERSION_2_0
    /* OpenCL 2.0+: property list; a zeroed list means default in-order, no profiling. */
    cl_queue_properties qprops[3] = {0};
    unsigned n = 0;
    if (flags != 0)
    {
        qprops[n++] = CL_QUEUE_PROPERTIES;
        qprops[n++] = (cl_queue_properties)flags;
    }
    qprops[n] = 0;
    cl_command_queue q = clCreateCommandQueueWithProperties(ctx, device_id, qprops, &err);
#else
    /* OpenCL 1.2 fallback: old-style flag bitmask. */
    cl_command_queue q = clCreateCommandQueue(ctx, device_id, flags, &err);
#endif
    if (err != CL_SUCCESS)
        return cvl_cl_status_from_cl_int(err);

    *out_q = q;
    return CVL_CL_SUCCESS;
}

void cvl_cl_queue_destroy(cl_command_queue *q)
{
    assert(q != NULL);
    if (*q != NULL)
    {
        clReleaseCommandQueue(*q);
        *q = NULL;
    }
}
