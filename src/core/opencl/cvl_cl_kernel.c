#include "cvl_cl_kernel.h"

#include <string.h>

cvl_cl_status_t cvl_cl_kernel_create(const cvl_cl_program_t *program, const char *name, cvl_cl_kernel_t *out)
{
    if (!program || !name || !out || !program->program)
        return CVL_CL_ERR_INVALID_PARAM;

    out->kernel = NULL;
    out->program = NULL;
    out->preferred_wg_multiple = 0;

    cl_int err;
    cl_kernel k = clCreateKernel(program->program, name, &err);
    if (err != CL_SUCCESS)
        return cvl_cl_status_from_cl_int(err);

    /* Cache preferred work-group size multiple. */
    size_t wg_multiple = 0;
    err =
        clGetKernelWorkGroupInfo(k, cvl_cl_device_id(cvl_cl_ctx_device(program->ctx)),
                                 CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(wg_multiple), &wg_multiple, NULL);
    if (err != CL_SUCCESS)
    {
        /* Non-fatal: just leave it as 0.  The runtime will pick a valid
         * size when local_work_size is NULL in clEnqueueNDRangeKernel. */
        wg_multiple = 0;
    }

    out->kernel = k;
    out->program = program;
    out->preferred_wg_multiple = wg_multiple;
    return CVL_CL_SUCCESS;
}

cvl_cl_status_t cvl_cl_kernel_set_args(cvl_cl_kernel_t *kernel, const cvl_cl_karg_t kargs[])
{
    if (!kernel || !kargs || !kernel->kernel)
        return CVL_CL_ERR_INVALID_PARAM;

    for (const cvl_cl_karg_t *arg = kargs; arg->type != CVL_CL_KARG_NONE; ++arg)
    {
        cl_int err;

        switch (arg->type)
        {
        case CVL_CL_KARG_BUFFER: {
            const cl_mem mem = arg->mem;
            err = clSetKernelArg(kernel->kernel, arg->index, sizeof(cl_mem), &mem);
            break;
        }
        case CVL_CL_KARG_SCALAR_INT:
            err = clSetKernelArg(kernel->kernel, arg->index, sizeof(int), &arg->scalar_int);
            break;
        case CVL_CL_KARG_SCALAR_UINT:
            err = clSetKernelArg(kernel->kernel, arg->index, sizeof(unsigned), &arg->scalar_uint);
            break;
        case CVL_CL_KARG_SCALAR_LONG:
            err = clSetKernelArg(kernel->kernel, arg->index, sizeof(long long), &arg->scalar_long);
            break;
        case CVL_CL_KARG_SCALAR_ULONG:
            err = clSetKernelArg(kernel->kernel, arg->index, sizeof(unsigned long long), &arg->scalar_ulong);
            break;
        case CVL_CL_KARG_SCALAR_FLOAT:
            err = clSetKernelArg(kernel->kernel, arg->index, sizeof(float), &arg->scalar_float);
            break;
        case CVL_CL_KARG_SCALAR_DOUBLE:
            err = clSetKernelArg(kernel->kernel, arg->index, sizeof(double), &arg->scalar_double);
            break;
        case CVL_CL_KARG_LOCAL:
            /* __local buffer - pass NULL pointer, size = local_size. */
            err = clSetKernelArg(kernel->kernel, arg->index, arg->local_size, NULL);
            break;
        default:
            return CVL_CL_ERR_KERNEL_ARG;
        }

        if (err != CL_SUCCESS)
            return cvl_cl_status_from_cl_int(err);
    }

    return CVL_CL_SUCCESS;
}

void cvl_cl_kernel_destroy(cvl_cl_kernel_t *kernel)
{
    if (!kernel)
        return;
    if (kernel->kernel)
    {
        clReleaseKernel(kernel->kernel);
        kernel->kernel = NULL;
    }
    kernel->program = NULL;
    kernel->preferred_wg_multiple = 0;
}
