#include "cvl_cl_kernel.h"

#include <assert.h>

cvl_cl_status_t cvl_cl_kernel_create(cl_program program, const char *name, cl_kernel *out)
{
    assert(program && name && out);

    cl_int err;
    cl_kernel k = clCreateKernel(program, name, &err);
    if (err != CL_SUCCESS)
    {
        *out = NULL;
        return cvl_cl_status_from_cl_int(err);
    }

    *out = k;
    return CVL_CL_SUCCESS;
}

cvl_cl_status_t cvl_cl_kernel_set_args(cl_kernel kernel, const cvl_cl_karg_t kargs[])
{
    assert(kernel && kargs);

    for (const cvl_cl_karg_t *arg = kargs; arg->type != CVL_CL_KARG_NONE; ++arg)
    {
        cl_int err;

        switch (arg->type)
        {
        case CVL_CL_KARG_BUFFER:
            /* cl_mem value (the union makes &arg->mem a cl_mem *). */
            err = clSetKernelArg(kernel, arg->index, sizeof(cl_mem), &arg->mem);
            break;
        case CVL_CL_KARG_SCALAR_INT:
            err = clSetKernelArg(kernel, arg->index, sizeof(int), &arg->scalar_int);
            break;
        case CVL_CL_KARG_SCALAR_UINT:
            err = clSetKernelArg(kernel, arg->index, sizeof(unsigned), &arg->scalar_uint);
            break;
        case CVL_CL_KARG_SCALAR_LONG:
            err = clSetKernelArg(kernel, arg->index, sizeof(long long), &arg->scalar_long);
            break;
        case CVL_CL_KARG_SCALAR_ULONG:
            err = clSetKernelArg(kernel, arg->index, sizeof(unsigned long long), &arg->scalar_ulong);
            break;
        case CVL_CL_KARG_SCALAR_FLOAT:
            err = clSetKernelArg(kernel, arg->index, sizeof(float), &arg->scalar_float);
            break;
        case CVL_CL_KARG_SCALAR_DOUBLE:
            err = clSetKernelArg(kernel, arg->index, sizeof(double), &arg->scalar_double);
            break;
        case CVL_CL_KARG_LOCAL:
            /* __local buffer - pass NULL pointer, size = local_size. */
            err = clSetKernelArg(kernel, arg->index, arg->local_size, NULL);
            break;
        default:
            return CVL_CL_ERR_KERNEL_ARG;
        }

        if (err != CL_SUCCESS)
            return CVL_CL_ERR_KERNEL_ARG;
    }

    return CVL_CL_SUCCESS;
}

void cvl_cl_kernel_destroy(cl_kernel *kernel)
{
    assert(kernel);
    clReleaseKernel(*kernel);
    *kernel = NULL;
}
