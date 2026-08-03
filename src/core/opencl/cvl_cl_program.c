#include "cvl_cl_program.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

cvl_cl_status_t cvl_cl_program_create(cl_context ctx, cl_device_id device, const cvl_cl_program_desc_t *desc, char *log,
                                      size_t log_capacity, cl_program *out)
{
    assert(desc && desc->source_string && out);
    assert(ctx);

    *out = NULL;

    /*
     * Build a combined options string that includes the precision define.
     *
     * FP32 mode defines CVL_CL_REAL_FP32 so the .cl.h headers switch
     * real_t from double → float.  FP64 / DEFAULT use the header's
     * default (double) and need no extra define - the cl_khr_fp64
     * pragma is handled inside the .cl.h type header itself.
     */
    char full_opts[1024];
    const char *opts;

    if (desc->precision == CVL_CL_PRECISION_FP32)
    {
        if (desc->build_options)
            snprintf(full_opts, sizeof(full_opts), "%s -DCVL_CL_REAL_FP32", desc->build_options);
        else
            snprintf(full_opts, sizeof(full_opts), "-DCVL_CL_REAL_FP32");
        full_opts[sizeof(full_opts) - 1] = '\0';
        opts = full_opts;
    }
    else
    {
        /* FP64 / DEFAULT - the .cl.h headers use double by default. */
        opts = desc->build_options;
    }

    const char *sources[] = {desc->source_string};
    const size_t lengths[] = {strlen(desc->source_string)};

    cl_int err;
    cl_program prog = clCreateProgramWithSource(ctx, 1, sources, lengths, &err);
    if (err != CL_SUCCESS)
        return cvl_cl_status_from_cl_int(err);

    /* Build with combined options. */
    err = clBuildProgram(prog, 1, &device, opts, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        /* Capture the build log into the caller's buffer (truncated). */
        if (log != NULL && log_capacity > 0)
        {
            size_t log_size = 0;
            clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
            const size_t n = (log_size < log_capacity - 1) ? log_size : (log_capacity - 1);
            clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, n, log, NULL);
            log[n] = '\0';
        }
        clReleaseProgram(prog);
        *out = NULL;
        return CVL_CL_ERR_PROGRAM_BUILD;
    }

    *out = prog;
    return CVL_CL_SUCCESS;
}

void cvl_cl_program_destroy(cl_program *program)
{
    assert(program);
    clReleaseProgram(*program);
    *program = NULL;
}
