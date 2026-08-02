#include "cvl_cl_program.h"
#include "cvl_cl_helpers.h"

#include <stdio.h>
#include <string.h>

cvl_cl_status_t cvl_cl_program_create(const cvl_cl_ctx_t *ctx, const cvl_cl_program_desc_t *desc, cl_device_id device,
                                      cvl_cl_program_t *out, const allocator_t *allocator)
{
    if (!ctx || !desc || !out || !ctx->context)
        return CVL_CL_ERR_INVALID_PARAM;

    out->program = NULL;
    out->ctx = NULL;
    out->allocator = cl_resolve_allocator(allocator);
    out->build_log = NULL;

    /* Create program from source. */
    cl_int err;
    cl_program prog;

    switch (desc->source_type)
    {
    case CVL_CL_PROGRAM_SOURCE_STRING: {
        if (!desc->source_string)
            return CVL_CL_ERR_INVALID_PARAM;

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
        prog = clCreateProgramWithSource(ctx->context, 1, sources, lengths, &err);
        if (err != CL_SUCCESS)
            return cvl_cl_status_from_cl_int(err);

        /* Build with combined options. */
        err = clBuildProgram(prog, 1, &device, opts, NULL, NULL);
        if (err != CL_SUCCESS)
        {
            /* Capture build log. */
            size_t log_size = 0;
            clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
            if (log_size > 0)
            {
                out->build_log = (char *)cl_alloc(out->allocator, log_size + 1);
                if (out->build_log)
                {
                    clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, log_size, out->build_log, NULL);
                    out->build_log[log_size] = '\0';
                }
            }
            clReleaseProgram(prog);
            out->program = NULL;
            return CVL_CL_ERR_PROGRAM_BUILD;
        }
        break;
    }

    default:
        return CVL_CL_ERR_INVALID_PARAM;
    }

    out->program = prog;
    out->ctx = ctx;
    return CVL_CL_SUCCESS;
}

const char *cvl_cl_program_build_log(const cvl_cl_program_t *program)
{
    return program ? program->build_log : NULL;
}

void cvl_cl_program_destroy(cvl_cl_program_t *program)
{
    if (!program)
        return;
    cl_free(program->allocator, program->build_log);
    program->build_log = NULL;
    if (program->program)
    {
        clReleaseProgram(program->program);
        program->program = NULL;
    }
    program->ctx = NULL;
}
