#include "cvl_cl_program.h"

#include <stdlib.h>
#include <string.h>

cvl_cl_status_t cvl_cl_program_create(const cvl_cl_ctx_t *ctx, const cvl_cl_program_desc_t *desc, cl_device_id device,
                                      cvl_cl_program_t *out)
{
    if (!ctx || !desc || !out || !ctx->context)
        return CVL_CL_ERR_INVALID_PARAM;

    out->program = NULL;
    out->ctx = NULL;
    out->build_log = NULL;

    /* Create program from source. */
    cl_int err;
    cl_program prog;

    switch (desc->source_type)
    {
    case CVL_CL_PROGRAM_SOURCE_STRING: {
        if (!desc->source_string)
            return CVL_CL_ERR_INVALID_PARAM;
        const char *sources[] = {desc->source_string};
        const size_t lengths[] = {strlen(desc->source_string)};
        prog = clCreateProgramWithSource(ctx->context, 1, sources, lengths, &err);
        if (err != CL_SUCCESS)
            return cvl_cl_status_from_cl_int(err);
        break;
    }

    default:
        return CVL_CL_ERR_INVALID_PARAM;
    }

    /* Build. */
    const char *opts = desc->build_options;
    err = clBuildProgram(prog, 1, &device, opts, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        /* Capture build log. */
        size_t log_size = 0;
        clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        if (log_size > 0)
        {
            out->build_log = (char *)malloc(log_size + 1);
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
    free(program->build_log);
    program->build_log = NULL;
    if (program->program)
    {
        clReleaseProgram(program->program);
        program->program = NULL;
    }
    program->ctx = NULL;
}
