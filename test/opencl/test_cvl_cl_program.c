#ifdef CVL_OPENCL

#include "../test_common.h"
#include "cvl_cl.h"
#include "cvl_cl_test_common.h"

#include <string.h>

/*
 * Trivial valid kernel that adds two vectors element-wise.
 */
static const char *VALID_KERNEL_SOURCE =
    "__kernel void add(__global const double *a, __global const double *b, __global double *c) {\n"
    "  int i = get_global_id(0);\n"
    "  c[i] = a[i] + b[i];\n"
    "}\n";

/*
 * Invalid kernel - syntax error that must trigger a build failure.
 */
static const char *INVALID_KERNEL_SOURCE = "__kernel void broken(__global const double *a) {\n"
                                           "  this is not valid OpenCL C syntax\n"
                                           "}\n";

int main(void)
{
    cvl_cl_status_t status = CVL_CL_SUCCESS;
    cvl_cl_device_t device = {0};
    cl_context ctx = NULL;
    cl_command_queue queue = NULL;
    cl_program valid_prog = NULL;
    cl_program invalid_prog = NULL;
    char build_log[4096];

    /* ---- Discover device (GPU preferred, CPU fallback) ---- */
    status = cvl_cl_device_first_gpu(&device);
    if (status != CVL_CL_SUCCESS)
    {
        status = cvl_cl_device_first_cpu(&device);
    }
    if (status != CVL_CL_SUCCESS)
    {
        fprintf(stderr, "No OpenCL device found - skipping test.\n");
        return 0;
    }

    /* ---- Context ---- */
    CVL_CL_CHECK(cvl_cl_ctx_create(&device, &ctx), cleanup);

    /* ---- Queue ---- */
    CVL_CL_CHECK(cvl_cl_queue_create(ctx, device.id, NULL, &queue), cleanup);

    /* ================================================================ */
    /*  Test 1: Compile a valid program                                 */
    /* ================================================================ */
    CVL_CL_CHECK(cvl_cl_program_create(ctx, device.id,
                                       &(cvl_cl_program_desc_t){
                                           .source_string = VALID_KERNEL_SOURCE,
                                       },
                                       NULL, 0, &valid_prog),
                 cleanup);
    TEST_ASSERT(valid_prog != NULL, "Valid program handle is NULL after successful creation");

    /* ================================================================ */
    /*  Test 2: Compile an invalid program - expect build failure + log */
    /* ================================================================ */
    memset(build_log, 0, sizeof build_log);
    status = cvl_cl_program_create(ctx, device.id,
                                   &(cvl_cl_program_desc_t){
                                       .source_string = INVALID_KERNEL_SOURCE,
                                   },
                                   build_log, sizeof build_log, &invalid_prog);
    TEST_ASSERT(status == CVL_CL_ERR_PROGRAM_BUILD, "Invalid kernel should yield PROGRAM_BUILD error, got %s",
                cvl_cl_status_str(status));

    /* The build log must be captured in the caller-provided buffer:
     * non-empty and NUL-terminated within capacity. */
    TEST_ASSERT(strlen(build_log) > 0, "Build log is empty after a failed compilation");
    TEST_ASSERT(strlen(build_log) < sizeof build_log, "Build log is not NUL-terminated within capacity");
    /* The log should contain some indication of the error. */
    TEST_ASSERT(strstr(build_log, "error") != NULL || strstr(build_log, "Error") != NULL ||
                    strstr(build_log, "syntax") != NULL || strstr(build_log, "Syntax") != NULL,
                "Build log should contain an error or syntax message, got: %s", build_log);

    /* ---- All tests passed ---- */
    status = CVL_CL_SUCCESS;

cleanup:
    cvl_cl_program_destroy(&invalid_prog);
    cvl_cl_program_destroy(&valid_prog);
    cvl_cl_queue_destroy(&queue);
    cvl_cl_ctx_destroy(&ctx);
    return status == CVL_CL_SUCCESS ? 0 : 1;
}

#else

#include <stdio.h>
int main(void)
{
    printf("OpenCL not available - skipping test.\n");
    return 0;
}

#endif
