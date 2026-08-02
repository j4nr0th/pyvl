#ifdef CVL_OPENCL

#include "../test_common.h"
#include "cvl_cl.h"

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
 * Invalid kernel – syntax error that must trigger a build failure.
 */
static const char *INVALID_KERNEL_SOURCE = "__kernel void broken(__global const double *a) {\n"
                                           "  this is not valid OpenCL C syntax\n"
                                           "}\n";

int main(void)
{
    cvl_cl_status_t status = CVL_CL_SUCCESS;
    cvl_cl_device_t device = {0};
    cvl_cl_ctx_t ctx = {0};
    cvl_cl_queue_t queue = {0};
    cvl_cl_program_t valid_prog = {0};
    cvl_cl_program_t invalid_prog = {0};
    unsigned count = 0;

    /* ---- Discover device (GPU preferred, CPU fallback) ---- */
    status = cvl_cl_device_discover(
        (cvl_cl_device_sel_t[]){
            {.type = CVL_CL_DEVICE_SEL_TYPE, .device_type = CL_DEVICE_TYPE_GPU},
            {},
        },
        1, &count, &device, NULL);
    if (status != CVL_CL_SUCCESS || count == 0)
    {
        status = cvl_cl_device_discover(
            (cvl_cl_device_sel_t[]){
                {.type = CVL_CL_DEVICE_SEL_TYPE, .device_type = CL_DEVICE_TYPE_CPU},
                {},
            },
            1, &count, &device, NULL);
    }
    if (status != CVL_CL_SUCCESS || count == 0)
    {
        fprintf(stderr, "No OpenCL device found – skipping test.\n");
        return 0;
    }

    /* ---- Context ---- */
    CVL_CL_CHECK(cvl_cl_ctx_create(&device, &ctx), cleanup);

    /* ---- Queue ---- */
    CVL_CL_CHECK(cvl_cl_queue_create(&ctx, NULL, &queue), cleanup);

    /* ================================================================ */
    /*  Test 1: Compile a valid program                                 */
    /* ================================================================ */
    CVL_CL_CHECK(cvl_cl_program_create(&ctx,
                                       &(cvl_cl_program_desc_t){
                                           .source_type = CVL_CL_PROGRAM_SOURCE_STRING,
                                           .source_string = VALID_KERNEL_SOURCE,
                                       },
                                       cvl_cl_device_id(&device), &valid_prog, NULL),
                 cleanup);
    TEST_ASSERT(cvl_cl_program_program(&valid_prog) != NULL, "Valid program handle is NULL after successful creation");
    TEST_ASSERT(cvl_cl_program_build_log(&valid_prog) == NULL, "Build log should be NULL when compilation succeeded");
    TEST_ASSERT(cvl_cl_program_ctx(&valid_prog) == &ctx, "Program context does not match");

    /* ================================================================ */
    /*  Test 2: Compile an invalid program – expect build failure + log */
    /* ================================================================ */
    status = cvl_cl_program_create(&ctx,
                                   &(cvl_cl_program_desc_t){
                                       .source_type = CVL_CL_PROGRAM_SOURCE_STRING,
                                       .source_string = INVALID_KERNEL_SOURCE,
                                   },
                                   cvl_cl_device_id(&device), &invalid_prog, NULL);
    TEST_ASSERT(status == CVL_CL_ERR_PROGRAM_BUILD, "Invalid kernel should yield PROGRAM_BUILD error, got %s",
                cvl_cl_status_str(status));

    const char *log = cvl_cl_program_build_log(&invalid_prog);
    TEST_ASSERT(log != NULL, "Build log is NULL after a failed compilation");
    TEST_ASSERT(strlen(log) > 0, "Build log is empty after a failed compilation");
    /* The log should contain some indication of the error. */
    TEST_ASSERT(strstr(log, "error") != NULL || strstr(log, "Error") != NULL || strstr(log, "syntax") != NULL ||
                    strstr(log, "Syntax") != NULL,
                "Build log should contain an error or syntax message, got: %s", log);

    /* ---- All tests passed ---- */
    status = CVL_CL_SUCCESS;

cleanup:
    cvl_cl_program_destroy(&invalid_prog);
    cvl_cl_program_destroy(&valid_prog);
    cvl_cl_queue_destroy(&queue);
    cvl_cl_ctx_destroy(&ctx);
    cvl_cl_device_destroy(&device);
    return status == CVL_CL_SUCCESS ? 0 : 1;
}

#else

#include <stdio.h>
int main(void)
{
    printf("OpenCL not available – skipping test.\n");
    return 0;
}

#endif
