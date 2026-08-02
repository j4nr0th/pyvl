#ifdef CVL_OPENCL

#include "../test_common.h"
#include "cvl_cl.h"

/*
 * SAXPY kernel: out[i] = a * x[i] + y[i]
 *
 * Has three buffer arguments and one scalar-double argument -
 * exercises both CVL_CL_KARG_BUFFER and CVL_CL_KARG_SCALAR_DOUBLE.
 */
static const char *SAXPY_KERNEL_SOURCE =
    "__kernel void saxpy(__global const double *x, __global const double *y, __global double *out, double a) {\n"
    "  int i = get_global_id(0);\n"
    "  out[i] = a * x[i] + y[i];\n"
    "}\n";

int main(void)
{
    cvl_cl_status_t status = CVL_CL_SUCCESS;
    cvl_cl_device_t device = {0};
    cvl_cl_ctx_t ctx = {0};
    cvl_cl_queue_t queue = {0};
    cvl_cl_program_t program = {0};
    cvl_cl_kernel_t kernel = {0};
    cvl_cl_buffer_t buf = {0};
    unsigned count = 0;

    /* ---- Discover device ---- */
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
    CVL_CL_CHECK(cvl_cl_queue_create(&ctx, NULL, &queue), cleanup);

    /* ---- Program (saxpy) ---- */
    CVL_CL_CHECK(cvl_cl_program_create(&ctx,
                                       &(cvl_cl_program_desc_t){
                                           .source_type = CVL_CL_PROGRAM_SOURCE_STRING,
                                           .source_string = SAXPY_KERNEL_SOURCE,
                                       },
                                       device.id, &program, NULL),
                 cleanup);

    /* ---- Kernel creation ---- */
    CVL_CL_CHECK(cvl_cl_kernel_create(&program, "saxpy", &kernel), cleanup);
    TEST_ASSERT(cvl_cl_kernel_kernel(&kernel) != NULL, "cl_kernel handle is NULL after creation");
    TEST_ASSERT(cvl_cl_kernel_program(&kernel) == &program, "cvl_cl_kernel_program does not match");

    /* ---- preferred_wg_multiple should be populated (> 0) ---- */
    TEST_ASSERT(cvl_cl_kernel_preferred_wg_multiple(&kernel) > 0, "preferred_wg_multiple should be > 0, got %zu",
                cvl_cl_kernel_preferred_wg_multiple(&kernel));

    /* ---- Create a small buffer so we have a valid cl_mem to pass ---- */
    CVL_CL_CHECK(cvl_cl_buffer_create(&ctx,
                                      &(cvl_cl_buffer_desc_t){
                                          .access = CVL_CL_BUF_READ_WRITE,
                                          .size_bytes = 256,
                                      },
                                      &buf),
                 cleanup);

    /* ---- Set kernel arguments via typed descriptor array ---- */
    CVL_CL_CHECK(cvl_cl_kernel_set_args(&kernel,
                                        (cvl_cl_karg_t[]){
                                            {.type = CVL_CL_KARG_BUFFER, .index = 0, .mem = cvl_cl_buffer_mem(&buf)},
                                            {.type = CVL_CL_KARG_BUFFER, .index = 1, .mem = cvl_cl_buffer_mem(&buf)},
                                            {.type = CVL_CL_KARG_BUFFER, .index = 2, .mem = cvl_cl_buffer_mem(&buf)},
                                            {.type = CVL_CL_KARG_SCALAR_DOUBLE, .index = 3, .scalar_double = 2.0},
                                            {},
                                        }),
                 cleanup);

    /* ---- All tests passed ---- */
    status = CVL_CL_SUCCESS;

cleanup:
    cvl_cl_buffer_destroy(&buf);
    cvl_cl_kernel_destroy(&kernel);
    cvl_cl_program_destroy(&program);
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
