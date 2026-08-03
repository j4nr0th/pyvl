#ifdef CVL_OPENCL

#include "../test_common.h"
#include "cvl_cl.h"
#include "cvl_cl_test_common.h"

#include <string.h>

enum
{
    N = 16, /* Number of elements. */
};

/*
 * Simple element-wise add kernel.
 */
static const char *ADD_KERNEL_SOURCE =
    "__kernel void add(__global const double *a, __global const double *b, __global double *c) {\n"
    "  int i = get_global_id(0);\n"
    "  c[i] = a[i] + b[i];\n"
    "}\n";

int main(void)
{
    cvl_cl_status_t status = CVL_CL_SUCCESS;
    cvl_cl_device_t device = {0};
    cl_context ctx = NULL;
    cl_command_queue queue = NULL;
    cl_program program = NULL;
    cl_kernel kernel = NULL;
    cvl_cl_buffer_t buf_a = {0};
    cvl_cl_buffer_t buf_b = {0};
    cvl_cl_buffer_t buf_c = {0};

    const size_t buf_bytes = N * sizeof(double);
    double host_a[N];
    double host_b[N];
    double host_c[N];

    /* ---- Initialise host data ---- */
    for (int i = 0; i < N; ++i)
    {
        host_a[i] = (double)i;
        host_b[i] = (double)(2 * i);
    }

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
    CVL_CL_CHECK(cvl_cl_queue_create(ctx, device.id, NULL, &queue), cleanup);

    /* ---- Program ---- */
    CVL_CL_CHECK(cvl_cl_program_create(ctx, device.id,
                                       &(cvl_cl_program_desc_t){
                                           .source_string = ADD_KERNEL_SOURCE,
                                       },
                                       NULL, 0, &program),
                 cleanup);

    /* ---- Kernel ---- */
    CVL_CL_CHECK(cvl_cl_kernel_create(program, "add", &kernel), cleanup);

    /* ---- Create three device buffers (a, b input; c output) ---- */
    CVL_CL_CHECK(cvl_cl_buffer_create(ctx,
                                      &(cvl_cl_buffer_desc_t){
                                          .access = CVL_CL_BUF_READ_ONLY,
                                          .size_bytes = buf_bytes,
                                      },
                                      &buf_a),
                 cleanup);
    CVL_CL_CHECK(cvl_cl_buffer_create(ctx,
                                      &(cvl_cl_buffer_desc_t){
                                          .access = CVL_CL_BUF_READ_ONLY,
                                          .size_bytes = buf_bytes,
                                      },
                                      &buf_b),
                 cleanup);
    CVL_CL_CHECK(cvl_cl_buffer_create(ctx,
                                      &(cvl_cl_buffer_desc_t){
                                          .access = CVL_CL_BUF_WRITE_ONLY,
                                          .size_bytes = buf_bytes,
                                      },
                                      &buf_c),
                 cleanup);

    /* ---- Write host data to device buffers ---- */
    CVL_CL_CHECK(cvl_cl_write_buffer(queue, &buf_a, 0, buf_bytes, host_a, 0, NULL, NULL), cleanup);
    CVL_CL_CHECK(cvl_cl_write_buffer(queue, &buf_b, 0, buf_bytes, host_b, 0, NULL, NULL), cleanup);

    /* ---- Launch kernel via cvl_cl_ndrange ---- */
    {
        const size_t global_work = N;
        const size_t local_work = N; /* Works for N <= max work-group size. */

        CVL_CL_CHECK(cvl_cl_ndrange(queue, kernel, 1, /* dims */
                                    &global_work,     /* global work size */
                                    &local_work,      /* local work size (explicit) */
                                    (cvl_cl_karg_t[]){
                                        {.type = CVL_CL_KARG_BUFFER, .index = 0, .mem = buf_a.mem},
                                        {.type = CVL_CL_KARG_BUFFER, .index = 1, .mem = buf_b.mem},
                                        {.type = CVL_CL_KARG_BUFFER, .index = 2, .mem = buf_c.mem},
                                        {},
                                    },
                                    0, NULL, NULL),
                     cleanup);
    }

    /* ---- Flush and finish ---- */
    CVL_CL_CHECK(cvl_cl_flush(queue), cleanup);
    CVL_CL_CHECK(cvl_cl_finish(queue), cleanup);

    /* ---- Read back result ---- */
    memset(host_c, 0, buf_bytes);
    CVL_CL_CHECK(cvl_cl_read_buffer(queue, &buf_c, 0, buf_bytes, host_c, 0, NULL, NULL), cleanup);

    /* ---- Ensure finish (read is async; finish to guarantee completion) ---- */
    CVL_CL_CHECK(cvl_cl_finish(queue), cleanup);

    /* ---- Verify ---- */
    for (int i = 0; i < N; ++i)
    {
        const double expected = host_a[i] + host_b[i];
        TEST_ASSERT(host_c[i] == expected, "c[%d] = %g, expected %g (a[%d]=%g, b[%d]=%g)", i, host_c[i], expected, i,
                    host_a[i], i, host_b[i]);
    }

    /* ---- All tests passed ---- */
    status = CVL_CL_SUCCESS;

cleanup:
    cvl_cl_buffer_destroy(&buf_c);
    cvl_cl_buffer_destroy(&buf_b);
    cvl_cl_buffer_destroy(&buf_a);
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
