#ifdef CVL_OPENCL

#include "../test_common.h"
#include "cvl_cl.h"

int main(void)
{
    cvl_cl_status_t status = CVL_CL_SUCCESS;
    cvl_cl_device_t device = {0};
    cvl_cl_ctx_t ctx = {0};
    cvl_cl_queue_t queue = {0};
    unsigned count = 0;

    /* ---- Discover a device (GPU preferred, CPU fallback) ---- */
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

    /* ---- Create context ---- */
    CVL_CL_CHECK(cvl_cl_ctx_create(&device, &ctx), cleanup);
    TEST_ASSERT(cvl_cl_ctx_context(&ctx) != NULL, "cl_context handle is NULL after creation");
    TEST_ASSERT(cvl_cl_ctx_device(&ctx) == &device, "cvl_cl_ctx_device does not match the device used at creation");

    /* ---- Create command queue (default properties) ---- */
    CVL_CL_CHECK(cvl_cl_queue_create(&ctx, NULL, &queue), cleanup);
    TEST_ASSERT(cvl_cl_queue_queue(&queue) != NULL, "cl_command_queue handle is NULL after creation");
    TEST_ASSERT(cvl_cl_queue_ctx(&queue) == &ctx, "cvl_cl_queue_ctx does not match the context used at creation");

    /* ---- All good ---- */
    status = CVL_CL_SUCCESS;

cleanup:
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
