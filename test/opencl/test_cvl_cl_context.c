#ifdef CVL_OPENCL

#include "../test_common.h"
#include "cvl_cl.h"
#include "cvl_cl_test_common.h"

int main(void)
{
    cvl_cl_status_t status = CVL_CL_SUCCESS;
    cvl_cl_device_t device = {0};
    cl_context ctx = NULL;
    cl_command_queue queue = NULL;
    cl_command_queue prof_queue = NULL;

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
    TEST_ASSERT(ctx != NULL, "cl_context handle is NULL after creation");

    /* ---- Create command queue (default properties) ---- */
    CVL_CL_CHECK(cvl_cl_queue_create(ctx, device.id, NULL, &queue), cleanup);
    TEST_ASSERT(queue != NULL, "cl_command_queue handle is NULL after creation");

    /* ---- Create a second queue with explicit properties ---- */
    CVL_CL_CHECK(cvl_cl_queue_create(ctx, device.id, &(cvl_cl_queue_props_t){.profiling = true}, &prof_queue), cleanup);
    TEST_ASSERT(prof_queue != NULL, "profiling cl_command_queue handle is NULL after creation");

    /* ---- All good ---- */
    status = CVL_CL_SUCCESS;

cleanup:
    cvl_cl_queue_destroy(&prof_queue);
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
