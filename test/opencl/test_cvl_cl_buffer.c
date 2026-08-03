#ifdef CVL_OPENCL

#include "../test_common.h"
#include "cvl_cl.h"

int main(void)
{
    cvl_cl_status_t status = CVL_CL_SUCCESS;
    cvl_cl_device_t device = {0};
    cvl_cl_ctx_t ctx = {0};
    cvl_cl_queue_t queue = {0};
    cvl_cl_buffer_t buf = {0};
    cvl_cl_buffer_t zero_buf = {0};
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

    /* ---- Queue (needed for buffer_reserve copy) ---- */
    CVL_CL_CHECK(cvl_cl_queue_create(&ctx, NULL, &queue), cleanup);

    /* ================================================================ */
    /*  Test 1: Create a 1024-byte READ_WRITE buffer                   */
    /* ================================================================ */
    CVL_CL_CHECK(cvl_cl_buffer_create(&ctx,
                                      &(cvl_cl_buffer_desc_t){
                                          .access = CVL_CL_BUF_READ_WRITE,
                                          .size_bytes = 1024,
                                      },
                                      &buf),
                 cleanup);

    TEST_ASSERT(buf.size == 1024, "Buffer size should be 1024, got %zu", buf.size);
    TEST_ASSERT(buf.capacity >= 1024, "Buffer capacity (%zu) should be >= 1024", buf.capacity);
    TEST_ASSERT(buf.mem != NULL, "cl_mem handle is NULL after creation");
    TEST_ASSERT(buf.access == CVL_CL_BUF_READ_WRITE, "Buffer access mode should be READ_WRITE");

    /* ================================================================ */
    /*  Test 2: Reserve - grow to 4096 bytes                           */
    /* ================================================================ */
    CVL_CL_CHECK(cvl_cl_buffer_reserve(&buf, &ctx, &queue, 4096), cleanup);

    TEST_ASSERT(buf.capacity >= 4096, "After reserve(4096), capacity (%zu) should be >= 4096", buf.capacity);
    /*
     * Size must remain unchanged after reserve (reserve only changes
     * capacity, not logical size).
     */
    TEST_ASSERT(buf.size == 1024, "Buffer size changed after reserve; expected 1024, got %zu", buf.size);

    /* ================================================================ */
    /*  Test 3: Zero-size buffer creation                               */
    /* ================================================================ */
    CVL_CL_CHECK(cvl_cl_buffer_create(&ctx,
                                      &(cvl_cl_buffer_desc_t){
                                          .access = CVL_CL_BUF_READ_ONLY,
                                          .size_bytes = 0,
                                      },
                                      &zero_buf),
                 cleanup);

    TEST_ASSERT(zero_buf.size == 0, "Zero-size buffer logical size should be 0, got %zu", zero_buf.size);
    TEST_ASSERT(zero_buf.capacity == 0, "Zero-size buffer capacity should be 0, got %zu", zero_buf.capacity);
    /*
     * The wrapper deliberately returns mem=NULL for zero-size buffers
     * (no underlying cl_mem is created). The handle is still valid for
     * destroy and query operations.
     */
    TEST_ASSERT(zero_buf.mem == NULL, "Zero-size buffer cl_mem handle should be NULL");

    /* ---- All tests passed ---- */
    status = CVL_CL_SUCCESS;

cleanup:
    cvl_cl_buffer_destroy(&zero_buf);
    cvl_cl_buffer_destroy(&buf);
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
