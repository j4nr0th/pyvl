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

    TEST_ASSERT(cvl_cl_buffer_size(&buf) == 1024, "Buffer size should be 1024, got %zu", cvl_cl_buffer_size(&buf));
    TEST_ASSERT(cvl_cl_buffer_capacity(&buf) >= 1024, "Buffer capacity (%zu) should be >= 1024",
                cvl_cl_buffer_capacity(&buf));
    TEST_ASSERT(cvl_cl_buffer_mem(&buf) != NULL, "cl_mem handle is NULL after creation");
    TEST_ASSERT(cvl_cl_buffer_access(&buf) == CVL_CL_BUF_READ_WRITE, "Buffer access mode should be READ_WRITE");

    /* ================================================================ */
    /*  Test 2: Reserve – grow to 4096 bytes                           */
    /* ================================================================ */
    CVL_CL_CHECK(cvl_cl_buffer_reserve(&buf, &ctx, &queue, 4096), cleanup);

    TEST_ASSERT(cvl_cl_buffer_capacity(&buf) >= 4096, "After reserve(4096), capacity (%zu) should be >= 4096",
                cvl_cl_buffer_capacity(&buf));
    /*
     * Size must remain unchanged after reserve (reserve only changes
     * capacity, not logical size).
     */
    TEST_ASSERT(cvl_cl_buffer_size(&buf) == 1024, "Buffer size changed after reserve; expected 1024, got %zu",
                cvl_cl_buffer_size(&buf));

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

    TEST_ASSERT(cvl_cl_buffer_size(&zero_buf) == 0, "Zero-size buffer logical size should be 0, got %zu",
                cvl_cl_buffer_size(&zero_buf));
    TEST_ASSERT(cvl_cl_buffer_capacity(&zero_buf) == 0, "Zero-size buffer capacity should be 0, got %zu",
                cvl_cl_buffer_capacity(&zero_buf));
    /*
     * The wrapper deliberately returns mem=NULL for zero-size buffers
     * (no underlying cl_mem is created). The handle is still valid for
     * destroy and query operations.
     */
    TEST_ASSERT(cvl_cl_buffer_mem(&zero_buf) == NULL, "Zero-size buffer cl_mem handle should be NULL");

    /* ---- All tests passed ---- */
    status = CVL_CL_SUCCESS;

cleanup:
    cvl_cl_buffer_destroy(&zero_buf);
    cvl_cl_buffer_destroy(&buf);
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
