/*
 * test_cvl_cl_chain.c — Tests for the cvl_cl_chain_t dependency-chaining
 * object (cvl_cl_chain.h).
 *
 * Pipeline tested:
 *   1. write → ndrange → read through a single chain; results match CPU.
 *   2. is_ready(): may be false while ops are pending; must be true after
 *      cvl_cl_chain_finish.
 *   3. Barrier compaction: more than CL_MAX_WAIT_EVENTS chained operations
 *      keep the pending list bounded (pending events are compacted into a
 *      single marker event).
 *   4. Extra wait events: a chained op can wait on a caller-supplied event
 *      (produced by a raw cvl_cl_write_buffer on a second queue); ordering
 *      is verified by the result.
 *
 * The test skips gracefully (returns 0) when no OpenCL device is available.
 */

#ifdef CVL_OPENCL

#include "../test_common.h"
#include "cvl_cl_test_common.h"

#include <math.h>
#include <string.h>

/* Minimal saxpy kernel (FP32) for the chain pipeline tests. */
static const char *SAXPY_SOURCE =
    "__kernel void saxpy(__global const float *x, __global float *y, float a, unsigned n)\n"
    "{\n"
    "    unsigned i = get_global_id(0);\n"
    "    if (i < n) y[i] = a * x[i] + y[i];\n"
    "}\n";

int main(void)
{
    cvl_cl_status_t status = CVL_CL_SUCCESS;
    cvl_cl_device_t device = {0};
    cl_context ctx = NULL;
    cl_command_queue queue = NULL;
    cl_command_queue queue2 = NULL;
    cl_program program = NULL;
    cl_kernel kernel = NULL;
    cvl_cl_buffer_t buf_x = {0};
    cvl_cl_buffer_t buf_y = {0};
    cvl_cl_buffer_t buf_flags = {0};
    cvl_cl_chain_t chain = {0};
    int ret = 1;

    /* ----------------------------------------------------------------- */
    /* 1. Device discovery (GPU preferred, CPU fallback)                 */
    /* ----------------------------------------------------------------- */
    status = cvl_cl_device_first_gpu(&device);
    if (status != CVL_CL_SUCCESS)
        status = cvl_cl_device_first_cpu(&device);
    if (status != CVL_CL_SUCCESS)
    {
        fprintf(stderr, "No OpenCL device found -- skipping chain test.\n");
        return 0;
    }

    CVL_CL_CHECK(cvl_cl_ctx_create(&device, &ctx), cleanup);
    CVL_CL_CHECK(cvl_cl_queue_create(ctx, device.id, NULL, &queue), cleanup);
    /* Second queue used by the extra-wait-event test (cross-queue sync). */
    CVL_CL_CHECK(cvl_cl_queue_create(ctx, device.id, NULL, &queue2), cleanup);

    /* ----------------------------------------------------------------- */
    /* 2. Build the saxpy program + kernel directly                       */
    /* ----------------------------------------------------------------- */
    {
        char log[4096];
        CVL_CL_CHECK(cvl_cl_program_create(ctx, device.id,
                                           &(cvl_cl_program_desc_t){.source_string = SAXPY_SOURCE,
                                                                    .build_options = NULL,
                                                                    .precision = CVL_CL_PRECISION_FP32},
                                           log, sizeof log, &program),
                     cleanup);
        CVL_CL_CHECK(cvl_cl_kernel_create(program, "saxpy", &kernel), cleanup);
    }

    /* ----------------------------------------------------------------- */
    /* 3. Pipeline test: write → ndrange → read through one chain        */
    /* ----------------------------------------------------------------- */
    enum
    {
        N = 256,
        A = 3
    };
    float x_host[N];
    float y_host[N];
    float y_out[N];
    for (unsigned i = 0; i < N; ++i)
    {
        x_host[i] = (float)(i + 1);
        y_host[i] = (float)(2 * (i + 1));
    }

    CVL_CL_CHECK(
        cvl_cl_buffer_create(
            ctx, &(cvl_cl_buffer_desc_t){.access = CVL_CL_BUF_READ_ONLY, .size_bytes = N * sizeof(float)}, &buf_x),
        cleanup);
    CVL_CL_CHECK(
        cvl_cl_buffer_create(
            ctx, &(cvl_cl_buffer_desc_t){.access = CVL_CL_BUF_READ_WRITE, .size_bytes = N * sizeof(float)}, &buf_y),
        cleanup);

    cvl_cl_chain_init(&chain, queue);
    CVL_CL_CHECK(cvl_cl_chain_write_buffer(&chain, &buf_x, 0, N * sizeof(float), x_host, 0, NULL, NULL), cleanup);
    CVL_CL_CHECK(cvl_cl_chain_write_buffer(&chain, &buf_y, 0, N * sizeof(float), y_host, 0, NULL, NULL), cleanup);

    /* is_ready may legitimately be false here, but on a very fast device
     * the writes may already have completed — so this is informational. */
    if (!cvl_cl_chain_is_ready(&chain))
        printf("chain is_ready false while ops pending (as expected)\n");
    else
        printf("chain is_ready true already (ops completed before the check)\n");

    const size_t global = N;
    CVL_CL_CHECK(cvl_cl_chain_ndrange(&chain, kernel, 1, &global, NULL,
                                      (cvl_cl_karg_t[]){
                                          {.type = CVL_CL_KARG_BUFFER, .index = 0, .mem = buf_x.mem},
                                          {.type = CVL_CL_KARG_BUFFER, .index = 1, .mem = buf_y.mem},
                                          {.type = CVL_CL_KARG_SCALAR_FLOAT, .index = 2, .scalar_float = (float)A},
                                          {.type = CVL_CL_KARG_SCALAR_UINT, .index = 3, .scalar_uint = N},
                                          {},
                                      },
                                      0, NULL, NULL),
                 cleanup);
    CVL_CL_CHECK(cvl_cl_chain_read_buffer(&chain, &buf_y, 0, N * sizeof(float), y_out, 0, NULL, NULL), cleanup);

    CVL_CL_CHECK(cvl_cl_chain_finish(&chain), cleanup);
    TEST_ASSERT(cvl_cl_chain_is_ready(&chain), "chain must be ready after finish");

    /* Verify against CPU: y = A*x + y. */
    for (unsigned i = 0; i < N; ++i)
    {
        const float expected = (float)A * x_host[i] + y_host[i];
        TEST_ASSERT(fabsf(y_out[i] - expected) < 1e-5f, "saxpy mismatch at %u: got %.6f expected %.6f", i, y_out[i],
                    expected);
    }
    printf("chain write->ndrange->read pipeline OK (N=%u)\n", N);

    /* ----------------------------------------------------------------- */
    /* 4. Barrier compaction: 20 chained writes without finishing        */
    /* ----------------------------------------------------------------- */
    {
        enum
        {
            N_FLAGS = 64,
            N_OPS = 20
        };
        uint8_t flags_out[N_FLAGS];
        const uint8_t mark = 0xAB;

        CVL_CL_CHECK(cvl_cl_buffer_create(
                         ctx, &(cvl_cl_buffer_desc_t){.access = CVL_CL_BUF_READ_WRITE, .size_bytes = sizeof flags_out},
                         &buf_flags),
                     cleanup);

        /* Enqueue more ops than CL_MAX_WAIT_EVENTS (16) — the pending list
         * must be compacted into a marker event, keeping n_pending bounded. */
        for (unsigned i = 0; i < N_OPS; ++i)
            CVL_CL_CHECK(cvl_cl_chain_write_buffer(&chain, &buf_flags, i, 1, &mark, 0, NULL, NULL), cleanup);

        TEST_ASSERT(chain.n_pending > 0 && chain.n_pending <= CL_MAX_WAIT_EVENTS,
                    "pending list not compacted after %u ops: n_pending=%u", N_OPS, chain.n_pending);
        printf("chain compaction: %u ops enqueued without finish -> n_pending=%u (bounded by %d)\n", N_OPS,
               chain.n_pending, CL_MAX_WAIT_EVENTS);

        CVL_CL_CHECK(cvl_cl_chain_finish(&chain), cleanup);
        TEST_ASSERT(cvl_cl_chain_is_ready(&chain), "chain must be ready after finish (compaction)");

        memset(flags_out, 0, sizeof flags_out);
        CVL_CL_CHECK(cvl_cl_read_buffer(queue, &buf_flags, 0, sizeof flags_out, flags_out, 0, NULL, NULL), cleanup);
        CVL_CL_CHECK(cvl_cl_finish(queue), cleanup);
        for (unsigned i = 0; i < N_OPS; ++i)
            TEST_ASSERT(flags_out[i] == mark, "flag byte %u not written (got 0x%02x)", i, flags_out[i]);
        printf("chain compaction: all %u writes verified after finish\n", N_OPS);
    }

    /* ----------------------------------------------------------------- */
    /* 5. Extra wait events: chained read waits on a raw write from a    */
    /*    second queue (cross-queue dependency via the event wait list)  */
    /* ----------------------------------------------------------------- */
    {
        const float x1 = 2.0f;   /* kernel input (written via the chain). */
        const float seed = 5.0f; /* raw write on queue2 into buf_y[0]. */
        float y1_out = 0.0f;
        cvl_cl_event_t ev = {0};

        /* Chain on queue: write x[0], then ndrange y = A*x + y = 3*2 + 0 = 6. */
        cvl_cl_chain_init(&chain, queue);
        CVL_CL_CHECK(cvl_cl_chain_write_buffer(&chain, &buf_x, 0, sizeof(float), &x1, 0, NULL, NULL), cleanup);
        {
            const size_t g = 1;
            CVL_CL_CHECK(cvl_cl_chain_ndrange(&chain, kernel, 1, &g, NULL,
                                              (cvl_cl_karg_t[]){
                                                  {.type = CVL_CL_KARG_BUFFER, .index = 0, .mem = buf_x.mem},
                                                  {.type = CVL_CL_KARG_BUFFER, .index = 1, .mem = buf_y.mem},
                                                  {.type = CVL_CL_KARG_SCALAR_FLOAT, .index = 2, .scalar_float = 3.0f},
                                                  {.type = CVL_CL_KARG_SCALAR_UINT, .index = 3, .scalar_uint = 1},
                                                  {},
                                              },
                                              0, NULL, NULL),
                         cleanup);
        }

        /* Raw write on the SECOND queue — produces the external event. */
        CVL_CL_CHECK(cvl_cl_write_buffer(queue2, &buf_y, 0, sizeof(float), &seed, 0, NULL, &ev), cleanup);

        /* The chained read waits on the external event.  Only the wait
         * list makes the result deterministic: without it the read on
         * queue could overtake the write on queue2. */
        CVL_CL_CHECK(cvl_cl_chain_read_buffer(&chain, &buf_y, 0, sizeof(float), &y1_out, 1, &ev, NULL), cleanup);
        CVL_CL_CHECK(cvl_cl_chain_finish(&chain), cleanup);
        cvl_cl_event_release(&ev);

        /* Kernel wrote 3*2 + 0 = 6 first, then queue2 wrote 5 — the read
         * must observe 5 (i.e. it waited on the external event). */
        TEST_ASSERT(fabsf(y1_out - 5.0f) < 1e-5f, "extra-wait ordering broken: got %.6f expected 5.0", y1_out);
        printf("chain extra wait event OK (y=%.2f after kernel then raw write)\n", y1_out);
    }

    printf("All chain tests passed.\n");
    ret = 0;

cleanup:
    cvl_cl_chain_destroy(&chain);
    cvl_cl_buffer_destroy(&buf_flags);
    cvl_cl_buffer_destroy(&buf_y);
    cvl_cl_buffer_destroy(&buf_x);
    cvl_cl_kernel_destroy(&kernel);
    cvl_cl_program_destroy(&program);
    cvl_cl_queue_destroy(&queue2);
    cvl_cl_queue_destroy(&queue);
    cvl_cl_ctx_destroy(&ctx);
    return ret;
}

#else

#include <stdio.h>
int main(void)
{
    printf("OpenCL not available -- skipping chain test.\n");
    return 0;
}

#endif /* CVL_OPENCL */
