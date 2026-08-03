/*
 * test_cvl_cl_direct_sum.c — End-to-end test of the direct N-body
 * OpenCL evaluation using the cl_compute API.
 *
 * Generates random sources + targets on the host, runs the GPU
 * direct-sum kernel via the cvl_cl_compute backend + staging
 * buffers, and compares results with the equivalent CPU direct
 * sum using particle_kernel().
 */

#ifdef CVL_OPENCL

#include "../test_common.h"
#include "cvl_cl.h"
#include "cvl_cl_compute.h"
#include "cvl_cl_staging_buffer.h"
#include "cvl_cl_test_common.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

int main(void)
{
    cvl_cl_status_t status = CVL_CL_SUCCESS;
    cvl_cl_device_t device = {0};
    cl_context ctx = NULL;
    cl_command_queue queue = NULL;
    cvl_cl_compute_t comp = {0};
    cvl_cl_chain_t chain = {0};
    int ret = 1;

    status = cvl_cl_device_first_gpu(&device);
    if (status != CVL_CL_SUCCESS)
    {
        status = cvl_cl_device_first_cpu(&device);
    }
    if (status != CVL_CL_SUCCESS)
    {
        fprintf(stderr, "No OpenCL device found -- skipping test.\n");
        return 0;
    }

    CVL_CL_CHECK(cvl_cl_ctx_create(&device, &ctx), cleanup);
    CVL_CL_CHECK(cvl_cl_queue_create(ctx, device.id, NULL, &queue), cleanup);
    cvl_cl_chain_init(&chain, queue);

    /* Init compute backend (FP64, DIRECT_SUM pack - kernel source embedded) */
    CVL_CL_CHECK(
        cvl_cl_compute_init(&comp, ctx, queue, &device, CVL_CL_PRECISION_FP64, (const char *[]){"direct_sum"}, 1),
        cleanup);

    /* Staging buffers */
    cvl_cl_staging_buffer_t buf_targets, buf_sources_pos, buf_sources_val, buf_results;

    /* Generate test data */
    uint64_t rng = 12345;
    enum
    {

        N_TARGETS = 50,
        N_SOURCES = 200,
    };

    cvl_cl_staging_buffer_init(&buf_targets, CVL_CL_PRECISION_FP64);
    cvl_cl_staging_buffer_init(&buf_sources_pos, CVL_CL_PRECISION_FP64);
    cvl_cl_staging_buffer_init(&buf_sources_val, CVL_CL_PRECISION_FP64);
    cvl_cl_staging_buffer_init(&buf_results, CVL_CL_PRECISION_FP64);

    real3_t targets[N_TARGETS];
    real3_t sources_pos[N_SOURCES];
    real3_t sources_val[N_SOURCES];
    real3_t gpu_results[N_TARGETS];
    real3_t cpu_results[N_TARGETS];

    for (unsigned i = 0; i < N_TARGETS; ++i)
    {
        targets[i].x = xorshift_uniform_range(&rng, -5.0, 5.0);
        targets[i].y = xorshift_uniform_range(&rng, -5.0, 5.0);
        targets[i].z = xorshift_uniform_range(&rng, -5.0, 5.0);
    }
    for (unsigned i = 0; i < N_SOURCES; ++i)
    {
        sources_pos[i].x = xorshift_uniform_range(&rng, -5.0, 5.0);
        sources_pos[i].y = xorshift_uniform_range(&rng, -5.0, 5.0);
        sources_pos[i].z = xorshift_uniform_range(&rng, -5.0, 5.0);
        sources_val[i].x = xorshift_uniform_range(&rng, -1.0, 1.0);
        sources_val[i].y = xorshift_uniform_range(&rng, -1.0, 1.0);
        sources_val[i].z = xorshift_uniform_range(&rng, -1.0, 1.0);
    }

    /* Reserve staging buffers */
    CVL_CL_CHECK(cvl_cl_staging_buffer_reserve(&buf_targets, ctx, queue, N_TARGETS), cleanup);
    CVL_CL_CHECK(cvl_cl_staging_buffer_reserve(&buf_sources_pos, ctx, queue, N_SOURCES), cleanup);
    CVL_CL_CHECK(cvl_cl_staging_buffer_reserve(&buf_sources_val, ctx, queue, N_SOURCES), cleanup);
    CVL_CL_CHECK(cvl_cl_staging_buffer_reserve(&buf_results, ctx, queue, N_TARGETS), cleanup);

    /* Async upload through the chain (FP64: no float scratch) */
    CVL_CL_CHECK(cvl_cl_staging_buffer_write_async(&buf_targets, &chain, targets, NULL, N_TARGETS, 0, NULL), cleanup);
    CVL_CL_CHECK(cvl_cl_staging_buffer_write_async(&buf_sources_pos, &chain, sources_pos, NULL, N_SOURCES, 0, NULL),
                 cleanup);
    CVL_CL_CHECK(cvl_cl_staging_buffer_write_async(&buf_sources_val, &chain, sources_val, NULL, N_SOURCES, 0, NULL),
                 cleanup);
    CVL_CL_CHECK(cvl_cl_chain_finish(&chain), cleanup);

    /* Launch kernel (raw cl_kernel from the compute registry) */
    {
        cl_kernel k = cvl_cl_compute_kernel(&comp, CVL_CL_PACK_DIRECT_SUM, CVL_CL_DIRECT_SUM_KERNEL);
        TEST_ASSERT(k != NULL, "kernel 'direct_sum' not found");

        const size_t global = N_TARGETS;
        CVL_CL_CHECK(
            cvl_cl_chain_ndrange(&chain, k, 1, &global, NULL,
                                 (cvl_cl_karg_t[]){
                                     {.type = CVL_CL_KARG_BUFFER, .index = 0, .mem = buf_targets.device.mem},
                                     {.type = CVL_CL_KARG_BUFFER, .index = 1, .mem = buf_sources_pos.device.mem},
                                     {.type = CVL_CL_KARG_BUFFER, .index = 2, .mem = buf_sources_val.device.mem},
                                     {.type = CVL_CL_KARG_SCALAR_UINT, .index = 3, .scalar_uint = N_SOURCES},
                                     {.type = CVL_CL_KARG_SCALAR_UINT, .index = 4, .scalar_uint = N_TARGETS},
                                     {.type = CVL_CL_KARG_BUFFER, .index = 5, .mem = buf_results.device.mem},
                                     {},
                                 },
                                 0, NULL, NULL),
            cleanup);
    }

    /* Sync download through the chain */
    CVL_CL_CHECK(cvl_cl_staging_buffer_read_and_wait(&buf_results, &chain, gpu_results, NULL, N_TARGETS, 0), cleanup);

    /* CPU reference */
    memset(cpu_results, 0, sizeof(cpu_results));
    for (unsigned t = 0; t < N_TARGETS; ++t)
    {
        real3_t acc = {0, 0, 0};
        for (unsigned s = 0; s < N_SOURCES; ++s)
        {
            real3_t dr = real3_sub(targets[t], sources_pos[s]);
            acc = real3_add(acc, particle_kernel(sources_val[s], dr));
        }
        cpu_results[t] = acc;
    }

    /* Compare GPU vs CPU */
    double max_abs_err = 0.0, max_rel_err = 0.0;
    unsigned max_err_idx = 0;
    for (unsigned t = 0; t < N_TARGETS; ++t)
    {
        double dx = fabs(gpu_results[t].x - cpu_results[t].x);
        double dy = fabs(gpu_results[t].y - cpu_results[t].y);
        double dz = fabs(gpu_results[t].z - cpu_results[t].z);
        double abs_err = dx > dy ? (dx > dz ? dx : dz) : (dy > dz ? dy : dz);
        double ref =
            fabs(cpu_results[t].x) > fabs(cpu_results[t].y)
                ? (fabs(cpu_results[t].x) > fabs(cpu_results[t].z) ? fabs(cpu_results[t].x) : fabs(cpu_results[t].z))
                : (fabs(cpu_results[t].y) > fabs(cpu_results[t].z) ? fabs(cpu_results[t].y) : fabs(cpu_results[t].z));
        double rel_err = abs_err / (ref + 1e-30);
        if (abs_err > max_abs_err)
        {
            max_abs_err = abs_err;
            max_rel_err = rel_err;
            max_err_idx = t;
        }
    }

    printf("Direct sum: %u targets, %u sources: max_abs_err=%.2e, max_rel_err=%.2e\n", N_TARGETS, N_SOURCES,
           max_abs_err, max_rel_err);

    TEST_ASSERT(max_abs_err < 1e-14 || max_rel_err < 1e-12,
                "GPU direct sum mismatch: t=%u max_abs_err=%.2e max_rel_err=%.2e", max_err_idx, max_abs_err,
                max_rel_err);

    printf("All direct sum tests passed.\n");
    ret = 0;

cleanup:
    cvl_cl_staging_buffer_destroy(&buf_results);
    cvl_cl_staging_buffer_destroy(&buf_sources_val);
    cvl_cl_staging_buffer_destroy(&buf_sources_pos);
    cvl_cl_staging_buffer_destroy(&buf_targets);
    cvl_cl_chain_destroy(&chain);
    cvl_cl_compute_destroy(&comp);
    cvl_cl_queue_destroy(&queue);
    cvl_cl_ctx_destroy(&ctx);
    return ret;
}

#else

#include <stdio.h>
int main(void)
{
    printf("OpenCL not available -- skipping test.\n");
    return 0;
}

#endif
