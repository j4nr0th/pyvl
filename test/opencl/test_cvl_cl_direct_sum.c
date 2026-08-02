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

#include <math.h>
#include <stdlib.h>
#include <string.h>

static const char *DIRECT_SUM_SOURCE = "#ifdef CVL_CL_REAL_FP32\n"
                                       "typedef float real_t;\n"
                                       "#else\n"
                                       "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
                                       "typedef double real_t;\n"
                                       "#endif\n"
                                       "typedef struct { real_t x, y, z; } ds_real3_t;\n"
                                       "__kernel void direct_sum(\n"
                                       "    __global const real_t *targets,\n"
                                       "    __global const real_t *sources_pos,\n"
                                       "    __global const real_t *sources_val,\n"
                                       "    unsigned n_sources,\n"
                                       "    unsigned n_targets,\n"
                                       "    __global real_t *results\n"
                                       ") {\n"
                                       "    unsigned tid = get_global_id(0);\n"
                                       "    if (tid >= n_targets) return;\n"
                                       "    ds_real3_t pt;\n"
                                       "    pt.x = targets[3*tid]; pt.y = targets[3*tid+1]; pt.z = targets[3*tid+2];\n"
                                       "    ds_real3_t acc = {0, 0, 0};\n"
                                       "    for (unsigned i = 0; i < n_sources; ++i) {\n"
                                       "        ds_real3_t dr;\n"
                                       "        dr.x = pt.x - sources_pos[3*i];\n"
                                       "        dr.y = pt.y - sources_pos[3*i+1];\n"
                                       "        dr.z = pt.z - sources_pos[3*i+2];\n"
                                       "        real_t r2 = dr.x*dr.x + dr.y*dr.y + dr.z*dr.z;\n"
                                       "        if (r2 > (real_t)1e-30) {\n"
                                       "            real_t inv_r2 = (real_t)1.0 / r2;\n"
                                       "            acc.x += sources_val[3*i] * inv_r2;\n"
                                       "            acc.y += sources_val[3*i+1] * inv_r2;\n"
                                       "            acc.z += sources_val[3*i+2] * inv_r2;\n"
                                       "        }\n"
                                       "    }\n"
                                       "    results[3*tid]   = acc.x;\n"
                                       "    results[3*tid+1] = acc.y;\n"
                                       "    results[3*tid+2] = acc.z;\n"
                                       "}\n";

int main(void)
{
    cvl_cl_status_t status = CVL_CL_SUCCESS;
    cvl_cl_device_t device = {0};
    cvl_cl_ctx_t ctx = {0};
    cvl_cl_queue_t queue = {0};
    cvl_cl_compute_t comp = {0};
    unsigned count = 0;
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
    CVL_CL_CHECK(cvl_cl_queue_create(&ctx, NULL, &queue), cleanup);

    /* Init compute backend (FP64, single kernel) */
    {
        const char *kernels[] = {"direct_sum"};
        CVL_CL_CHECK(
            cvl_cl_compute_init(&comp, &ctx, &queue, &device, CVL_CL_PRECISION_FP64, DIRECT_SUM_SOURCE, kernels, 1),
            cleanup);
    }

    /* Staging buffers */
    cvl_cl_staging_buffer_t buf_targets, buf_sources_pos, buf_sources_val, buf_results;

    /* Generate test data */
    uint64_t rng = 12345;
    enum
    {

        N_TARGETS = 50,
        N_SOURCES = 200,
    };

    cvl_cl_staging_buffer_init(&buf_targets, CVL_CL_PRECISION_FP64, cvl_cl_compute_unified_memory(&comp), N_TARGETS,
                               NULL);
    cvl_cl_staging_buffer_init(&buf_sources_pos, CVL_CL_PRECISION_FP64, cvl_cl_compute_unified_memory(&comp), N_SOURCES,
                               NULL);
    cvl_cl_staging_buffer_init(&buf_sources_val, CVL_CL_PRECISION_FP64, cvl_cl_compute_unified_memory(&comp), N_SOURCES,
                               NULL);
    cvl_cl_staging_buffer_init(&buf_results, CVL_CL_PRECISION_FP64, cvl_cl_compute_unified_memory(&comp), N_TARGETS,
                               NULL);

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
    CVL_CL_CHECK(cvl_cl_staging_buffer_reserve(&buf_targets, &ctx, &queue, N_TARGETS), cleanup);
    CVL_CL_CHECK(cvl_cl_staging_buffer_reserve(&buf_sources_pos, &ctx, &queue, N_SOURCES), cleanup);
    CVL_CL_CHECK(cvl_cl_staging_buffer_reserve(&buf_sources_val, &ctx, &queue, N_SOURCES), cleanup);
    CVL_CL_CHECK(cvl_cl_staging_buffer_reserve(&buf_results, &ctx, &queue, N_TARGETS), cleanup);

    /* Async upload with chained futures */
    {
        cvl_cl_future_t f[3];
        CVL_CL_CHECK(cvl_cl_staging_buffer_write_async(&buf_targets, &queue, targets, N_TARGETS, 0, &f[0]), cleanup);
        CVL_CL_CHECK(cvl_cl_staging_buffer_write_async(&buf_sources_pos, &queue, sources_pos, N_SOURCES, 0, &f[1]),
                     cleanup);
        CVL_CL_CHECK(cvl_cl_staging_buffer_write_async(&buf_sources_val, &queue, sources_val, N_SOURCES, 0, &f[2]),
                     cleanup);
        for (int i = 0; i < 3; ++i)
            CVL_CL_CHECK(cvl_cl_future_wait(&f[i]), cleanup);
    }

    /* Launch kernel */
    {
        cvl_cl_kernel_t *k = cvl_cl_compute_kernel(&comp, "direct_sum");
        TEST_ASSERT(k != NULL, "kernel 'direct_sum' not found");

        CVL_CL_CHECK(
            cvl_cl_kernel_set_args(k,
                                   (cvl_cl_karg_t[]){
                                       {.type = CVL_CL_KARG_BUFFER, .index = 0, .mem = buf_targets.device.mem},
                                       {.type = CVL_CL_KARG_BUFFER, .index = 1, .mem = buf_sources_pos.device.mem},
                                       {.type = CVL_CL_KARG_BUFFER, .index = 2, .mem = buf_sources_val.device.mem},
                                       {.type = CVL_CL_KARG_SCALAR_UINT, .index = 3, .scalar_uint = N_SOURCES},
                                       {.type = CVL_CL_KARG_SCALAR_UINT, .index = 4, .scalar_uint = N_TARGETS},
                                       {.type = CVL_CL_KARG_BUFFER, .index = 5, .mem = buf_results.device.mem},
                                       {},
                                   }),
            cleanup);

        const size_t global = N_TARGETS;
        CVL_CL_CHECK(cvl_cl_ndrange(&queue, k, 1, &global, NULL, NULL, 0, NULL, NULL), cleanup);
    }

    /* Sync download */
    CVL_CL_CHECK(cvl_cl_staging_buffer_read_and_wait(&buf_results, &queue, gpu_results, N_TARGETS, 0), cleanup);

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
