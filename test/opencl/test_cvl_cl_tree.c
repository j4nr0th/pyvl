/*
 * test_cvl_cl_tree.c — Persistent OpenCL tree handle with asynchronous
 * build/eval jobs.
 *
 * Tests:
 *   1. Sync build (cvl_cl_tree_build) from random sources
 *   2. Tree-code eval at arbitrary targets vs CPU Barnes-Hut (same settings)
 *   3. Direct-sum eval vs exact direct sum
 *   4. Async build (begin/finish) + eval (begin/finish) path
 *   5. Rebuild reuse: build again with different sources, verify eval changes
 *   6. Error paths: eval before build, build while in flight
 */

#ifdef CVL_OPENCL

#include "../test_common.h"
#include "cvl_cl.h"
#include "cvl_cl_compute.h"
#include "cvl_cl_test_common.h"
#include "cvl_cl_tree.h"

#include "../../src/core/barnes_hut_tree.h"
#include "../../src/core/octree.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/*  Main                                                               */
/* ------------------------------------------------------------------ */

int main(void)
{
    cvl_cl_status_t status = CVL_CL_SUCCESS;
    cvl_cl_device_t device = {0};
    cl_context ctx = NULL;
    cl_command_queue queue = NULL;
    cvl_cl_compute_t comp = {0};
    cvl_cl_tree_t tree = {0};
    barnes_hut_tree_t bh_tree = {0};
    int ret = 1;

#define N_SOURCES 500
#define N_TARGETS 100
#define MAX_DEPTH 4
#define CRIT 4
#define ORDER 4

    /* ----------------------------------------------------------------- */
    /* 1. Device discovery                                               */
    /* ----------------------------------------------------------------- */
    {
        status = cvl_cl_device_first_gpu(&device);
        if (status != CVL_CL_SUCCESS)
            status = cvl_cl_device_first_cpu(&device);
        if (status != CVL_CL_SUCCESS)
        {
            fprintf(stderr, "No OpenCL device found -- skipping tree test.\n");
            return 0;
        }
        if (cvl_cl_device_is_intel_neo_cpu(&device))
        {
            fprintf(stderr, "Intel NEO CPU backend detected -- skipping (JIT instability).\n");
            return 0;
        }
    }

    CVL_CL_CHECK(cvl_cl_ctx_create(&device, &ctx), cleanup);
    CVL_CL_CHECK(cvl_cl_queue_create(ctx, device.id, NULL, &queue), cleanup);

    /* ----------------------------------------------------------------- */
    /* 2. Compute backend: BH_COEFFS + BH_EVAL + DIRECT_SUM packs        */
    /* ----------------------------------------------------------------- */
    {
        const char *kernels[] = {"kernel_p2m_leaves", "kernel_build_internal_m2m", "bh_flat_eval", "direct_sum"};
        CVL_CL_CHECK(cvl_cl_compute_init(&comp, ctx, queue, &device, CVL_CL_PRECISION_FP64, kernels,
                                         (unsigned)(sizeof(kernels) / sizeof(kernels[0]))),
                     cleanup);
    }

    /* ----------------------------------------------------------------- */
    /* 3. Generate data                                                  */
    /* ----------------------------------------------------------------- */
    uint64_t rng = 12345;
    real3_t sources[N_SOURCES];
    real3_t values[N_SOURCES];
    real3_t targets[N_TARGETS];
    for (unsigned i = 0; i < N_SOURCES; ++i)
    {
        sources[i].x = xorshift_uniform_range(&rng, -5.0, 5.0);
        sources[i].y = xorshift_uniform_range(&rng, -5.0, 5.0);
        sources[i].z = xorshift_uniform_range(&rng, -5.0, 5.0);
        values[i].x = xorshift_uniform_range(&rng, -1.0, 1.0);
        values[i].y = xorshift_uniform_range(&rng, -1.0, 1.0);
        values[i].z = xorshift_uniform_range(&rng, -1.0, 1.0);
    }
    for (unsigned i = 0; i < N_TARGETS; ++i)
    {
        /* Exterior targets (spherical shell radius 8-12): the multipole
         * series converges there, so the tree-code comparison vs the CPU
         * BH tree is meaningful. */
        real_t phi = xorshift_uniform_range(&rng, 0.0, 2.0 * 3.14159265358979323846);
        real_t cth = xorshift_uniform_range(&rng, -1.0, 1.0);
        real_t r = xorshift_uniform_range(&rng, 8.0, 12.0);
        real_t sth = sqrt(1.0 - cth * cth);
        targets[i].x = r * sth * cos(phi);
        targets[i].y = r * sth * sin(phi);
        targets[i].z = r * cth;
    }

    /* ----------------------------------------------------------------- */
    /* 4. Init tree + build                                              */
    /* ----------------------------------------------------------------- */
    cvl_cl_flat_tree_settings_t settings = {
        .max_depth = MAX_DEPTH,
        .critical_particle_count = CRIT,
        .order = ORDER,
    };
    CVL_CL_CHECK(cvl_cl_tree_init(&tree, &comp, CVL_CL_PRECISION_FP64, &settings, ORDER), cleanup);

    size_t work_sz = cvl_cl_tree_build_work_size(N_SOURCES, MAX_DEPTH);
    void *work = malloc(work_sz);
    TEST_ASSERT(work != NULL, "malloc(%zu) failed", work_sz);

    CVL_CL_CHECK(cvl_cl_tree_build(&tree, N_SOURCES, sources, values, work, work_sz), cleanup);
    printf("Tree build: %u nodes (%u int, %u mp, %u ptcl) depth=%u\n", tree.n_nodes, tree.n_internal,
           tree.n_multipole_leaves, tree.n_particle_leaves, tree.max_depth_used);
    TEST_ASSERT(tree.built, "tree not marked built");
    TEST_ASSERT(tree.n_nodes == tree.n_internal + tree.n_multipole_leaves + tree.n_particle_leaves,
                "node count invariant broken");

    /* ----------------------------------------------------------------- */
    /* 5. Direct-sum eval vs exact (isolates tree structure + upload)    */
    /* ----------------------------------------------------------------- */
    real3_t gpu_direct[N_TARGETS];
    memset(gpu_direct, 0, sizeof(gpu_direct));
    CVL_CL_CHECK(cvl_cl_tree_eval(&tree, N_TARGETS, targets, CVL_CL_TREE_EVAL_DIRECT, 0.0, gpu_direct, NULL), cleanup);
    {
        double max_abs = 0.0;
        for (unsigned t = 0; t < N_TARGETS; ++t)
        {
            real3_t acc = {0, 0, 0};
            for (unsigned s = 0; s < N_SOURCES; ++s)
            {
                real3_t dr = real3_sub(targets[t], sources[s]);
                acc = real3_add(acc, particle_kernel(values[s], dr));
            }
            double dx = fabs(gpu_direct[t].x - acc.x);
            double dy = fabs(gpu_direct[t].y - acc.y);
            double dz = fabs(gpu_direct[t].z - acc.z);
            double abs_err = dx > dy ? (dx > dz ? dx : dz) : (dy > dz ? dy : dz);
            if (abs_err > max_abs)
                max_abs = abs_err;
        }
        printf("Direct-sum eval vs exact: max_abs=%.2e\n", max_abs);
        TEST_ASSERT(max_abs < 1e-12, "direct eval mismatch: max_abs=%.2e", max_abs);
    }

    /* ----------------------------------------------------------------- */
    /* 6. Tree-code eval vs CPU Barnes-Hut                               */
    /* ----------------------------------------------------------------- */
    {
        /* Verify the root multipole coefficients before the eval. */
        const size_t n_coeffs = (size_t)(ORDER + 1) * (ORDER + 2) * (ORDER + 3) * (ORDER + 4) / 24u;
        cvl_cl_flat_node_t root_node;
        memset(&root_node, 0, sizeof(root_node));
        CVL_CL_CHECK(
            cvl_cl_read_buffer(queue, &tree.buf_nodes, 0, sizeof(cvl_cl_flat_node_t), &root_node, 0, NULL, NULL),
            cleanup);
        real_t *root_coeffs = (real_t *)malloc(3u * n_coeffs * sizeof(real_t));
        TEST_ASSERT(root_coeffs != NULL, "malloc failed");
        CVL_CL_CHECK(
            cvl_cl_read_buffer(queue, &tree.buf_coeffs, 0, 3u * n_coeffs * sizeof(real_t), root_coeffs, 0, NULL, NULL),
            cleanup);
        CVL_CL_CHECK(cvl_cl_finish(queue), cleanup);

        const size_t scratch_sz = multipole_scratch_size(ORDER);
        const size_t buf_sz = 3u * n_coeffs > scratch_sz ? 3u * n_coeffs : scratch_sz;
        real_t *host_coeffs = (real_t *)malloc(buf_sz * sizeof(real_t));
        real_t *cur = (real_t *)malloc(scratch_sz * sizeof(real_t));
        real_t *nxt = (real_t *)malloc(scratch_sz * sizeof(real_t));
        memset(host_coeffs, 0, buf_sz * sizeof(real_t));
        multipole_t ref_mp;
        const bool ok = multipole_create(ORDER, (unsigned)buf_sz, host_coeffs, root_node.center, N_SOURCES, sources,
                                         values, cur, nxt, &ref_mp);
        TEST_ASSERT(ok, "multipole_create failed");
        double max_rel = 0.0;
        for (size_t i = 0; i < 3u * n_coeffs; ++i)
        {
            double rel = fabs(root_coeffs[i] - host_coeffs[i]) / (fabs(host_coeffs[i]) + 1e-30);
            if (rel > max_rel)
                max_rel = rel;
        }
        printf("Tree root coeffs vs host: max_rel=%.2e\n", max_rel);
        TEST_ASSERT(max_rel < 1e-8, "tree root coeffs mismatch: max_rel=%.2e", max_rel);
        free(nxt);
        free(cur);
        free(host_coeffs);
        free(root_coeffs);
    }

    real3_t gpu_tc[N_TARGETS];
    real3_t cpu_bh[N_TARGETS];
    memset(gpu_tc, 0, sizeof(gpu_tc));
    memset(cpu_bh, 0, sizeof(cpu_bh));
    CVL_CL_CHECK(cvl_cl_tree_eval(&tree, N_TARGETS, targets, CVL_CL_TREE_EVAL_TREE_CODE, 0.0, gpu_tc, NULL), cleanup);

    {
        const barnes_hut_settings_t bh_settings = {
            .order = ORDER,
            .critical_particle_count = CRIT,
            .max_depth = MAX_DEPTH,
            .work_order = ORDER,
            .alpha_centroid = 0.0,
            .theta = 0.0,
        };
        TEST_ASSERT(
            barnes_hut_tree_build(N_SOURCES, 1, sources, values, &bh_settings, &CVL_DEFAULT_ALLOCATOR, &bh_tree),
            "barnes_hut_tree_build failed");
        const barnes_hut_eval_settings_t eval_settings = {.theta = 0.0};
        barnes_hut_tree_eval_all(&bh_tree, sources, values, N_TARGETS, targets, cpu_bh, eval_settings, 1);

        double max_rel = 0.0;
        for (unsigned t = 0; t < N_TARGETS; ++t)
        {
            double dx = fabs(gpu_tc[t].x - cpu_bh[t].x);
            double dy = fabs(gpu_tc[t].y - cpu_bh[t].y);
            double dz = fabs(gpu_tc[t].z - cpu_bh[t].z);
            double abs_err = dx > dy ? (dx > dz ? dx : dz) : (dy > dz ? dy : dz);
            double ref = fabs(cpu_bh[t].x) > fabs(cpu_bh[t].y)
                             ? (fabs(cpu_bh[t].x) > fabs(cpu_bh[t].z) ? fabs(cpu_bh[t].x) : fabs(cpu_bh[t].z))
                             : (fabs(cpu_bh[t].y) > fabs(cpu_bh[t].z) ? fabs(cpu_bh[t].y) : fabs(cpu_bh[t].z));
            double rel = abs_err / (ref + 1e-30);
            if (rel > max_rel)
                max_rel = rel;
        }
        printf("Tree-code eval vs CPU BH: max_rel=%.2e\n", max_rel);

        /* Both are order-4 multipole approximations with different cell
         * structures (uniform flat vs adaptive), so their truncation errors
         * differ in detail.  Assert the GPU is no worse than 10x the CPU
         * error vs the exact direct sum (the coefficient pipeline is correct
         * if the GPU is not dramatically worse). */
        {
            double gpu_worst = 0.0, cpu_worst = 0.0;
            for (unsigned t = 0; t < N_TARGETS; ++t)
            {
                real3_t exact = {0, 0, 0};
                for (unsigned s = 0; s < N_SOURCES; ++s)
                {
                    real3_t dr = real3_sub(targets[t], sources[s]);
                    exact = real3_add(exact, particle_kernel(values[s], dr));
                }
                for (int c = 0; c < 3; ++c)
                {
                    double gx = c == 0 ? gpu_tc[t].x : (c == 1 ? gpu_tc[t].y : gpu_tc[t].z);
                    double cx = c == 0 ? cpu_bh[t].x : (c == 1 ? cpu_bh[t].y : cpu_bh[t].z);
                    double ex = c == 0 ? exact.x : (c == 1 ? exact.y : exact.z);
                    double eg = fabs(gx - ex) / (fabs(ex) + 1e-30);
                    double ec = fabs(cx - ex) / (fabs(ex) + 1e-30);
                    if (eg > gpu_worst)
                        gpu_worst = eg;
                    if (ec > cpu_worst)
                        cpu_worst = ec;
                }
            }
            printf("  vs exact: GPU worst rel=%.2e, CPU BH worst rel=%.2e\n", gpu_worst, cpu_worst);
            TEST_ASSERT(gpu_worst < 10.0 * cpu_worst + 1e-6, "GPU tree-code much worse than CPU BH: gpu=%.2e cpu=%.2e",
                        gpu_worst, cpu_worst);
        }
    }

    /* ----------------------------------------------------------------- */
    /* 7. Async build + eval path                                        */
    /* ----------------------------------------------------------------- */
    {
        cvl_cl_tree_build_job_t bj;
        cvl_cl_tree_eval_job_t ej;

        /* Build in flight -> second build_begin must fail. */
        CVL_CL_CHECK(cvl_cl_tree_build_begin(&tree, N_SOURCES, sources, values, &bj), cleanup);
        cvl_cl_tree_build_job_t bj2;
        TEST_ASSERT(cvl_cl_tree_build_begin(&tree, N_SOURCES, sources, values, &bj2) == CVL_CL_ERR_INVALID_PARAM,
                    "second build_begin should fail while in flight");

        CVL_CL_CHECK(cvl_cl_tree_build_finish(&bj, work, work_sz), cleanup);

        CVL_CL_CHECK(cvl_cl_tree_eval_begin(&tree, N_TARGETS, targets, CVL_CL_TREE_EVAL_DIRECT, 0.0, &ej), cleanup);
        real3_t async_out[N_TARGETS];
        memset(async_out, 0, sizeof(async_out));
        CVL_CL_CHECK(cvl_cl_tree_eval_finish(&ej, async_out, NULL), cleanup);
        double amax = 0.0;
        for (unsigned t = 0; t < N_TARGETS; ++t)
        {
            double dx = fabs(async_out[t].x - gpu_direct[t].x);
            double dy = fabs(async_out[t].y - gpu_direct[t].y);
            double dz = fabs(async_out[t].z - gpu_direct[t].z);
            double abs_err = dx > dy ? (dx > dz ? dx : dz) : (dy > dz ? dy : dz);
            if (abs_err > amax)
                amax = abs_err;
        }
        printf("Async direct eval vs sync: max_abs=%.2e\n", amax);
        TEST_ASSERT(amax < 1e-12, "async eval mismatch: max_abs=%.2e", amax);
    }

    /* ----------------------------------------------------------------- */
    /* 8. Rebuild with different data                                    */
    /* ----------------------------------------------------------------- */
    {
        uint64_t rng2 = 999;
        real3_t sources2[N_SOURCES];
        real3_t values2[N_SOURCES];
        for (unsigned i = 0; i < N_SOURCES; ++i)
        {
            sources2[i].x = xorshift_uniform_range(&rng2, -3.0, 3.0);
            sources2[i].y = xorshift_uniform_range(&rng2, -3.0, 3.0);
            sources2[i].z = xorshift_uniform_range(&rng2, -3.0, 3.0);
            values2[i].x = xorshift_uniform_range(&rng2, -1.0, 1.0);
            values2[i].y = xorshift_uniform_range(&rng2, -1.0, 1.0);
            values2[i].z = xorshift_uniform_range(&rng2, -1.0, 1.0);
        }
        CVL_CL_CHECK(cvl_cl_tree_build(&tree, N_SOURCES, sources2, values2, work, work_sz), cleanup);

        real3_t out2[N_TARGETS];
        memset(out2, 0, sizeof(out2));
        CVL_CL_CHECK(cvl_cl_tree_eval(&tree, N_TARGETS, targets, CVL_CL_TREE_EVAL_DIRECT, 0.0, out2, NULL), cleanup);

        /* Must differ from the first tree's result. */
        double diff = 0.0;
        for (unsigned t = 0; t < N_TARGETS; ++t)
        {
            double dx = fabs(out2[t].x - gpu_direct[t].x);
            double dy = fabs(out2[t].y - gpu_direct[t].y);
            double dz = fabs(out2[t].z - gpu_direct[t].z);
            double abs_err = dx > dy ? (dx > dz ? dx : dz) : (dy > dz ? dy : dz);
            if (abs_err > diff)
                diff = abs_err;
        }
        printf("Rebuild changed result: max_abs=%.2e\n", diff);
        TEST_ASSERT(diff > 1e-3, "rebuild produced identical results (buffers not reused?)");
    }

    /* ----------------------------------------------------------------- */
    /* 9. Error paths                                                    */
    /* ----------------------------------------------------------------- */
    {
        cvl_cl_tree_t empty = {0};
        cvl_cl_flat_tree_settings_t es = {.max_depth = 4, .critical_particle_count = 4, .order = 4};
        CVL_CL_CHECK(cvl_cl_tree_init(&empty, &comp, CVL_CL_PRECISION_FP64, &es, 4), cleanup);
        real3_t one = {0, 0, 0};
        TEST_ASSERT(cvl_cl_tree_eval(&empty, 1, &one, CVL_CL_TREE_EVAL_DIRECT, 0.0, &one, NULL) ==
                        CVL_CL_ERR_INVALID_PARAM,
                    "eval before build should fail");
        cvl_cl_tree_destroy(&empty);
    }

    printf("All tree tests passed.\n");
    ret = 0;

cleanup:
    if (bh_tree.buffer)
    {
        octree_free(&CVL_DEFAULT_ALLOCATOR, bh_tree.buffer);
        bh_tree.buffer = NULL;
    }
    cvl_cl_tree_destroy(&tree);
    cvl_cl_compute_destroy(&comp);
    cvl_cl_queue_destroy(&queue);
    cvl_cl_ctx_destroy(&ctx);
    return ret;
}

#else

#include <stdio.h>
int main(void)
{
    printf("OpenCL not available -- skipping tree test.\n");
    return 0;
}

#endif
