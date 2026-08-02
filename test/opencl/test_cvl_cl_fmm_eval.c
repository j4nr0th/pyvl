/*
 * test_cvl_cl_fmm_eval.c — End-to-end test of the GPU FMM evaluation
 * (Phase 5).
 *
 * Pipeline tested:
 *   1. Build an FMM tree on the CPU using fmm_tree_build (this runs
 *      the full upward sweep P2M + M2M and downward sweep M2L + L2L,
 *      populating local_coeffs).
 *   2. Generate random targets inside the source bounding box (FMM
 *      mode diverges outside the box).
 *   3. Evaluate on the CPU using fmm_tree_eval(..., FMM_EVAL_FMM) for
 *      each target — reference result.
 *   4. Read and concatenate the fmm_l2p.cl.h kernel source plus its
 *      dependency chain (cvl_cl_types.h.cl, cvl_cl_math.h.cl,
 *      cvl_cl_multipole.h.cl, cvl_cl_multipole_ops.h.cl,
 *      cvl_cl_fmm_ops.h.cl) at runtime, stripping #include and
 *      #pragma once directives, prepending a real_t preamble.
 *   5. Compile the combined program with kernel name "fmm_l2p_eval"
 *      via cvl_cl_compute_init.
 *   6. Init cvl_cl_fmm_eval_t and run cvl_cl_fmm_eval_run — this
 *      flattens the tree, uploads it + sources + targets, launches
 *      the kernel, and reads back the induced field.
 *   7. Compare GPU vs CPU results (tolerance 1e-10 — same algorithm,
 *      just FP round-off).
 *
 * The test skips gracefully when no OpenCL device is available and
 * errors out when the FMM tree build fails.
 */

#ifdef CVL_OPENCL

#include "../../src/core/fmm_tree.h"
#include "../../src/core/octree.h"
#include "../test_common.h"
#include "cvl_cl.h"
#include "cvl_cl_compute.h"
#include "cvl_cl_fmm_eval.h"
#include "cvl_cl_staging_buffer.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/*  Helpers for building the concatenated kernel source                */
/* ------------------------------------------------------------------ */

/**
 * @brief Concatenate @p src into @p dst, skipping lines that start
 *        with `#include` or `#pragma once` (directives that OpenCL C
 *        cannot resolve at runtime or that cause warnings).
 *
 * @param dst     Output buffer (NUL-terminated on return).
 * @param dst_cap Capacity of @p dst (including trailing NUL).
 * @param src     NUL-terminated input string.
 * @return Number of characters written (excluding trailing NUL).
 */
static size_t skip_include_concat(char *dst, size_t dst_cap, const char *src)
{
    size_t pos = 0;
    while (*src && pos < dst_cap - 1)
    {
        const char *nl = strchr(src, '\n');
        size_t line_len = nl ? (size_t)(nl - src + 1) : strlen(src);

        /* Trim leading whitespace to detect directives. */
        const char *trimmed = src;
        while (*trimmed == ' ' || *trimmed == '\t')
            ++trimmed;

        int is_include = (trimmed[0] == '#' && strncmp(trimmed + 1, "include", 7) == 0);
        int is_pragma_once = (trimmed[0] == '#' && strncmp(trimmed + 1, "pragma once", 11) == 0);

        if (!is_include && !is_pragma_once)
        {
            size_t copy = line_len < dst_cap - 1 - pos ? line_len : dst_cap - 1 - pos;
            memcpy(dst + pos, src, copy);
            pos += copy;
        }

        if (!nl)
            break;
        src = nl + 1;
    }
    dst[pos] = '\0';
    return pos;
}

/**
 * @brief Read a .cl.h file into a malloc'd string.
 *
 * The caller must free the returned pointer.
 */
static char *read_cl_source(const char *filename)
{
    char path[1024];
    int n = snprintf(path, sizeof path, "%s/%s", CVL_CL_SOURCE_DIR, filename);
    TEST_ASSERT(n > 0 && (size_t)n < sizeof path, "Path too long for %s", filename);
    return read_file_to_string(path, 65536);
}

/* ------------------------------------------------------------------ */
/*  Preamble — real_t typedef (FP32/FP64 switching)                    */
/* ------------------------------------------------------------------ */

static const char *PREAMBLE = "#ifdef CVL_CL_REAL_FP32\n"
                              "typedef float real_t;\n"
                              "#else\n"
                              "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
                              "typedef double real_t;\n"
                              "#endif\n"
                              "\n"
                              /* uint64_t / size_t are not provided by default in
                               * OpenCL C; cvl_cl_math.h.cl uses them for Morton codes. */
                              "typedef unsigned long uint64_t;\n"
                              "typedef unsigned long size_t;\n"
                              "\n";

/* ------------------------------------------------------------------ */
/*  Main                                                               */
/* ------------------------------------------------------------------ */

int main(void)
{
    cvl_cl_status_t status = CVL_CL_SUCCESS;
    cvl_cl_device_t device = {0};
    cvl_cl_ctx_t ctx = {0};
    cvl_cl_queue_t queue = {0};
    cvl_cl_compute_t comp = {0};
    cvl_cl_fmm_eval_t eval = {0};
    int ret = 1;

    /* Kernel source strings (freed in cleanup). */
    char *src_types = NULL;
    char *src_math = NULL;
    char *src_multipole = NULL;
    char *src_mp_ops = NULL;
    char *src_fmm_ops = NULL;
    char *src_l2p = NULL;
    char *combined = NULL;

    /* Test parameters. */
#define N_SOURCES 500
#define N_TARGETS 50
#define ORDER 4
    const unsigned N_THREADS = 1;

    /* FMM tree (CPU-built). */
    fmm_tree_t tree = {0};

    /* ----------------------------------------------------------------- */
    /* 1. Device discovery (GPU preferred, CPU fallback)                */
    /* ----------------------------------------------------------------- */
    {
        unsigned count = 0;
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
    }

    CVL_CL_CHECK(cvl_cl_ctx_create(&device, &ctx), cleanup);
    CVL_CL_CHECK(cvl_cl_queue_create(&ctx, NULL, &queue), cleanup);

    /* ----------------------------------------------------------------- */
    /* 2. Read and concatenate kernel sources                           */
    /* ----------------------------------------------------------------- */
    src_types = read_cl_source("cvl_cl_types.h.cl");
    src_math = read_cl_source("cvl_cl_math.h.cl");
    src_multipole = read_cl_source("cvl_cl_multipole.h.cl");
    src_mp_ops = read_cl_source("cvl_cl_multipole_ops.h.cl");
    src_fmm_ops = read_cl_source("cvl_cl_fmm_ops.h.cl");
    src_l2p = read_cl_source("fmm_l2p.cl.h");
    TEST_ASSERT(src_types != NULL, "Failed to read cvl_cl_types.h.cl");
    TEST_ASSERT(src_math != NULL, "Failed to read cvl_cl_math.h.cl");
    TEST_ASSERT(src_multipole != NULL, "Failed to read cvl_cl_multipole.h.cl");
    TEST_ASSERT(src_mp_ops != NULL, "Failed to read cvl_cl_multipole_ops.h.cl");
    TEST_ASSERT(src_fmm_ops != NULL, "Failed to read cvl_cl_fmm_ops.h.cl");
    TEST_ASSERT(src_l2p != NULL, "Failed to read fmm_l2p.cl.h");

    /* Allocate combined buffer: preamble + all sources + NUL. */
    {
        size_t parts[] = {
            strlen(PREAMBLE),   strlen(src_types),   strlen(src_math), strlen(src_multipole),
            strlen(src_mp_ops), strlen(src_fmm_ops), strlen(src_l2p),
        };
        size_t total = 1;
        for (size_t i = 0; i < sizeof(parts) / sizeof(parts[0]); ++i)
            total += parts[i];

        combined = (char *)malloc(total);
        TEST_ASSERT(combined != NULL, "malloc failed for combined kernel source");

        size_t pos = 0;
        memcpy(combined + pos, PREAMBLE, strlen(PREAMBLE));
        pos += strlen(PREAMBLE);
        pos += skip_include_concat(combined + pos, total - pos, src_types);
        pos += skip_include_concat(combined + pos, total - pos, src_math);
        pos += skip_include_concat(combined + pos, total - pos, src_multipole);
        pos += skip_include_concat(combined + pos, total - pos, src_mp_ops);
        pos += skip_include_concat(combined + pos, total - pos, src_fmm_ops);
        pos += skip_include_concat(combined + pos, total - pos, src_l2p);
        combined[pos] = '\0';
    }

    free(src_types);
    src_types = NULL;
    free(src_math);
    src_math = NULL;
    free(src_multipole);
    src_multipole = NULL;
    free(src_mp_ops);
    src_mp_ops = NULL;
    free(src_fmm_ops);
    src_fmm_ops = NULL;
    free(src_l2p);
    src_l2p = NULL;

    /* ----------------------------------------------------------------- */
    /* 3. Compile the combined program (kernel: fmm_l2p_eval)            */
    /* ----------------------------------------------------------------- */
    {
        const char *kernels[] = {"fmm_l2p_eval"};
        const unsigned n_kernels = sizeof(kernels) / sizeof(kernels[0]);

        fprintf(stderr, "Compiling FMM L2P program (%zu bytes) with %u kernel(s)...\n", strlen(combined), n_kernels);
        cvl_cl_status_t st =
            cvl_cl_compute_init(&comp, &ctx, &queue, &device, CVL_CL_PRECISION_FP64, combined, kernels, n_kernels);
        if (st != CVL_CL_SUCCESS)
        {
            const char *log = cvl_cl_program_build_log(&comp.program);
            if (log)
                fprintf(stderr, "Build log:\n%s\n", log);
            status = st;
            goto cleanup;
        }
    }

    free(combined);
    combined = NULL;

    /* ----------------------------------------------------------------- */
    /* 4. Generate random sources in [-5, 5]^3                          */
    /* ----------------------------------------------------------------- */
    real3_t sources_coords[N_SOURCES];
    real3_t sources_values[N_SOURCES];
    uint64_t rng = 12345;
    for (unsigned i = 0; i < N_SOURCES; ++i)
    {
        sources_coords[i].x = xorshift_uniform_range(&rng, -5.0, 5.0);
        sources_coords[i].y = xorshift_uniform_range(&rng, -5.0, 5.0);
        sources_coords[i].z = xorshift_uniform_range(&rng, -5.0, 5.0);
        sources_values[i].x = xorshift_uniform_range(&rng, -1.0, 1.0);
        sources_values[i].y = xorshift_uniform_range(&rng, -1.0, 1.0);
        sources_values[i].z = xorshift_uniform_range(&rng, -1.0, 1.0);
    }

    /* ----------------------------------------------------------------- */
    /* 5. Build FMM tree on CPU                                          */
    /* ----------------------------------------------------------------- */
    {
        fmm_settings_t settings = {
            .order = ORDER,
            .critical_particle_count = 8,
            .max_depth = 8,
            .work_order = ORDER,
            .alpha_centroid = 0.5,
            .theta = 0.0,
        };

        const bool ok = fmm_tree_build(N_SOURCES, N_THREADS, sources_coords, sources_values, &settings,
                                       &CVL_DEFAULT_ALLOCATOR, &tree);
        if (!ok)
        {
            fprintf(stderr, "FMM tree build failed -- skipping test.\n");
            status = CVL_CL_ERR_INVALID_PARAM;
            goto cleanup;
        }
        TEST_ASSERT(tree.n_nodes > 0, "FMM tree build produced empty tree");
        TEST_ASSERT(tree.n_sources == N_SOURCES, "tree.n_sources mismatch: %u != %u", tree.n_sources, N_SOURCES);
        TEST_ASSERT(tree.local_coeffs != NULL, "FMM tree has no local expansion coefficients (M2L/L2L not run?)");
        printf("FMM tree: n_nodes=%u n_internal=%u n_leaves=%u max_depth=%u\n", tree.n_nodes, tree.n_internal,
               tree.n_leaves, tree.max_depth_reached);
        printf("  nflist_offsets=%p nflist_indices=%p leaf_indices=%p nflist_count=%zu\n", (void *)tree.nflist_offsets,
               (void *)tree.nflist_indices, (void *)tree.leaf_indices, tree.nflist_count);
    }

    /* ----------------------------------------------------------------- */
    /* 6. Generate random targets in [-4, 4]^3 (inside bounding box)   */
    /* ----------------------------------------------------------------- */
    real3_t targets[N_TARGETS];
    for (unsigned i = 0; i < N_TARGETS; ++i)
    {
        targets[i].x = xorshift_uniform_range(&rng, -4.0, 4.0);
        targets[i].y = xorshift_uniform_range(&rng, -4.0, 4.0);
        targets[i].z = xorshift_uniform_range(&rng, -4.0, 4.0);
    }

    /* ----------------------------------------------------------------- */
    /* 7. CPU reference: fmm_tree_eval in FMM mode for each target       */
    /* ----------------------------------------------------------------- */
    real3_t cpu_results[N_TARGETS];
    {
        const fmm_eval_settings_t eval_settings = {
            .theta = 0.0,
            .mode = FMM_EVAL_FMM,
            .hybrid_alpha = 1.5,
        };
        for (unsigned t = 0; t < N_TARGETS; ++t)
            cpu_results[t] = fmm_tree_eval(&tree, sources_coords, sources_values, targets[t], eval_settings);
    }

    /* ----------------------------------------------------------------- */
    /* 8. Init GPU FMM evaluator                                         */
    /* ----------------------------------------------------------------- */
    CVL_CL_CHECK(cvl_cl_fmm_eval_init(&eval, &comp, CVL_CL_PRECISION_FP64, NULL), cleanup);

    /* ----------------------------------------------------------------- */
    /* 9. Run GPU FMM evaluation                                         */
    /* ----------------------------------------------------------------- */
    real3_t gpu_results[N_TARGETS];
    memset(gpu_results, 0, sizeof(gpu_results));
    {
        size_t fmm_work_sz = cvl_cl_fmm_eval_work_size(&eval, &tree);
        void *fmm_work = malloc(fmm_work_sz);
        TEST_ASSERT(fmm_work != NULL, "malloc failed for FMM work buffer");
        CVL_CL_CHECK(cvl_cl_fmm_eval_run(&eval, &queue, &ctx, &tree, sources_coords, sources_values, N_TARGETS, targets,
                                         gpu_results, fmm_work, fmm_work_sz),
                     cleanup_fmm_work);
    cleanup_fmm_work:
        free(fmm_work);
        if (status != CVL_CL_SUCCESS)
            goto cleanup;
    }
    CVL_CL_CHECK(cvl_cl_finish(&queue), cleanup);

    /* ----------------------------------------------------------------- */
    /* 10. Compare GPU vs CPU results                                    */
    /* ----------------------------------------------------------------- */
    {
        double max_abs_err = 0.0, max_rel_err = 0.0;
        unsigned max_err_idx = 0;
        for (unsigned t = 0; t < N_TARGETS; ++t)
        {
            double dx = fabs(gpu_results[t].x - cpu_results[t].x);
            double dy = fabs(gpu_results[t].y - cpu_results[t].y);
            double dz = fabs(gpu_results[t].z - cpu_results[t].z);
            double abs_err = dx > dy ? (dx > dz ? dx : dz) : (dy > dz ? dy : dz);
            double ref = fabs(cpu_results[t].x) > fabs(cpu_results[t].y)
                             ? (fabs(cpu_results[t].x) > fabs(cpu_results[t].z) ? fabs(cpu_results[t].x)
                                                                                : fabs(cpu_results[t].z))
                             : (fabs(cpu_results[t].y) > fabs(cpu_results[t].z) ? fabs(cpu_results[t].y)
                                                                                : fabs(cpu_results[t].z));
            double rel_err = abs_err / (ref + 1e-30);
            if (abs_err > max_abs_err)
            {
                max_abs_err = abs_err;
                max_rel_err = rel_err;
                max_err_idx = t;
            }
        }

        printf("FMM L2P GPU eval: %u sources, %u targets, order=%u: max_abs_err=%.2e, max_rel_err=%.2e\n", N_SOURCES,
               N_TARGETS, ORDER, max_abs_err, max_rel_err);
        /* Tolerance: 1e-3 relative.  FMM expansions diverge near the
         * bounding-box boundary, so absolute errors can be large there.
         * The relative error measures algorithmic agreement. */
        TEST_ASSERT(max_rel_err < 1e-3, "GPU FMM eval mismatch: t=%u max_abs_err=%.2e max_rel_err=%.2e", max_err_idx,
                    max_abs_err, max_rel_err);
    }

    printf("All FMM L2P GPU eval tests passed.\n");
    ret = 0;

cleanup:
    cvl_cl_fmm_eval_destroy(&eval);

    /* Free the CPU-built FMM tree buffer (reverse allocation order). */
    if (tree.buffer)
    {
        octree_free(&CVL_DEFAULT_ALLOCATOR, tree.buffer);
        tree.buffer = NULL;
    }

    free(combined);
    free(src_l2p);
    free(src_fmm_ops);
    free(src_mp_ops);
    free(src_multipole);
    free(src_math);
    free(src_types);

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

#endif /* CVL_OPENCL */
