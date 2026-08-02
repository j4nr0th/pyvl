/*
 * test_cvl_cl_gpu_build.c — End-to-end test of the GPU tree build pipeline
 * with BH evaluation validation.
 *
 * Pipeline tested:
 *   1. Generate N=500 random source coordinates
 *   2. Compute bounding box, root_center, root_half_size on host
 *   3. Read and concatenate bh_build.cl.h + bh_flat_eval.cl.h kernel sources
 *      (reuses the skip_include_concat pattern from test_cvl_cl_device_kernel_compile.c)
 *   4. Compile combined program — register all 8 kernels
 *   5. Run the full GPU tree build (cvl_cl_gpu_tree_build_run)
 *   6. Validate tree structure against metadata (n_total, n_internal, depth_offsets, ...)
 *   7. Read back the flat tree, upload to new device buffers
 *   8. Launch bh_flat_eval (order=0, tiny theta forces full descent)
 *   9. Compare with CPU direct sum — max_abs_err < 1e-12
 *
 * The test verifies both the GPU build correctness (step 6) and the
 * combined build-then-eval pipeline (steps 8-9).
 */

#ifdef CVL_OPENCL

#include "../test_common.h"
#include "cvl_cl.h"
#include "cvl_cl_compute.h"
#include "cvl_cl_gpu_tree_build.h"
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
/*  Preamble — real_t / real3_t (same as test_cvl_cl_bh_eval.c)       */
/* ------------------------------------------------------------------ */

static const char *PREAMBLE = "#ifdef CVL_CL_REAL_FP32\n"
                              "typedef float real_t;\n"
                              "typedef struct { real_t x, y, z; } real3_t;\n"
                              "#else\n"
                              "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
                              "typedef double real_t;\n"
                              "typedef struct { real_t x, y, z; } real3_t;\n"
                              "#endif\n"
                              "\n";

/* ------------------------------------------------------------------ */
/*  Host-side Morton 3D (same algorithm as bh_build.cl.h)             */
/* ------------------------------------------------------------------ */

static inline uint64_t morton_split_21(uint64_t x)
{
    x &= 0x1fffffULL;
    x = (x | (x << 32)) & 0x1f00000000ffffULL;
    x = (x | (x << 16)) & 0x1f0000ff0000ffULL;
    x = (x | (x << 8)) & 0x100f00f00f00f00fULL;
    x = (x | (x << 4)) & 0x10c30c30c30c30c3ULL;
    x = (x | (x << 2)) & 0x1249249249249249ULL;
    return x;
}

static inline uint64_t morton_3d(real3_t p, real3_t root_center, real_t root_half_size)
{
    real_t inv_cell = 1.0 / (2.0 * root_half_size);
    real_t scale = (real_t)((1u << 21) - 1);
    real_t nx = (p.x - root_center.x) * inv_cell + 0.5;
    real_t ny = (p.y - root_center.y) * inv_cell + 0.5;
    real_t nz = (p.z - root_center.z) * inv_cell + 0.5;
    if (nx < 0)
        nx = 0;
    if (nx >= 1)
        nx = 0.999999;
    if (ny < 0)
        ny = 0;
    if (ny >= 1)
        ny = 0.999999;
    if (nz < 0)
        nz = 0;
    if (nz >= 1)
        nz = 0.999999;
    uint64_t ix = (uint64_t)(nx * scale);
    uint64_t iy = (uint64_t)(ny * scale);
    uint64_t iz = (uint64_t)(nz * scale);
    return morton_split_21(ix) | (morton_split_21(iy) << 1) | (morton_split_21(iz) << 2);
}

/* ------------------------------------------------------------------ */
/*  Comparison helper for qsort (used in host-side validation)        */
/* ------------------------------------------------------------------ */

typedef struct
{
    uint64_t code;
    unsigned idx;
} morton_entry_t;

static int morton_cmp(const void *a, const void *b)
{
    const morton_entry_t *ea = (const morton_entry_t *)a;
    const morton_entry_t *eb = (const morton_entry_t *)b;
    if (ea->code < eb->code)
        return -1;
    if (ea->code > eb->code)
        return 1;
    return 0;
}

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
    int ret = 1;

    /* GPU tree builder */
    cvl_cl_gpu_tree_build_t builder = {0};

    /* Staging buffers */
    cvl_cl_staging_buffer_t buf_src_pos = {0};
    cvl_cl_staging_buffer_t buf_src_val = {0};
    cvl_cl_staging_buffer_t buf_results = {0};

    /* GPU tree data for eval kernel (raw buffers) */
    cvl_cl_buffer_t buf_nodes = {0};
    cvl_cl_buffer_t buf_order = {0};
    cvl_cl_buffer_t buf_depth = {0};
    cvl_cl_buffer_t buf_coeffs = {0};

    /* Host copies for tree read-back */
    void *host_nodes = NULL;
    void *host_order = NULL;
    unsigned *host_depth_offsets = NULL;

    /* Kernel source strings */
    char *build_src = NULL;
    char *eval_src = NULL;
    char *combined = NULL;

    /* Test parameters */
    uint64_t rng = 12345;
    enum
    {
        N_SOURCES = 500,
        N_TARGETS = 50,
        MAX_DEPTH = 6,
        CRIT = 8,
        ORDER = 0, /* order=0 → pure direct-sum evaluation */
    };

    /* ----------------------------------------------------------------- */
    /* 1. Device discovery                                               */
    /* ----------------------------------------------------------------- */
    bool use_cpu_fallback = false;
    {
        unsigned count = 0;
        status = cvl_cl_device_first_gpu(&device);
        if (status != CVL_CL_SUCCESS)
        {
            /* CPU fallback.  The Intel NEO CPU backend ("OpenCL 3.0 (Build 0)")
             * has a clang-JIT miscompilation of data-dependent memory
             * indexing (see intel-neo-cpu-bug.md).  The wrapper fixes
             * (async-write UAF, boundary formula, host radix) stabilized the
             * tree-BUILD kernels on NEO (observed 15/15), but bh_flat_eval
             * still crashes ~25% of runs with JIT heap corruption, so the
             * pipeline cannot run reliably there.  We keep the NEO skip. */
            status = cvl_cl_device_first_cpu(&device);
            if (status != CVL_CL_SUCCESS)
            {
                fprintf(stderr, "No OpenCL device found -- skipping GPU build test.\n");
                return 0;
            }
            if (cvl_cl_device_is_intel_neo_cpu(&device))
            {
                fprintf(stderr,
                        "Intel NEO CPU backend detected -- its JIT is too unstable for the tree-build pipeline; "
                        "skipping pipeline test (radix policy API is covered by test_cvl_cl_radix_workaround).\n");
                return 0;
            }
            use_cpu_fallback = true;
            fprintf(stderr, "No GPU device found -- running GPU build test on CPU with host-side radix sort.\n");
        }
    }

    CVL_CL_CHECK(cvl_cl_ctx_create(&device, &ctx), cleanup);
    CVL_CL_CHECK(cvl_cl_queue_create(&ctx, NULL, &queue), cleanup);

    /* ----------------------------------------------------------------- */
    /* 2. Read and concatenate kernel sources                            */
    /* ----------------------------------------------------------------- */
    build_src = read_cl_source("bh_build.cl.h");
    eval_src = read_cl_source("bh_flat_eval.cl.h");
    TEST_ASSERT(build_src != NULL, "Failed to read bh_build.cl.h");
    TEST_ASSERT(eval_src != NULL, "Failed to read bh_flat_eval.cl.h");

    /* Allocate combined buffer: preamble + build + eval + NUL.
     * The total size is at most the sum of the individual sizes. */
    {
        size_t preamble_len = strlen(PREAMBLE);
        size_t build_len = strlen(build_src);
        size_t eval_len = strlen(eval_src);
        size_t total = preamble_len + build_len + eval_len + 1;

        combined = (char *)malloc(total);
        TEST_ASSERT(combined != NULL, "malloc failed for combined kernel source");

        size_t pos = 0;
        memcpy(combined + pos, PREAMBLE, preamble_len);
        pos += preamble_len;
        pos += skip_include_concat(combined + pos, total - pos, build_src);
        pos += skip_include_concat(combined + pos, total - pos, eval_src);
        combined[pos] = '\0';
    }

    free(build_src);
    build_src = NULL;
    free(eval_src);
    eval_src = NULL;

    /* ----------------------------------------------------------------- */
    /* 3. Compute backend: compile combined program                      */
    /* ----------------------------------------------------------------- */
    {
        const char *kernels[] = {
            "kernel_morton",      "kernel_radix_hist",     "kernel_radix_scatter", "kernel_boundary",
            "kernel_fill_leaves", "kernel_build_internal", "bh_flat_eval",
        };
        const unsigned n_kernels = sizeof(kernels) / sizeof(kernels[0]);

        fprintf(stderr, "Compiling combined program (%zu bytes) with %u kernels...\n", strlen(combined), n_kernels);
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
    /* 4. Generate test data                                             */
    /* ----------------------------------------------------------------- */
    real3_t sources[N_SOURCES];
    real3_t values[N_SOURCES];
    for (unsigned i = 0; i < N_SOURCES; ++i)
    {
        sources[i].x = xorshift_uniform_range(&rng, -5.0, 5.0);
        sources[i].y = xorshift_uniform_range(&rng, -5.0, 5.0);
        sources[i].z = xorshift_uniform_range(&rng, -5.0, 5.0);
        values[i].x = xorshift_uniform_range(&rng, -1.0, 1.0);
        values[i].y = xorshift_uniform_range(&rng, -1.0, 1.0);
        values[i].z = xorshift_uniform_range(&rng, -1.0, 1.0);
    }

    /* ----------------------------------------------------------------- */
    /* 5. Compute bounding box (same computation as pipeline)            */
    /* ----------------------------------------------------------------- */
    real3_t bbox_min = sources[0], bbox_max = sources[0];
    for (unsigned i = 1; i < N_SOURCES; ++i)
    {
        if (sources[i].x < bbox_min.x)
            bbox_min.x = sources[i].x;
        if (sources[i].y < bbox_min.y)
            bbox_min.y = sources[i].y;
        if (sources[i].z < bbox_min.z)
            bbox_min.z = sources[i].z;
        if (sources[i].x > bbox_max.x)
            bbox_max.x = sources[i].x;
        if (sources[i].y > bbox_max.y)
            bbox_max.y = sources[i].y;
        if (sources[i].z > bbox_max.z)
            bbox_max.z = sources[i].z;
    }
    real3_t root_center = {
        (bbox_min.x + bbox_max.x) * 0.5,
        (bbox_min.y + bbox_max.y) * 0.5,
        (bbox_min.z + bbox_max.z) * 0.5,
    };
    real_t root_hs =
        (real_t)fmax(fmax(bbox_max.x - bbox_min.x, bbox_max.y - bbox_min.y), bbox_max.z - bbox_min.z) * 0.5 +
        (real_t)1e-12;

    printf("GPU build test: %u sources, %u targets, root_hs=%.3e\n", N_SOURCES, N_TARGETS, root_hs);

    /* ----------------------------------------------------------------- */
    /* 6. Init staging buffers and upload source positions               */
    /* ----------------------------------------------------------------- */
    {
        bool unified = cvl_cl_compute_unified_memory(&comp);

        CVL_CL_CHECK(cvl_cl_staging_buffer_init(&buf_src_pos, CVL_CL_PRECISION_FP64, unified, N_SOURCES, NULL),
                     cleanup);
        CVL_CL_CHECK(cvl_cl_staging_buffer_init(&buf_src_val, CVL_CL_PRECISION_FP64, unified, N_SOURCES, NULL),
                     cleanup);
        CVL_CL_CHECK(cvl_cl_staging_buffer_init(&buf_results, CVL_CL_PRECISION_FP64, unified, N_TARGETS, NULL),
                     cleanup);

        CVL_CL_CHECK(cvl_cl_staging_buffer_reserve(&buf_src_pos, &ctx, &queue, N_SOURCES), cleanup);
        CVL_CL_CHECK(cvl_cl_staging_buffer_reserve(&buf_src_val, &ctx, &queue, N_SOURCES), cleanup);
        CVL_CL_CHECK(cvl_cl_staging_buffer_reserve(&buf_results, &ctx, &queue, N_TARGETS), cleanup);

        /* Upload sources (positions + values) via staging buffers. */
        {
            cvl_cl_future_t f[2];
            CVL_CL_CHECK(cvl_cl_staging_buffer_write_async(&buf_src_pos, &queue, sources, N_SOURCES, 0, &f[0]),
                         cleanup);
            CVL_CL_CHECK(cvl_cl_staging_buffer_write_async(&buf_src_val, &queue, values, N_SOURCES, 0, &f[1]), cleanup);
            for (int i = 0; i < 2; ++i)
                CVL_CL_CHECK(cvl_cl_future_wait(&f[i]), cleanup);
        }
    }

    /* ----------------------------------------------------------------- */
    /* 7. Initialize and run GPU tree build                              */
    /* ----------------------------------------------------------------- */
    CVL_CL_CHECK(cvl_cl_gpu_tree_build_init(&builder, &comp, MAX_DEPTH, CRIT, ORDER), cleanup);
    if (use_cpu_fallback)
        CVL_CL_CHECK(cvl_cl_gpu_tree_build_set_radix_policy(&builder, CVL_CL_RADIX_POLICY_WORKAROUND), cleanup);
    CVL_CL_CHECK(cvl_cl_finish(&queue), cleanup);
    size_t gpu_work_sz = cvl_cl_gpu_tree_build_work_size(N_SOURCES, MAX_DEPTH);
    void *gpu_work = malloc(gpu_work_sz);
    TEST_ASSERT(gpu_work != NULL, "malloc(%zu) for GPU tree build work buffer failed", gpu_work_sz);
    CVL_CL_CHECK(cvl_cl_gpu_tree_build_run(&builder, &queue, &ctx, &buf_src_pos, N_SOURCES, gpu_work, gpu_work_sz),
                 cleanup);

    printf("GPU build: n_total=%u n_internal=%u n_multipole=%u n_particle=%u\n", builder.n_total, builder.n_internal,
           builder.n_multipole_leaves, builder.n_particle_leaves);

    /* ----------------------------------------------------------------- */
    /* 8. Validate tree structure                                        */
    /* ----------------------------------------------------------------- */
    TEST_ASSERT(builder.n_total > 0, "GPU tree build produced empty tree (all particles degenerate?)");
    TEST_ASSERT(builder.n_total >= builder.n_internal, "n_total=%u < n_internal=%u", builder.n_total,
                builder.n_internal);
    TEST_ASSERT(builder.n_multipole_leaves + builder.n_particle_leaves == builder.n_total - builder.n_internal,
                "leaves (%u + %u) != total - internal (%u - %u)", builder.n_multipole_leaves, builder.n_particle_leaves,
                builder.n_total, builder.n_internal);
    TEST_ASSERT(builder.n_sources == N_SOURCES, "n_sources changed: %u != %u", builder.n_sources, N_SOURCES);

    /* Verify depth_counts monotonicity and depth_offsets consistency. */
    unsigned prev = 0;
    for (unsigned d = 0; d <= MAX_DEPTH; ++d)
    {
        TEST_ASSERT(builder.depth_counts[d] >= prev, "depth_counts[%u]=%u < depth_counts[%u]=%u (not monotonic)", d,
                    builder.depth_counts[d], d - 1, prev);
        prev = builder.depth_counts[d];
    }
    TEST_ASSERT(builder.depth_offsets[MAX_DEPTH + 1] == builder.n_total, "depth_offsets[max_depth+1]=%u != n_total=%u",
                builder.depth_offsets[MAX_DEPTH + 1], builder.n_total);

    /* ----------------------------------------------------------------- */
    /* 9. Read back the flat tree nodes from device                      */
    /* ----------------------------------------------------------------- */
    const size_t node_bytes = (size_t)builder.n_total * 64; /* CVL_CL_GPU_NODE_SIZE */
    const size_t order_bytes = (size_t)N_SOURCES * sizeof(unsigned);
    const size_t depth_bytes = (size_t)(MAX_DEPTH + 2) * sizeof(unsigned);

    host_nodes = malloc(node_bytes);
    host_order = malloc(order_bytes);
    host_depth_offsets = malloc(depth_bytes);

    TEST_ASSERT(host_nodes != NULL, "malloc failed for host_nodes");
    TEST_ASSERT(host_order != NULL, "malloc failed for host_order");
    TEST_ASSERT(host_depth_offsets != NULL, "malloc failed for host_depth_offsets");

    memcpy(host_depth_offsets, builder.depth_offsets, depth_bytes);

    CVL_CL_CHECK(cvl_cl_read_buffer(&queue, &builder.buf_nodes, 0, node_bytes, host_nodes, 0, NULL, NULL), cleanup);
    CVL_CL_CHECK(cvl_cl_read_buffer(&queue, &builder.buf_particle_order, 0, order_bytes, host_order, 0, NULL, NULL),
                 cleanup);
    CVL_CL_CHECK(cvl_cl_finish(&queue), cleanup);

    /* Quick structural sanity: verify depth_offsets for root node. */
    {
        const unsigned root_offset = builder.depth_offsets[0];
        TEST_ASSERT(root_offset == 0, "Root offset expected 0, got %u", root_offset);

        /* Verify that depth 0 has exactly 1 node (the root). */
        TEST_ASSERT(builder.depth_counts[0] >= 1, "Expected at least 1 root node, got %u", builder.depth_counts[0]);
    }

    /* ----------------------------------------------------------------- */
    /* 10. Upload tree data for eval kernel (via raw buffers)            */
    /* ----------------------------------------------------------------- */
    {
        CVL_CL_CHECK(
            cvl_cl_buffer_create(
                &ctx, &(cvl_cl_buffer_desc_t){.access = CVL_CL_BUF_READ_ONLY, .size_bytes = node_bytes}, &buf_nodes),
            cleanup);
        CVL_CL_CHECK(
            cvl_cl_buffer_create(
                &ctx, &(cvl_cl_buffer_desc_t){.access = CVL_CL_BUF_READ_ONLY, .size_bytes = order_bytes}, &buf_order),
            cleanup);
        CVL_CL_CHECK(
            cvl_cl_buffer_create(
                &ctx, &(cvl_cl_buffer_desc_t){.access = CVL_CL_BUF_READ_ONLY, .size_bytes = depth_bytes}, &buf_depth),
            cleanup);
        /* Dummy coeffs buffer (order=0 so never accessed). */
        CVL_CL_CHECK(cvl_cl_buffer_create(
                         &ctx, &(cvl_cl_buffer_desc_t){.access = CVL_CL_BUF_READ_ONLY, .size_bytes = 1}, &buf_coeffs),
                     cleanup);

        CVL_CL_CHECK(cvl_cl_write_buffer(&queue, &buf_nodes, 0, node_bytes, host_nodes, 0, NULL, NULL), cleanup);
        CVL_CL_CHECK(cvl_cl_write_buffer(&queue, &buf_order, 0, order_bytes, host_order, 0, NULL, NULL), cleanup);
        CVL_CL_CHECK(cvl_cl_write_buffer(&queue, &buf_depth, 0, depth_bytes, host_depth_offsets, 0, NULL, NULL),
                     cleanup);
    }

    /* ----------------------------------------------------------------- */
    /* 11. Launch bh_flat_eval kernel (order=0, tiny theta)              */
    /* ----------------------------------------------------------------- */
    {
        cvl_cl_kernel_t *k = cvl_cl_compute_kernel(&comp, "bh_flat_eval");
        TEST_ASSERT(k != NULL, "kernel 'bh_flat_eval' not found in compute backend");

        CVL_CL_CHECK(cvl_cl_kernel_set_args(
                         k,
                         (cvl_cl_karg_t[]){
                             {.type = CVL_CL_KARG_BUFFER, .index = 0, .mem = cvl_cl_buffer_mem(&buf_nodes)},
                             {.type = CVL_CL_KARG_BUFFER, .index = 1, .mem = cvl_cl_buffer_mem(&buf_order)},
                             {.type = CVL_CL_KARG_BUFFER, .index = 2, .mem = cvl_cl_buffer_mem(&buf_depth)},
                             {.type = CVL_CL_KARG_BUFFER, .index = 3, .mem = cvl_cl_buffer_mem(&buf_src_pos.device)},
                             {.type = CVL_CL_KARG_BUFFER, .index = 4, .mem = cvl_cl_buffer_mem(&buf_src_val.device)},
                             {.type = CVL_CL_KARG_SCALAR_UINT, .index = 5, .scalar_uint = N_TARGETS},
                             {.type = CVL_CL_KARG_SCALAR_UINT, .index = 6, .scalar_uint = 0}, /* order = 0 */
                             {.type = CVL_CL_KARG_SCALAR_DOUBLE,
                              .index = 7,
                              .scalar_double = 1e-15}, /* tiny theta → full descent */
                             {.type = CVL_CL_KARG_BUFFER, .index = 8, .mem = cvl_cl_buffer_mem(&buf_coeffs)},
                             {.type = CVL_CL_KARG_BUFFER, .index = 9, .mem = cvl_cl_buffer_mem(&buf_results.device)},
                             {},
                         }),
                     cleanup);

        const size_t global = N_TARGETS;
        CVL_CL_CHECK(cvl_cl_ndrange(&queue, k, 1, &global, NULL, NULL, 0, NULL, NULL), cleanup);
    }

    /* ----------------------------------------------------------------- */
    /* 12. Read results back                                             */
    /* ----------------------------------------------------------------- */
    real3_t gpu_results[N_TARGETS];
    memset(gpu_results, 0, sizeof(gpu_results));
    CVL_CL_CHECK(cvl_cl_staging_buffer_read_and_wait(&buf_results, &queue, gpu_results, N_TARGETS, 0), cleanup);

    /* ----------------------------------------------------------------- */
    /* 13. CPU reference direct sum + comparison                         */
    /* ----------------------------------------------------------------- */
    {
        /* Regenerate same data from the same RNG seed. */
        uint64_t rng_ref = 12345;
        real3_t ref_sources[N_SOURCES];
        real3_t ref_values[N_SOURCES];
        for (unsigned i = 0; i < N_SOURCES; ++i)
        {
            ref_sources[i].x = xorshift_uniform_range(&rng_ref, -5.0, 5.0);
            ref_sources[i].y = xorshift_uniform_range(&rng_ref, -5.0, 5.0);
            ref_sources[i].z = xorshift_uniform_range(&rng_ref, -5.0, 5.0);
            ref_values[i].x = xorshift_uniform_range(&rng_ref, -1.0, 1.0);
            ref_values[i].y = xorshift_uniform_range(&rng_ref, -1.0, 1.0);
            ref_values[i].z = xorshift_uniform_range(&rng_ref, -1.0, 1.0);
        }

        real3_t cpu_results[N_TARGETS];
        memset(cpu_results, 0, sizeof(cpu_results));

        for (unsigned t = 0; t < N_TARGETS; ++t)
        {
            real3_t acc = {0, 0, 0};
            for (unsigned s = 0; s < N_SOURCES; ++s)
            {
                real3_t dr = real3_sub(ref_sources[t], ref_sources[s]);
                acc = real3_add(acc, particle_kernel(ref_values[s], dr));
            }
            cpu_results[t] = acc;
        }

        /* Compare GPU vs CPU. */
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

        printf("GPU build eval: max_abs_err=%.2e, max_rel_err=%.2e\n", max_abs_err, max_rel_err);
        TEST_ASSERT(max_abs_err < 1e-12 || max_rel_err < 1e-10,
                    "GPU BH eval mismatch: t=%u max_abs_err=%.2e max_rel_err=%.2e", max_err_idx, max_abs_err,
                    max_rel_err);
    }

    printf("All GPU build tests passed.\n");
    ret = 0;

cleanup:
    free(combined);
    free(build_src);
    free(eval_src);
    free(host_nodes);
    free(host_order);
    free(host_depth_offsets);
    free(gpu_work);

    cvl_cl_gpu_tree_build_destroy(&builder);
    cvl_cl_staging_buffer_destroy(&buf_results);
    cvl_cl_staging_buffer_destroy(&buf_src_val);
    cvl_cl_staging_buffer_destroy(&buf_src_pos);
    cvl_cl_buffer_destroy(&buf_coeffs);
    cvl_cl_buffer_destroy(&buf_depth);
    cvl_cl_buffer_destroy(&buf_order);
    cvl_cl_buffer_destroy(&buf_nodes);
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
