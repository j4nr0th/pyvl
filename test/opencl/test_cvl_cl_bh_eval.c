/*
 * test_cvl_cl_bh_eval.c — End-to-end test of the BH flat-tree
 * GPU evaluation kernel.
 *
 * 1. Build a flat octree from random particles (CPU side)
 * 2. Upload tree + particle data to GPU
 * 3. Compile + launch bh_flat_eval kernel (order=0, theta=1e-15)
 * 4. Read results back and compare with CPU direct sum
 *
 * With order=0 and a tiny theta the opening-angle MAC never accepts
 * (half_size/dist ≫ 1e-15 for any node), so the kernel traverses
 * the full tree to every leaf and does direct particle summation,
 * which must match the host-side direct sum.
 */

#ifdef CVL_OPENCL

#include "../test_common.h"
#include "cvl_cl.h"
#include "cvl_cl_compute.h"
#include "cvl_cl_flat_tree.h"
#include "cvl_cl_staging_buffer.h"
#include "cvl_cl_test_common.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/*  Host-side Morton 3D (same algorithm as cvl_cl_math.h.cl)          */
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
/*  Comparison helper for qsort                                        */
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
    cl_context ctx = NULL;
    cl_command_queue queue = NULL;
    cvl_cl_compute_t comp = {0};
    cvl_cl_chain_t chain = {0};
    int ret = 1;

    /* Flat tree data */
    cvl_cl_flat_tree_t tree = {0};
    /* Work buffer for the flat tree build (allocated before build, freed in cleanup). */
    void *flat_work = NULL;

    /* GPU buffers for tree data (non-real3_t, managed with raw cvl_cl_buffer_t) */
    cvl_cl_buffer_t buf_nodes = {0};
    cvl_cl_buffer_t buf_order = {0};
    cvl_cl_buffer_t buf_depth = {0};
    cvl_cl_buffer_t buf_coeffs = {0};

    /* Staging buffers for sources and results */
    cvl_cl_staging_buffer_t buf_src_pos = {0};
    cvl_cl_staging_buffer_t buf_src_val = {0};
    cvl_cl_staging_buffer_t buf_results = {0};

    /* Test parameters */
    uint64_t rng = 12345;
#define N_SOURCES 100
#define N_TARGETS 10
#define MAX_DEPTH 6
#define CRIT 8

    /* ----------------------------------------------------------------- */
    /* 1. Device discovery                                               */
    /* ----------------------------------------------------------------- */
    {
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
    CVL_CL_CHECK(cvl_cl_queue_create(ctx, device.id, NULL, &queue), cleanup);
    cvl_cl_chain_init(&chain, queue);

    /* ----------------------------------------------------------------- */
    /* 2. Compute backend: compile kernel (BH_EVAL pack, embedded source) */
    /* ----------------------------------------------------------------- */
    CVL_CL_CHECK(
        cvl_cl_compute_init(&comp, ctx, queue, &device, CVL_CL_PRECISION_FP64, (const char *[]){"bh_flat_eval"}, 1),
        cleanup);

    /* ----------------------------------------------------------------- */
    /* 3. Generate test data, build flat tree, upload to GPU             */
    /* ----------------------------------------------------------------- */
    {
        /* ---- Generate source positions + values ---- */
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

        /* ---- Compute bounding box ---- */
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
        real3_t root_center = {(bbox_min.x + bbox_max.x) * 0.5, (bbox_min.y + bbox_max.y) * 0.5,
                               (bbox_min.z + bbox_max.z) * 0.5};
        real_t root_hs =
            (real_t)fmax(fmax(bbox_max.x - bbox_min.x, bbox_max.y - bbox_min.y), bbox_max.z - bbox_min.z) * 0.5 +
            (real_t)1e-12;

        /* ---- Compute + sort Morton codes ---- */
        morton_entry_t entries[N_SOURCES];
        for (unsigned i = 0; i < N_SOURCES; ++i)
            entries[i] = (morton_entry_t){.code = morton_3d(sources[i], root_center, root_hs), .idx = i};
        qsort(entries, N_SOURCES, sizeof(morton_entry_t), morton_cmp);

        uint64_t mcodes[N_SOURCES];
        unsigned sorted_indices[N_SOURCES];
        for (unsigned i = 0; i < N_SOURCES; ++i)
        {
            mcodes[i] = entries[i].code;
            sorted_indices[i] = entries[i].idx;
        }

        /* ---- Build flat tree ---- */
        cvl_cl_flat_tree_settings_t settings = {
            .max_depth = MAX_DEPTH,
            .critical_particle_count = CRIT,
            .order = 0,
        };

        /* Count nodes first to size the build work buffer. */
        unsigned depth_counts[CVL_CL_FLAT_TREE_MAX_DEPTH + 2];
        unsigned n_total_work = 0, max_depth_used = 0;
        status = cvl_cl_flat_tree_count(N_SOURCES, mcodes, &settings, depth_counts, &n_total_work, &max_depth_used);
        TEST_ASSERT(status == CVL_CL_SUCCESS, "flat_tree_count failed: %s", cvl_cl_status_str(status));

        const unsigned work_depth = MAX_DEPTH > CVL_CL_FLAT_TREE_MAX_DEPTH ? CVL_CL_FLAT_TREE_MAX_DEPTH : MAX_DEPTH;
        const size_t flat_work_sz = cvl_cl_flat_tree_work_size(n_total_work, N_SOURCES, work_depth);
        flat_work = malloc(flat_work_sz);
        TEST_ASSERT(flat_work != NULL, "malloc(%zu) for flat tree work buffer failed", flat_work_sz);

        status = cvl_cl_flat_tree_build(N_SOURCES, sources, sorted_indices, mcodes, &settings, &tree, flat_work,
                                        flat_work_sz);
        TEST_ASSERT(status == CVL_CL_SUCCESS, "flat_tree_build failed: %s", cvl_cl_status_str(status));

        printf("BH eval test: %u nodes, %u sources, %u targets\n", tree.n_nodes, N_SOURCES, N_TARGETS);

        /* ---- Init staging buffers (FP64: no unified-memory flag needed) ---- */
        cvl_cl_staging_buffer_init(&buf_src_pos, CVL_CL_PRECISION_FP64);
        cvl_cl_staging_buffer_init(&buf_src_val, CVL_CL_PRECISION_FP64);
        cvl_cl_staging_buffer_init(&buf_results, CVL_CL_PRECISION_FP64);

        CVL_CL_CHECK(cvl_cl_staging_buffer_reserve(&buf_src_pos, ctx, queue, N_SOURCES), cleanup);
        CVL_CL_CHECK(cvl_cl_staging_buffer_reserve(&buf_src_val, ctx, queue, N_SOURCES), cleanup);
        CVL_CL_CHECK(cvl_cl_staging_buffer_reserve(&buf_results, ctx, queue, N_TARGETS), cleanup);

        /* ---- Upload sources via staging buffers (through the chain) ---- */
        CVL_CL_CHECK(cvl_cl_staging_buffer_write_async(&buf_src_pos, &chain, sources, NULL, N_SOURCES, 0, NULL),
                     cleanup);
        CVL_CL_CHECK(cvl_cl_staging_buffer_write_async(&buf_src_val, &chain, values, NULL, N_SOURCES, 0, NULL),
                     cleanup);
        CVL_CL_CHECK(cvl_cl_chain_finish(&chain), cleanup);

        /* ---- Upload tree data via raw buffers ---- */
        size_t nodes_bytes = tree.n_nodes * sizeof(cvl_cl_flat_node_t);
        size_t order_bytes = N_SOURCES * sizeof(unsigned);
        size_t depth_bytes = (MAX_DEPTH + 2) * sizeof(unsigned);

        CVL_CL_CHECK(
            cvl_cl_buffer_create(
                ctx, &(cvl_cl_buffer_desc_t){.access = CVL_CL_BUF_READ_ONLY, .size_bytes = nodes_bytes}, &buf_nodes),
            cleanup);
        CVL_CL_CHECK(
            cvl_cl_buffer_create(
                ctx, &(cvl_cl_buffer_desc_t){.access = CVL_CL_BUF_READ_ONLY, .size_bytes = order_bytes}, &buf_order),
            cleanup);
        CVL_CL_CHECK(
            cvl_cl_buffer_create(
                ctx, &(cvl_cl_buffer_desc_t){.access = CVL_CL_BUF_READ_ONLY, .size_bytes = depth_bytes}, &buf_depth),
            cleanup);
        /* Dummy coeffs buffer (never accessed with order=0) */
        CVL_CL_CHECK(cvl_cl_buffer_create(ctx, &(cvl_cl_buffer_desc_t){.access = CVL_CL_BUF_READ_ONLY, .size_bytes = 1},
                                          &buf_coeffs),
                     cleanup);

        CVL_CL_CHECK(cvl_cl_write_buffer(queue, &buf_nodes, 0, nodes_bytes, tree.nodes, 0, NULL, NULL), cleanup);
        CVL_CL_CHECK(cvl_cl_write_buffer(queue, &buf_order, 0, order_bytes, tree.particle_order, 0, NULL, NULL),
                     cleanup);
        CVL_CL_CHECK(cvl_cl_write_buffer(queue, &buf_depth, 0, depth_bytes, tree.depth_offsets, 0, NULL, NULL),
                     cleanup);
    }

    /* ----------------------------------------------------------------- */
    /* 4. Launch kernel (raw cl_kernel from the compute registry)         */
    /* ----------------------------------------------------------------- */
    {
        cl_kernel k = cvl_cl_compute_kernel(&comp, CVL_CL_PACK_BH_EVAL, CVL_CL_BH_EVAL_FLAT_EVAL);
        TEST_ASSERT(k != NULL, "kernel 'bh_flat_eval' not found");

        const size_t global = N_TARGETS;
        CVL_CL_CHECK(
            cvl_cl_chain_ndrange(&chain, k, 1, &global, NULL,
                                 (cvl_cl_karg_t[]){
                                     {.type = CVL_CL_KARG_BUFFER, .index = 0, .mem = buf_nodes.mem},
                                     {.type = CVL_CL_KARG_BUFFER, .index = 1, .mem = buf_order.mem},
                                     {.type = CVL_CL_KARG_BUFFER, .index = 2, .mem = buf_depth.mem},
                                     {.type = CVL_CL_KARG_BUFFER, .index = 3, .mem = buf_src_pos.device.mem},
                                     {.type = CVL_CL_KARG_BUFFER, .index = 4, .mem = buf_src_val.device.mem},
                                     {.type = CVL_CL_KARG_SCALAR_UINT, .index = 5, .scalar_uint = N_TARGETS},
                                     {.type = CVL_CL_KARG_SCALAR_UINT, .index = 6, .scalar_uint = 0}, /* order = 0 */
                                     {.type = CVL_CL_KARG_SCALAR_DOUBLE,
                                      .index = 7,
                                      .scalar_double = 1e-15}, /* tiny theta forces full descent */
                                     {.type = CVL_CL_KARG_BUFFER, .index = 8, .mem = buf_coeffs.mem},
                                     {.type = CVL_CL_KARG_BUFFER, .index = 9, .mem = buf_results.device.mem},
                                     {},
                                 },
                                 0, NULL, NULL),
            cleanup);
    }

    /* ----------------------------------------------------------------- */
    /* 5. Read results back (through the chain)                          */
    /* ----------------------------------------------------------------- */
    real3_t gpu_results[N_TARGETS];
    memset(gpu_results, 0, sizeof(gpu_results));
    CVL_CL_CHECK(cvl_cl_staging_buffer_read_and_wait(&buf_results, &chain, gpu_results, NULL, N_TARGETS, 0), cleanup);

    /* ----------------------------------------------------------------- */
    /* 6. CPU reference direct sum + comparison                          */
    /* ----------------------------------------------------------------- */
    {
        /* Regenerate same data from the same RNG seed */
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

        /* Compare GPU vs CPU */
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

        printf("BH eval: max_abs_err=%.2e, max_rel_err=%.2e\n", max_abs_err, max_rel_err);
        TEST_ASSERT(max_abs_err < 1e-12 || max_rel_err < 1e-10,
                    "GPU BH eval mismatch: t=%u max_abs_err=%.2e max_rel_err=%.2e", max_err_idx, max_abs_err,
                    max_rel_err);
    }

    printf("All BH eval tests passed.\n");
    ret = 0;

cleanup:
    cvl_cl_staging_buffer_destroy(&buf_results);
    cvl_cl_staging_buffer_destroy(&buf_src_val);
    cvl_cl_staging_buffer_destroy(&buf_src_pos);
    cvl_cl_buffer_destroy(&buf_coeffs);
    cvl_cl_buffer_destroy(&buf_depth);
    cvl_cl_buffer_destroy(&buf_order);
    cvl_cl_buffer_destroy(&buf_nodes);
    cvl_cl_chain_destroy(&chain);
    cvl_cl_compute_destroy(&comp);
    cvl_cl_flat_tree_destroy(&tree);
    free(flat_work);
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
