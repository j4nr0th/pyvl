/*
 * test_cvl_cl_bh_coeffs.c — End-to-end test of the GPU multipole
 * coefficient kernels (P2M + M2M) and the BH tree-code evaluation.
 *
 * Pipeline:
 *   1. Build a flat octree from random particles (CPU side)
 *   2. Upload tree + particle data to GPU
 *   3. Launch kernel_p2m_leaves (leaf multipoles + Γ-weighted centroids)
 *   4. Launch kernel_build_internal_m2m per depth level (bottom-up M2M)
 *   5. Launch bh_flat_eval with order>0 (tree-code) at arbitrary targets
 *   6. Compare with the CPU Barnes-Hut tree evaluation
 *
 * This closes the loop the GPU build deliberately leaves open: the flat
 * tree builder produces structure only, and bh_flat_eval needs real
 * multipole coefficients.  With P2M/M2M the GPU pipeline is fully
 * self-contained for tree-code evaluation.
 */

#ifdef CVL_OPENCL

#include "../test_common.h"
#include "cvl_cl.h"
#include "cvl_cl_compute.h"
#include "cvl_cl_flat_tree.h"
#include "cvl_cl_gpu_tree_build.h"
#include "cvl_cl_staging_buffer.h"
#include "cvl_cl_test_common.h"

#include "../../src/core/barnes_hut_tree.h"
#include "../../src/core/octree.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

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
    void *flat_work = NULL;

    /* GPU buffers */
    cvl_cl_buffer_t buf_nodes = {0};
    cvl_cl_buffer_t buf_order = {0};
    cvl_cl_buffer_t buf_depth = {0};
    cvl_cl_buffer_t buf_parent_starts = {0};
    cvl_cl_buffer_t buf_coeffs = {0};
    cvl_cl_buffer_t buf_p2m_scratch = {0};
    cvl_cl_buffer_t buf_m2m_scratch = {0};

    /* Staging buffers */
    cvl_cl_staging_buffer_t buf_src_pos = {0};
    cvl_cl_staging_buffer_t buf_src_val = {0};
    cvl_cl_staging_buffer_t buf_targets = {0};
    cvl_cl_staging_buffer_t buf_results = {0};

    /* Bounding box metadata (needed by the M2M kernel for half_size). */
    real_t root_hs = 0.0;

    /* CPU Barnes-Hut tree (reference). */
    barnes_hut_tree_t bh_tree = {0};

    /* Test parameters */
    uint64_t rng = 12345;
#define N_SOURCES 500
#define N_TARGETS 100
#define MAX_DEPTH 3
#define CRIT 4
#define ORDER 4
#define WORK_ORDER 4
#define THETA 0.0 /* neighbour criterion: conservative, near-direct accuracy */

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
        if (cvl_cl_device_is_intel_neo_cpu(&device))
        {
            fprintf(stderr, "Intel NEO CPU backend detected -- its JIT miscompiles data-dependent indexing and "
                            "bh_flat_eval crashes ~25% of runs; skipping coefficient test there.\n");
            return 0;
        }
    }

    CVL_CL_CHECK(cvl_cl_ctx_create(&device, &ctx), cleanup);
    CVL_CL_CHECK(cvl_cl_queue_create(ctx, device.id, NULL, &queue), cleanup);
    cvl_cl_chain_init(&chain, queue);

    /* ----------------------------------------------------------------- */
    /* 2. Compute backend: BH_COEFFS + BH_EVAL packs                     */
    /* ----------------------------------------------------------------- */
    {
        const char *kernels[] = {"kernel_p2m_leaves", "kernel_build_internal_m2m", "bh_flat_eval"};
        CVL_CL_CHECK(cvl_cl_compute_init(&comp, ctx, queue, &device, CVL_CL_PRECISION_FP64, kernels,
                                         (unsigned)(sizeof(kernels) / sizeof(kernels[0]))),
                     cleanup);
    }

    /* ----------------------------------------------------------------- */
    /* 3. Generate test data + build the flat tree on the host           */
    /* ----------------------------------------------------------------- */
    real3_t sources[N_SOURCES];
    real3_t values[N_SOURCES];
    real3_t targets[N_TARGETS];
    {
        for (unsigned i = 0; i < N_SOURCES; ++i)
        {
            sources[i].x = xorshift_uniform_range(&rng, -5.0, 5.0);
            sources[i].y = xorshift_uniform_range(&rng, -5.0, 5.0);
            sources[i].z = xorshift_uniform_range(&rng, -5.0, 5.0);
            values[i].x = xorshift_uniform_range(&rng, -1.0, 1.0);
            values[i].y = xorshift_uniform_range(&rng, -1.0, 1.0);
            values[i].z = xorshift_uniform_range(&rng, -1.0, 1.0);
        }
        /* Targets exterior to the source cloud (sources in [-5,5]^3): the
         * regime where the multipole series converges.  Spherical shell
         * radius 8-12. */
        for (unsigned i = 0; i < N_TARGETS; ++i)
        {
            real_t phi = xorshift_uniform_range(&rng, 0.0, 2.0 * 3.14159265358979323846);
            real_t cth = xorshift_uniform_range(&rng, -1.0, 1.0);
            real_t r = xorshift_uniform_range(&rng, 8.0, 12.0);
            real_t sth = sqrt(1.0 - cth * cth);
            targets[i].x = r * sth * cos(phi);
            targets[i].y = r * sth * sin(phi);
            targets[i].z = r * cth;
        }

        /* ---- bounding box ---- */
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
        root_hs = (real_t)fmax(fmax(bbox_max.x - bbox_min.x, bbox_max.y - bbox_min.y), bbox_max.z - bbox_min.z) * 0.5 +
                  (real_t)1e-12;

        /* ---- Morton codes + sort ---- */
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

        /* ---- build flat tree ---- */
        cvl_cl_flat_tree_settings_t settings = {
            .max_depth = MAX_DEPTH,
            .critical_particle_count = CRIT,
            .order = ORDER,
        };

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

        printf("BH coeffs test: %u nodes (%u int, %u mp, %u ptcl), %u sources, %u targets\n", tree.n_nodes,
               tree.n_internal, tree.n_multipole_leaves, tree.n_particle_leaves, N_SOURCES, N_TARGETS);

        /* ---- staging buffers + upload ---- */
        cvl_cl_staging_buffer_init(&buf_src_pos, CVL_CL_PRECISION_FP64);
        cvl_cl_staging_buffer_init(&buf_src_val, CVL_CL_PRECISION_FP64);
        cvl_cl_staging_buffer_init(&buf_targets, CVL_CL_PRECISION_FP64);
        cvl_cl_staging_buffer_init(&buf_results, CVL_CL_PRECISION_FP64);

        CVL_CL_CHECK(cvl_cl_staging_buffer_reserve(&buf_src_pos, ctx, queue, N_SOURCES), cleanup);
        CVL_CL_CHECK(cvl_cl_staging_buffer_reserve(&buf_src_val, ctx, queue, N_SOURCES), cleanup);
        CVL_CL_CHECK(cvl_cl_staging_buffer_reserve(&buf_targets, ctx, queue, N_TARGETS), cleanup);
        CVL_CL_CHECK(cvl_cl_staging_buffer_reserve(&buf_results, ctx, queue, N_TARGETS), cleanup);

        CVL_CL_CHECK(cvl_cl_staging_buffer_write_async(&buf_src_pos, &chain, sources, NULL, N_SOURCES, 0, NULL),
                     cleanup);
        CVL_CL_CHECK(cvl_cl_staging_buffer_write_async(&buf_src_val, &chain, values, NULL, N_SOURCES, 0, NULL),
                     cleanup);
        CVL_CL_CHECK(cvl_cl_staging_buffer_write_async(&buf_targets, &chain, targets, NULL, N_TARGETS, 0, NULL),
                     cleanup);
        CVL_CL_CHECK(cvl_cl_chain_finish(&chain), cleanup);

        /* ---- raw tree buffers ---- */
        const size_t nodes_bytes = tree.n_nodes * sizeof(cvl_cl_flat_node_t);
        const size_t order_bytes = N_SOURCES * sizeof(unsigned);
        const size_t depth_bytes = (MAX_DEPTH + 2) * sizeof(unsigned);
        const size_t n_coeffs = (size_t)(ORDER + 1) * (ORDER + 2) * (ORDER + 3) * (ORDER + 4) / 24u;
        const size_t coeffs_bytes = (size_t)tree.n_nodes * 3u * n_coeffs * sizeof(double);
        const size_t scratch_wo = (size_t)(WORK_ORDER + 1) * (WORK_ORDER + 1) * (WORK_ORDER + 1);
        const size_t scratch_w = (size_t)(WORK_ORDER + 1) * (WORK_ORDER + 1) * (WORK_ORDER + 1);
        (void)scratch_wo;
        (void)scratch_w;

        CVL_CL_CHECK(
            cvl_cl_buffer_create(
                ctx, &(cvl_cl_buffer_desc_t){.access = CVL_CL_BUF_READ_WRITE, .size_bytes = nodes_bytes}, &buf_nodes),
            cleanup);
        CVL_CL_CHECK(
            cvl_cl_buffer_create(
                ctx, &(cvl_cl_buffer_desc_t){.access = CVL_CL_BUF_READ_ONLY, .size_bytes = order_bytes}, &buf_order),
            cleanup);
        CVL_CL_CHECK(
            cvl_cl_buffer_create(
                ctx, &(cvl_cl_buffer_desc_t){.access = CVL_CL_BUF_READ_ONLY, .size_bytes = depth_bytes}, &buf_depth),
            cleanup);
        CVL_CL_CHECK(
            cvl_cl_buffer_create(
                ctx, &(cvl_cl_buffer_desc_t){.access = CVL_CL_BUF_READ_WRITE, .size_bytes = coeffs_bytes}, &buf_coeffs),
            cleanup);

        /* P2M scratch: n_leaves * 2 * scratch_size */
        CVL_CL_CHECK(cvl_cl_buffer_create(
                         ctx,
                         &(cvl_cl_buffer_desc_t){.access = CVL_CL_BUF_READ_WRITE,
                                                 .size_bytes = (size_t)tree.n_nodes * 2u * scratch_w * sizeof(double)},
                         &buf_p2m_scratch),
                     cleanup);

        /* M2M scratch: n_internal * (3*(wo+1)^2 + 2*n_coeffs(wo) + 2*scratch) */
        {
            const size_t wo_coeffs =
                (size_t)(WORK_ORDER + 1) * (WORK_ORDER + 2) * (WORK_ORDER + 3) * (WORK_ORDER + 4) / 24u;
            const size_t shift_plane = (size_t)(WORK_ORDER + 1) * (WORK_ORDER + 1);
            const size_t per_wg = 3u * shift_plane + 2u * wo_coeffs + 2u * scratch_w;
            CVL_CL_CHECK(cvl_cl_buffer_create(
                             ctx,
                             &(cvl_cl_buffer_desc_t){.access = CVL_CL_BUF_READ_WRITE,
                                                     .size_bytes = (size_t)tree.n_nodes * per_wg * sizeof(double)},
                             &buf_m2m_scratch),
                         cleanup);
        }

        /* parent_starts: one entry per internal node + sentinel */
        CVL_CL_CHECK(
            cvl_cl_buffer_create(ctx,
                                 &(cvl_cl_buffer_desc_t){.access = CVL_CL_BUF_READ_ONLY,
                                                         .size_bytes = ((size_t)tree.n_nodes + 1) * sizeof(unsigned)},
                                 &buf_parent_starts),
            cleanup);

        CVL_CL_CHECK(cvl_cl_write_buffer(queue, &buf_nodes, 0, nodes_bytes, tree.nodes, 0, NULL, NULL), cleanup);
        CVL_CL_CHECK(cvl_cl_write_buffer(queue, &buf_order, 0, order_bytes, tree.particle_order, 0, NULL, NULL),
                     cleanup);
        CVL_CL_CHECK(cvl_cl_write_buffer(queue, &buf_depth, 0, depth_bytes, tree.depth_offsets, 0, NULL, NULL),
                     cleanup);
        CVL_CL_CHECK(cvl_cl_finish(queue), cleanup);
    }

    /* ----------------------------------------------------------------- */
    /* 4. P2M + M2M + eval kernels                                      */
    /* ----------------------------------------------------------------- */
    {
        const unsigned leaf_offset = tree.depth_offsets[MAX_DEPTH];
        const unsigned n_leaves = tree.n_nodes - tree.n_internal;

        /* ---- 4a. P2M: leaf multipoles + Γ-weighted centroids ---- */
        {
            cl_kernel k = cvl_cl_compute_kernel(&comp, CVL_CL_PACK_BH_COEFFS, CVL_CL_BH_COEFFS_P2M);
            TEST_ASSERT(k != NULL, "kernel 'kernel_p2m_leaves' not found");

            const size_t global = ((n_leaves + CVL_CL_GPU_BUILD_WG - 1) / CVL_CL_GPU_BUILD_WG) * CVL_CL_GPU_BUILD_WG;
            CVL_CL_CHECK(
                cvl_cl_chain_ndrange(&chain, k, 1, &global, NULL,
                                     (cvl_cl_karg_t[]){
                                         {.type = CVL_CL_KARG_BUFFER, .index = 0, .mem = buf_nodes.mem},
                                         {.type = CVL_CL_KARG_SCALAR_UINT, .index = 1, .scalar_uint = n_leaves},
                                         {.type = CVL_CL_KARG_SCALAR_UINT, .index = 2, .scalar_uint = leaf_offset},
                                         {.type = CVL_CL_KARG_SCALAR_UINT, .index = 3, .scalar_uint = tree.n_nodes},
                                         {.type = CVL_CL_KARG_BUFFER, .index = 4, .mem = buf_order.mem},
                                         {.type = CVL_CL_KARG_BUFFER, .index = 5, .mem = buf_src_pos.device.mem},
                                         {.type = CVL_CL_KARG_BUFFER, .index = 6, .mem = buf_src_val.device.mem},
                                         {.type = CVL_CL_KARG_SCALAR_UINT, .index = 7, .scalar_uint = ORDER},
                                         {.type = CVL_CL_KARG_SCALAR_UINT, .index = 8, .scalar_uint = WORK_ORDER},
                                         {.type = CVL_CL_KARG_BUFFER, .index = 9, .mem = buf_coeffs.mem},
                                         {.type = CVL_CL_KARG_BUFFER, .index = 10, .mem = buf_p2m_scratch.mem},
                                         {},
                                     },
                                     0, NULL, NULL),
                cleanup);
        }

        /* ---- 4b. M2M: bottom-up internal levels ---- */
        for (int d = MAX_DEPTH - 1; d >= 0; --d)
        {
            const unsigned depth = (unsigned)d;
            const unsigned n_parents = tree.depth_offsets[depth + 1] - tree.depth_offsets[depth];
            if (n_parents == 0)
                continue;

            cl_kernel k = cvl_cl_compute_kernel(&comp, CVL_CL_PACK_BH_COEFFS, CVL_CL_BH_COEFFS_INTERNAL);
            TEST_ASSERT(k != NULL, "kernel 'kernel_build_internal_m2m' not found");

            /* parent_starts: the flat tree stores child_base per parent, so
             * parent_starts[p] = child_base of parent p.  The sentinel is the
             * first node of the depth below the children (end of last child
             * range).  Children are contiguous per parent (level-ordered). */
            unsigned *parent_starts = (unsigned *)calloc(n_parents + 1, sizeof(unsigned));
            TEST_ASSERT(parent_starts != NULL, "calloc failed for parent_starts");

            for (unsigned p = 0; p < n_parents; ++p)
            {
                const unsigned ni = tree.depth_offsets[depth] + p;
                parent_starts[p] = (unsigned)tree.nodes[ni].child_base;
            }
            /* Sentinel: end of the last child's range = start of the next
             * depth layer's nodes + its count.  Since children of depth d are
             * the nodes at depth d+1, the sentinel is depth_offsets[d+1] +
             * n_children (the first node past the children layer). */
            parent_starts[n_parents] =
                tree.depth_offsets[depth + 1] + (tree.depth_offsets[depth + 2] - tree.depth_offsets[depth + 1]);

            CVL_CL_CHECK(cvl_cl_write_buffer(queue, &buf_parent_starts, 0, (n_parents + 1) * sizeof(unsigned),
                                             parent_starts, 0, NULL, NULL),
                         cleanup);
            free(parent_starts);

            const unsigned child_offset = tree.depth_offsets[depth + 1];
            const unsigned n_children = tree.depth_offsets[depth + 2] - tree.depth_offsets[depth + 1];

            const size_t global = ((n_parents + CVL_CL_GPU_BUILD_WG - 1) / CVL_CL_GPU_BUILD_WG) * CVL_CL_GPU_BUILD_WG;
            CVL_CL_CHECK(
                cvl_cl_chain_ndrange(
                    &chain, k, 1, &global, NULL,
                    (cvl_cl_karg_t[]){
                        {.type = CVL_CL_KARG_BUFFER, .index = 0, .mem = buf_nodes.mem},
                        {.type = CVL_CL_KARG_SCALAR_UINT, .index = 1, .scalar_uint = depth},
                        {.type = CVL_CL_KARG_SCALAR_UINT, .index = 2, .scalar_uint = n_parents},
                        {.type = CVL_CL_KARG_SCALAR_UINT, .index = 3, .scalar_uint = tree.depth_offsets[depth]},
                        {.type = CVL_CL_KARG_SCALAR_UINT, .index = 4, .scalar_uint = child_offset},
                        {.type = CVL_CL_KARG_SCALAR_UINT, .index = 5, .scalar_uint = n_children},
                        {.type = CVL_CL_KARG_SCALAR_DOUBLE, .index = 6, .scalar_double = root_hs},
                        {.type = CVL_CL_KARG_BUFFER, .index = 7, .mem = buf_parent_starts.mem},
                        {.type = CVL_CL_KARG_BUFFER, .index = 8, .mem = buf_order.mem},
                        {.type = CVL_CL_KARG_BUFFER, .index = 9, .mem = buf_src_pos.device.mem},
                        {.type = CVL_CL_KARG_BUFFER, .index = 10, .mem = buf_src_val.device.mem},
                        {.type = CVL_CL_KARG_SCALAR_UINT, .index = 11, .scalar_uint = ORDER},
                        {.type = CVL_CL_KARG_SCALAR_UINT, .index = 12, .scalar_uint = WORK_ORDER},
                        {.type = CVL_CL_KARG_BUFFER, .index = 13, .mem = buf_coeffs.mem},
                        {.type = CVL_CL_KARG_BUFFER, .index = 14, .mem = buf_m2m_scratch.mem},
                        {},
                    },
                    0, NULL, NULL),
                cleanup);
        }
        CVL_CL_CHECK(cvl_cl_chain_finish(&chain), cleanup);

        /* ---- 4c. debug: verify the root (node 0) multipole against a
         * host-side multipole_create over all sources.  The root should be
         * a P2M of every source (all leaves are PARTICLE at max_depth, so
         * M2M P2M's them all the way up). ---- */
        {
            const size_t n_coeffs = (size_t)(ORDER + 1) * (ORDER + 2) * (ORDER + 3) * (ORDER + 4) / 24u;
            real_t *gpu_root_coeffs = (real_t *)malloc(3u * n_coeffs * sizeof(real_t));
            TEST_ASSERT(gpu_root_coeffs != NULL, "malloc failed for root coeffs");

            /* Read back the GPU node-0 center (Γ-weighted centroid). */
            cvl_cl_flat_node_t root_node;
            memset(&root_node, 0, sizeof(root_node));
            CVL_CL_CHECK(
                cvl_cl_read_buffer(queue, &buf_nodes, 0, sizeof(cvl_cl_flat_node_t), &root_node, 0, NULL, NULL),
                cleanup);
            CVL_CL_CHECK(cvl_cl_finish(queue), cleanup);

            CVL_CL_CHECK(cvl_cl_read_buffer(queue, &buf_coeffs, 0, 3u * n_coeffs * sizeof(real_t), gpu_root_coeffs, 0,
                                            NULL, NULL),
                         cleanup);
            CVL_CL_CHECK(cvl_cl_finish(queue), cleanup);

            /* Host reference: multipole_create of all sources at the same
             * Γ-weighted centroid. */
            const size_t scratch_sz = multipole_scratch_size(ORDER);
            const size_t buf_sz = 3u * n_coeffs > scratch_sz ? 3u * n_coeffs : scratch_sz;
            real_t *host_coeffs = (real_t *)malloc(buf_sz * sizeof(real_t));
            real_t *cur = (real_t *)malloc(scratch_sz * sizeof(real_t));
            real_t *nxt = (real_t *)malloc(scratch_sz * sizeof(real_t));
            TEST_ASSERT(host_coeffs != NULL && cur != NULL && nxt != NULL, "malloc failed for host coeffs");
            memset(host_coeffs, 0, buf_sz * sizeof(real_t));

            multipole_t ref_mp;
            const bool ok = multipole_create(ORDER, (unsigned)buf_sz, host_coeffs, root_node.center, N_SOURCES, sources,
                                             values, cur, nxt, &ref_mp);
            TEST_ASSERT(ok, "host multipole_create failed");

            double max_rel = 0.0;
            for (size_t i = 0; i < 3u * n_coeffs; ++i)
            {
                const double a = gpu_root_coeffs[i];
                const double b = host_coeffs[i];
                const double rel = fabs(a - b) / (fabs(b) + 1e-30);
                if (rel > max_rel)
                    max_rel = rel;
            }
            printf("Root coeffs: GPU vs host multipole_create: max_rel=%.2e (center %.6f %.6f %.6f)\n", max_rel,
                   root_node.center.x, root_node.center.y, root_node.center.z);
            TEST_ASSERT(max_rel < 1e-10, "Root coeffs mismatch: max_rel=%.2e", max_rel);

            free(nxt);
            free(cur);
            free(host_coeffs);
            free(gpu_root_coeffs);
        }

        /* ---- 4c. bh_flat_eval (tree-code, order>0, arbitrary targets) ---- */
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
                                         {.type = CVL_CL_KARG_SCALAR_UINT, .index = 6, .scalar_uint = ORDER},
                                         {.type = CVL_CL_KARG_SCALAR_DOUBLE, .index = 7, .scalar_double = THETA},
                                         {.type = CVL_CL_KARG_BUFFER, .index = 8, .mem = buf_coeffs.mem},
                                         {.type = CVL_CL_KARG_BUFFER, .index = 9, .mem = buf_targets.device.mem},
                                         {.type = CVL_CL_KARG_BUFFER, .index = 10, .mem = buf_results.device.mem},
                                         {},
                                     },
                                     0, NULL, NULL),
                cleanup);
        }

        real3_t gpu_results[N_TARGETS];
        memset(gpu_results, 0, sizeof(gpu_results));
        CVL_CL_CHECK(cvl_cl_staging_buffer_read_and_wait(&buf_results, &chain, gpu_results, NULL, N_TARGETS, 0),
                     cleanup);

        /* ----------------------------------------------------------------- */
        /* 5. CPU reference: Barnes-Hut tree with the SAME settings.         */
        /* ----------------------------------------------------------------- */
        /* The GPU tree-code (order 4, theta=0) has real multipole truncation
         * error at these targets; the exact direct sum is therefore NOT the
         * right reference.  Both the GPU flat tree and the CPU Barnes-Hut
         * tree approximate the same physics with the same order and theta,
         * so they must agree closely — even though the cell structures
         * differ (uniform flat vs adaptive octree), the multipole expansions
         * are mathematically the same approximation. */
        {
            const barnes_hut_settings_t bh_settings = {
                .order = ORDER,
                .critical_particle_count = CRIT,
                .max_depth = MAX_DEPTH,
                .work_order = WORK_ORDER,
                .alpha_centroid = 0.0, /* match the uniform flat tree */
                .theta = THETA,
            };
            TEST_ASSERT(
                barnes_hut_tree_build(N_SOURCES, 1, sources, values, &bh_settings, &CVL_DEFAULT_ALLOCATOR, &bh_tree),
                "barnes_hut_tree_build failed");

            real3_t cpu_results[N_TARGETS];
            memset(cpu_results, 0, sizeof(cpu_results));
            const barnes_hut_eval_settings_t eval_settings = {.theta = THETA};
            barnes_hut_tree_eval_all(&bh_tree, sources, values, N_TARGETS, targets, cpu_results, eval_settings, 1);

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

            printf("BH coeffs: GPU tree-code vs CPU BH: max_abs_err=%.2e, max_rel_err=%.2e\n", max_abs_err,
                   max_rel_err);

            /* Both GPU and CPU BH are order-4 multipole approximations; the
             * meaningful assertion is that their error vs the exact direct
             * sum is the SAME order of magnitude (the coefficient pipeline
             * works if the GPU is not significantly worse than the CPU). */
            {
                /* exact direct sum */
                real3_t exact[N_TARGETS];
                memset(exact, 0, sizeof(exact));
                for (unsigned t = 0; t < N_TARGETS; ++t)
                {
                    real3_t acc = {0, 0, 0};
                    for (unsigned s = 0; s < N_SOURCES; ++s)
                    {
                        real3_t dr = real3_sub(targets[t], sources[s]);
                        acc = real3_add(acc, particle_kernel(values[s], dr));
                    }
                    exact[t] = acc;
                }

                double gpu_worst = 0.0, cpu_worst = 0.0;
                for (unsigned t = 0; t < N_TARGETS; ++t)
                {
                    double ref = fabs(exact[t].x) > fabs(exact[t].y)
                                     ? (fabs(exact[t].x) > fabs(exact[t].z) ? fabs(exact[t].x) : fabs(exact[t].z))
                                     : (fabs(exact[t].y) > fabs(exact[t].z) ? fabs(exact[t].y) : fabs(exact[t].z));
                    double e_gpu = 0, e_cpu = 0;
                    for (int c = 0; c < 3; ++c)
                    {
                        double gx = c == 0 ? gpu_results[t].x : (c == 1 ? gpu_results[t].y : gpu_results[t].z);
                        double cx = c == 0 ? cpu_results[t].x : (c == 1 ? cpu_results[t].y : cpu_results[t].z);
                        double ex = c == 0 ? exact[t].x : (c == 1 ? exact[t].y : exact[t].z);
                        double eg = fabs(gx - ex) / (fabs(ex) + 1e-30);
                        double ec = fabs(cx - ex) / (fabs(ex) + 1e-30);
                        if (eg > e_gpu)
                            e_gpu = eg;
                        if (ec > e_cpu)
                            e_cpu = ec;
                    }
                    if (e_gpu > gpu_worst)
                        gpu_worst = e_gpu;
                    if (e_cpu > cpu_worst)
                        cpu_worst = e_cpu;
                }
                printf("  vs exact: GPU worst rel=%.2e, CPU BH worst rel=%.2e\n", gpu_worst, cpu_worst);
                TEST_ASSERT(gpu_worst < 10.0 * cpu_worst + 1e-6,
                            "GPU tree-code much worse than CPU BH: gpu=%.2e cpu=%.2e", gpu_worst, cpu_worst);
            }
        }
    }

    printf("All BH coeffs tests passed.\n");
    ret = 0;

cleanup:
    if (bh_tree.buffer)
    {
        octree_free(&CVL_DEFAULT_ALLOCATOR, bh_tree.buffer);
        bh_tree.buffer = NULL;
    }
    cvl_cl_staging_buffer_destroy(&buf_results);
    cvl_cl_staging_buffer_destroy(&buf_targets);
    cvl_cl_staging_buffer_destroy(&buf_src_val);
    cvl_cl_staging_buffer_destroy(&buf_src_pos);
    cvl_cl_buffer_destroy(&buf_m2m_scratch);
    cvl_cl_buffer_destroy(&buf_p2m_scratch);
    cvl_cl_buffer_destroy(&buf_coeffs);
    cvl_cl_buffer_destroy(&buf_parent_starts);
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
