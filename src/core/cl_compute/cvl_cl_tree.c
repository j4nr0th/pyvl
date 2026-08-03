/*
 * Persistent OpenCL tree handle with asynchronous build/eval jobs.
 *
 * See cvl_cl_tree.h for the design.  The build pipeline is:
 *
 *   build_begin:
 *     - write sources to device staging (async, chained)
 *   build_finish (host):
 *     - compute bounding box + Morton codes (host)
 *     - qsort (host)
 *     - cvl_cl_flat_tree_count + cvl_cl_flat_tree_build (host)
 *     - upload flat nodes / particle_order / depth_offsets
 *     - launch kernel_p2m_leaves (leaf multipoles + Γ-weighted centroids)
 *     - launch kernel_build_internal_m2m per depth level (bottom-up M2M)
 *     - chain_finish; update metadata
 *
 * The host-side steps run inside build_finish (the "future" semantics);
 * build_begin returns immediately after the async source upload.  Eval
 * jobs are enqueued on the same in-order queue, so they serialize
 * against any in-flight build.
 */

#include "cvl_cl_tree.h"
#include "../multipole.h"
#include "../opencl/cvl_cl_helpers.h"
#include "cvl_cl_gpu_tree_build.h" /* CVL_CL_GPU_BUILD_WG */

#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/*  Host-side Morton helpers (same algorithm as cvl_cl_math.h.cl)      */
/* ------------------------------------------------------------------ */

static uint64_t tree_morton_split_21(uint64_t x)
{
    x &= 0x1fffffULL;
    x = (x | (x << 32)) & 0x1f00000000ffffULL;
    x = (x | (x << 16)) & 0x1f0000ff0000ffULL;
    x = (x | (x << 8)) & 0x100f00f00f00f00fULL;
    x = (x | (x << 4)) & 0x10c30c30c30c30c3ULL;
    x = (x | (x << 2)) & 0x1249249249249249ULL;
    return x;
}

static uint64_t tree_morton_3d(real3_t p, real3_t root_center, real_t root_half_size)
{
    const real_t inv_cell = (real_t)1.0 / ((real_t)2.0 * root_half_size);
    const real_t scale = (real_t)((1u << 21) - 1);
    real_t nx = (p.x - root_center.x) * inv_cell + (real_t)0.5;
    real_t ny = (p.y - root_center.y) * inv_cell + (real_t)0.5;
    real_t nz = (p.z - root_center.z) * inv_cell + (real_t)0.5;
    if (nx < 0)
        nx = 0;
    if (nx >= 1)
        nx = (real_t)0.999999;
    if (ny < 0)
        ny = 0;
    if (ny >= 1)
        ny = (real_t)0.999999;
    if (nz < 0)
        nz = 0;
    if (nz >= 1)
        nz = (real_t)0.999999;
    const uint64_t ix = (uint64_t)(nx * scale);
    const uint64_t iy = (uint64_t)(ny * scale);
    const uint64_t iz = (uint64_t)(nz * scale);
    return tree_morton_split_21(ix) | (tree_morton_split_21(iy) << 1) | (tree_morton_split_21(iz) << 2);
}

typedef struct
{
    uint64_t code;
    unsigned idx;
} tree_morton_entry_t;

static int tree_morton_cmp(const void *a, const void *b)
{
    const tree_morton_entry_t *ea = (const tree_morton_entry_t *)a;
    const tree_morton_entry_t *eb = (const tree_morton_entry_t *)b;
    if (ea->code < eb->code)
        return -1;
    if (ea->code > eb->code)
        return 1;
    return 0;
}

/* ------------------------------------------------------------------ */
/*  Lifecycle                                                          */
/* ------------------------------------------------------------------ */

cvl_cl_status_t cvl_cl_tree_init(cvl_cl_tree_t *tree, cvl_cl_compute_t *compute, cvl_cl_precision_t precision,
                                 const cvl_cl_flat_tree_settings_t *settings, unsigned work_order)
{
    assert(tree);
    assert(compute);
    assert(settings);

    *tree = (cvl_cl_tree_t){0};
    tree->compute = compute;
    tree->precision = precision;
    tree->settings = *settings;
    tree->work_order = work_order;

    cvl_cl_staging_buffer_init(&tree->buf_src_pos, precision);
    cvl_cl_staging_buffer_init(&tree->buf_src_val, precision);
    cvl_cl_staging_buffer_init(&tree->buf_targets, precision);
    cvl_cl_staging_buffer_init(&tree->buf_results, precision);

    return CVL_CL_SUCCESS;
}

void cvl_cl_tree_destroy(cvl_cl_tree_t *tree)
{
    if (!tree)
        return;

    cvl_cl_staging_buffer_destroy(&tree->buf_results);
    cvl_cl_staging_buffer_destroy(&tree->buf_targets);
    cvl_cl_staging_buffer_destroy(&tree->buf_src_val);
    cvl_cl_staging_buffer_destroy(&tree->buf_src_pos);

    cvl_cl_buffer_destroy(&tree->buf_parent_starts);
    cvl_cl_buffer_destroy(&tree->buf_m2m_scratch);
    cvl_cl_buffer_destroy(&tree->buf_p2m_scratch);
    cvl_cl_buffer_destroy(&tree->buf_coeffs);
    cvl_cl_buffer_destroy(&tree->buf_depth_offsets);
    cvl_cl_buffer_destroy(&tree->buf_particle_order);
    cvl_cl_buffer_destroy(&tree->buf_nodes);

    *tree = (cvl_cl_tree_t){0};
}

/* ------------------------------------------------------------------ */
/*  Build                                                              */
/* ------------------------------------------------------------------ */

size_t cvl_cl_tree_build_work_size(unsigned n_sources, unsigned max_depth)
{
    /* Worst-case n_total for the flat tree build. */
    const size_t max_n_total = (size_t)n_sources * (max_depth + 1u);
    return (size_t)n_sources * sizeof(uint64_t) +            /* morton codes */
           (size_t)n_sources * sizeof(unsigned) +            /* particle indices (permutation) */
           (size_t)n_sources * sizeof(tree_morton_entry_t) + /* entries for qsort */
           cvl_cl_flat_tree_work_size(max_n_total, n_sources, max_depth) +
           (size_t)(max_depth + 2) * sizeof(unsigned); /* depth counts */
}

cvl_cl_status_t cvl_cl_tree_build_begin(cvl_cl_tree_t *tree, unsigned n_sources,
                                        const real3_t sources_coords[restrict n_sources],
                                        const real3_t sources_values[restrict n_sources], cvl_cl_tree_build_job_t *job)
{
    assert(tree);
    assert(job);

    if (tree->build_in_flight)
        return CVL_CL_ERR_INVALID_PARAM;

    *job = (cvl_cl_tree_build_job_t){.tree = tree};
    cvl_cl_chain_init(&job->chain, tree->compute->queue);

    /* Ensure staging capacity and enqueue the async source upload. */
    cvl_cl_status_t st;
    if ((st = cvl_cl_staging_buffer_reserve_chained(&tree->buf_src_pos, tree->compute->ctx, &job->chain, n_sources)) !=
            CVL_CL_SUCCESS ||
        (st = cvl_cl_staging_buffer_reserve_chained(&tree->buf_src_val, tree->compute->ctx, &job->chain, n_sources)) !=
            CVL_CL_SUCCESS ||
        (st = cvl_cl_staging_buffer_write_async(&tree->buf_src_pos, &job->chain, sources_coords, NULL, n_sources, 0,
                                                NULL)) != CVL_CL_SUCCESS ||
        (st = cvl_cl_staging_buffer_write_async(&tree->buf_src_val, &job->chain, sources_values, NULL, n_sources, 0,
                                                NULL)) != CVL_CL_SUCCESS)
    {
        cvl_cl_chain_destroy(&job->chain);
        return st;
    }

    tree->n_sources = n_sources;
    tree->build_in_flight = true;
    tree->build_chain = job->chain;
    return CVL_CL_SUCCESS;
}

cvl_cl_status_t cvl_cl_tree_build_finish(cvl_cl_tree_build_job_t *job, void *work, size_t work_size)
{
    assert(job);
    assert(work);
    cvl_cl_tree_t *tree = job->tree;
    assert(tree);
    assert(tree->build_in_flight);

    const unsigned n_sources = tree->n_sources;
    const unsigned max_depth = tree->settings.max_depth;
    cvl_cl_status_t status = CVL_CL_SUCCESS;

    /* ---- 1. Partition the work buffer ---- */
    uint8_t *bp = (uint8_t *)work;
    uint64_t *morton_codes = (uint64_t *)bp;
    bp += (size_t)n_sources * sizeof(uint64_t);
    unsigned *particle_indices = (unsigned *)bp;
    bp += (size_t)n_sources * sizeof(unsigned);
    tree_morton_entry_t *entries = (tree_morton_entry_t *)bp;
    bp += (size_t)n_sources * sizeof(tree_morton_entry_t);
    unsigned *depth_counts = (unsigned *)bp;
    bp += (size_t)(max_depth + 2) * sizeof(unsigned);
    void *flat_work = bp;
    const size_t flat_work_avail = work_size - (size_t)(bp - (uint8_t *)work);

    if (work_size < cvl_cl_tree_build_work_size(n_sources, max_depth))
        return CVL_CL_ERR_BUFFER_SIZE;

    /* ---- 2. Read back the uploaded sources (waits on the upload) ---- */
    real3_t *coords = (real3_t *)malloc((size_t)n_sources * sizeof(real3_t));
    real3_t *values = (real3_t *)malloc((size_t)n_sources * sizeof(real3_t));
    if (!coords || !values)
    {
        free(coords);
        free(values);
        cvl_cl_chain_finish(&job->chain);
        return CVL_CL_ERR_MEMORY;
    }
    status = cvl_cl_staging_buffer_read_and_wait(&tree->buf_src_pos, &job->chain, coords, NULL, n_sources, 0);
    if (status != CVL_CL_SUCCESS)
        goto cleanup_host;
    status = cvl_cl_staging_buffer_read_and_wait(&tree->buf_src_val, &job->chain, values, NULL, n_sources, 0);
    if (status != CVL_CL_SUCCESS)
        goto cleanup_host;

    /* ---- 3. Bounding box + Morton codes + sort ---- */
    {
        real3_t bbox_min = coords[0];
        real3_t bbox_max = coords[0];
        for (unsigned i = 1; i < n_sources; ++i)
        {
            if (coords[i].x < bbox_min.x)
                bbox_min.x = coords[i].x;
            if (coords[i].y < bbox_min.y)
                bbox_min.y = coords[i].y;
            if (coords[i].z < bbox_min.z)
                bbox_min.z = coords[i].z;
            if (coords[i].x > bbox_max.x)
                bbox_max.x = coords[i].x;
            if (coords[i].y > bbox_max.y)
                bbox_max.y = coords[i].y;
            if (coords[i].z > bbox_max.z)
                bbox_max.z = coords[i].z;
        }
        const real_t root_extent =
            fmax(fmax(bbox_max.x - bbox_min.x, bbox_max.y - bbox_min.y), bbox_max.z - bbox_min.z);
        const real3_t root_center = {(bbox_min.x + bbox_max.x) * (real_t)0.5, (bbox_min.y + bbox_max.y) * (real_t)0.5,
                                     (bbox_min.z + bbox_max.z) * (real_t)0.5};
        const real_t root_hs = root_extent * (real_t)0.5 + (real_t)1e-12;

        for (unsigned i = 0; i < n_sources; ++i)
            entries[i] = (tree_morton_entry_t){.code = tree_morton_3d(coords[i], root_center, root_hs), .idx = i};
        qsort(entries, n_sources, sizeof(tree_morton_entry_t), tree_morton_cmp);
        for (unsigned i = 0; i < n_sources; ++i)
        {
            morton_codes[i] = entries[i].code;
            particle_indices[i] = entries[i].idx;
        }
    }

    /* ---- 4. Count + build the flat tree into the work buffer ---- */
    unsigned n_total = 0, max_depth_used = 0;
    status = cvl_cl_flat_tree_count(n_sources, morton_codes, &tree->settings, depth_counts, &n_total, &max_depth_used);
    if (status != CVL_CL_SUCCESS)
        goto cleanup_host;

    const size_t needed_flat = cvl_cl_flat_tree_work_size(n_total, n_sources, max_depth);
    if (flat_work_avail < needed_flat)
    {
        status = CVL_CL_ERR_BUFFER_SIZE;
        goto cleanup_host;
    }

    cvl_cl_flat_tree_t host_tree = {0};
    status = cvl_cl_flat_tree_build(n_sources, coords, particle_indices, morton_codes, &tree->settings, &host_tree,
                                    flat_work, needed_flat);
    if (status != CVL_CL_SUCCESS)
        goto cleanup_host;

    /* ---- 5. Upload flat tree + run P2M/M2M ---- */
    {
        const unsigned n_coeffs = (unsigned)multipole_num_coeffs(tree->settings.order);
        const size_t scratch_sz = multipole_scratch_size(tree->settings.order);
        const size_t nodes_bytes = (size_t)n_total * sizeof(cvl_cl_flat_node_t);
        const size_t order_bytes = (size_t)n_sources * sizeof(unsigned);
        const size_t doff_bytes = (size_t)(max_depth + 2) * sizeof(unsigned);
        const size_t coeffs_bytes = (size_t)n_total * 3u * n_coeffs * sizeof(real_t);
        const unsigned n_internal = n_total - host_tree.n_multipole_leaves - host_tree.n_particle_leaves;

        /* Scratch sizes (see bh_p2m_m2m.cl.h). */
        const size_t wo = tree->work_order ? tree->work_order : tree->settings.order;
        const size_t wo_scratch = multipole_scratch_size((unsigned)wo);
        const size_t wo_coeffs = multipole_num_coeffs((unsigned)wo);
        const size_t shift_plane = (size_t)(wo + 1) * (size_t)(wo + 1);
        const size_t m2m_per_wg = 3u * shift_plane + 2u * wo_coeffs + 2u * wo_scratch;
        const size_t n_leaves = host_tree.n_multipole_leaves + host_tree.n_particle_leaves;

        /* Ensure device buffers (grow-only). */
        const cl_context ctx = tree->compute->ctx;
        const cl_command_queue queue = tree->compute->queue;
        cvl_cl_chain_t *ch = &job->chain;

        if ((status = cl_ensure_buffer_chained(&tree->buf_nodes, ctx, ch, nodes_bytes)) != CVL_CL_SUCCESS ||
            (status = cl_ensure_buffer_chained(&tree->buf_particle_order, ctx, ch, order_bytes)) != CVL_CL_SUCCESS ||
            (status = cl_ensure_buffer_chained(&tree->buf_depth_offsets, ctx, ch, doff_bytes)) != CVL_CL_SUCCESS ||
            (status = cl_ensure_buffer_chained(&tree->buf_coeffs, ctx, ch, coeffs_bytes)) != CVL_CL_SUCCESS ||
            (status = cl_ensure_buffer_chained(&tree->buf_p2m_scratch, ctx, ch,
                                               (size_t)n_leaves * 2u * scratch_sz * sizeof(real_t))) !=
                CVL_CL_SUCCESS ||
            (status = cl_ensure_buffer_chained(&tree->buf_m2m_scratch, ctx, ch,
                                               (size_t)n_internal * m2m_per_wg * sizeof(real_t))) != CVL_CL_SUCCESS ||
            (status = cl_ensure_buffer_chained(&tree->buf_parent_starts, ctx, ch,
                                               ((size_t)n_internal + 1u) * sizeof(unsigned))) != CVL_CL_SUCCESS)
            goto cleanup_host;

        /* Upload nodes + order + depth offsets. */
        if ((status = cvl_cl_chain_write_buffer(ch, &tree->buf_nodes, 0, nodes_bytes, host_tree.nodes, 0, NULL,
                                                NULL)) != CVL_CL_SUCCESS ||
            (status = cvl_cl_chain_write_buffer(ch, &tree->buf_particle_order, 0, order_bytes, host_tree.particle_order,
                                                0, NULL, NULL)) != CVL_CL_SUCCESS ||
            (status = cvl_cl_chain_write_buffer(ch, &tree->buf_depth_offsets, 0, doff_bytes, host_tree.depth_offsets, 0,
                                                NULL, NULL)) != CVL_CL_SUCCESS)
            goto cleanup_host;

        /* ---- P2M: leaf multipoles + Γ-weighted centroids ---- */
        {
            cl_kernel k = cvl_cl_compute_kernel(tree->compute, CVL_CL_PACK_BH_COEFFS, CVL_CL_BH_COEFFS_P2M);
            if (!k)
            {
                status = CVL_CL_ERR_INTERNAL;
                goto cleanup_host;
            }
            const unsigned leaf_offset = host_tree.depth_offsets[max_depth];
            const size_t global = ((n_leaves + CVL_CL_GPU_BUILD_WG - 1) / CVL_CL_GPU_BUILD_WG) * CVL_CL_GPU_BUILD_WG;
            status = cvl_cl_chain_ndrange(
                ch, k, 1, &global, NULL,
                (cvl_cl_karg_t[]){
                    {.type = CVL_CL_KARG_BUFFER, .index = 0, .mem = tree->buf_nodes.mem},
                    {.type = CVL_CL_KARG_SCALAR_UINT, .index = 1, .scalar_uint = n_leaves},
                    {.type = CVL_CL_KARG_SCALAR_UINT, .index = 2, .scalar_uint = leaf_offset},
                    {.type = CVL_CL_KARG_SCALAR_UINT, .index = 3, .scalar_uint = n_total},
                    {.type = CVL_CL_KARG_BUFFER, .index = 4, .mem = tree->buf_particle_order.mem},
                    {.type = CVL_CL_KARG_BUFFER, .index = 5, .mem = tree->buf_src_pos.device.mem},
                    {.type = CVL_CL_KARG_BUFFER, .index = 6, .mem = tree->buf_src_val.device.mem},
                    {.type = CVL_CL_KARG_SCALAR_UINT, .index = 7, .scalar_uint = tree->settings.order},
                    {.type = CVL_CL_KARG_SCALAR_UINT, .index = 8, .scalar_uint = tree->work_order},
                    {.type = CVL_CL_KARG_BUFFER, .index = 9, .mem = tree->buf_coeffs.mem},
                    {.type = CVL_CL_KARG_BUFFER, .index = 10, .mem = tree->buf_p2m_scratch.mem},
                    {},
                },
                0, NULL, NULL);
            if (status != CVL_CL_SUCCESS)
                goto cleanup_host;
        }

        /* ---- M2M: bottom-up internal levels ---- */
        for (int d = (int)max_depth - 1; d >= 0; --d)
        {
            const unsigned depth = (unsigned)d;
            const unsigned n_parents = host_tree.depth_offsets[depth + 1] - host_tree.depth_offsets[depth];
            if (n_parents == 0)
                continue;

            cl_kernel k = cvl_cl_compute_kernel(tree->compute, CVL_CL_PACK_BH_COEFFS, CVL_CL_BH_COEFFS_INTERNAL);
            if (!k)
            {
                status = CVL_CL_ERR_INTERNAL;
                goto cleanup_host;
            }

            /* parent_starts[p] = child_base of parent p; sentinel at end. */
            unsigned *parent_starts = (unsigned *)malloc(((size_t)n_parents + 1u) * sizeof(unsigned));
            if (!parent_starts)
            {
                status = CVL_CL_ERR_MEMORY;
                goto cleanup_host;
            }
            for (unsigned p = 0; p < n_parents; ++p)
            {
                const unsigned ni = host_tree.depth_offsets[depth] + p;
                parent_starts[p] = (unsigned)host_tree.nodes[ni].child_base;
            }
            parent_starts[n_parents] = host_tree.depth_offsets[depth + 1] +
                                       (host_tree.depth_offsets[depth + 2] - host_tree.depth_offsets[depth + 1]);

            status =
                cvl_cl_chain_write_buffer(ch, &tree->buf_parent_starts, 0, ((size_t)n_parents + 1u) * sizeof(unsigned),
                                          parent_starts, 0, NULL, NULL);
            free(parent_starts);
            if (status != CVL_CL_SUCCESS)
                goto cleanup_host;

            const unsigned child_offset = host_tree.depth_offsets[depth + 1];
            const unsigned n_children = host_tree.depth_offsets[depth + 2] - host_tree.depth_offsets[depth + 1];
            const size_t global = ((n_parents + CVL_CL_GPU_BUILD_WG - 1) / CVL_CL_GPU_BUILD_WG) * CVL_CL_GPU_BUILD_WG;
            status = cvl_cl_chain_ndrange(
                ch, k, 1, &global, NULL,
                (cvl_cl_karg_t[]){
                    {.type = CVL_CL_KARG_BUFFER, .index = 0, .mem = tree->buf_nodes.mem},
                    {.type = CVL_CL_KARG_SCALAR_UINT, .index = 1, .scalar_uint = depth},
                    {.type = CVL_CL_KARG_SCALAR_UINT, .index = 2, .scalar_uint = n_parents},
                    {.type = CVL_CL_KARG_SCALAR_UINT, .index = 3, .scalar_uint = host_tree.depth_offsets[depth]},
                    {.type = CVL_CL_KARG_SCALAR_UINT, .index = 4, .scalar_uint = child_offset},
                    {.type = CVL_CL_KARG_SCALAR_UINT, .index = 5, .scalar_uint = n_children},
                    {.type = CVL_CL_KARG_SCALAR_DOUBLE, .index = 6, .scalar_double = 0.0},
                    {.type = CVL_CL_KARG_BUFFER, .index = 7, .mem = tree->buf_parent_starts.mem},
                    {.type = CVL_CL_KARG_BUFFER, .index = 8, .mem = tree->buf_particle_order.mem},
                    {.type = CVL_CL_KARG_BUFFER, .index = 9, .mem = tree->buf_src_pos.device.mem},
                    {.type = CVL_CL_KARG_BUFFER, .index = 10, .mem = tree->buf_src_val.device.mem},
                    {.type = CVL_CL_KARG_SCALAR_UINT, .index = 11, .scalar_uint = tree->settings.order},
                    {.type = CVL_CL_KARG_SCALAR_UINT, .index = 12, .scalar_uint = tree->work_order},
                    {.type = CVL_CL_KARG_BUFFER, .index = 13, .mem = tree->buf_coeffs.mem},
                    {.type = CVL_CL_KARG_BUFFER, .index = 14, .mem = tree->buf_m2m_scratch.mem},
                    {},
                },
                0, NULL, NULL);
            if (status != CVL_CL_SUCCESS)
                goto cleanup_host;
        }

        /* ---- Finish: wait for the full build pipeline ---- */
        status = cvl_cl_chain_finish(ch);
        if (status != CVL_CL_SUCCESS)
            goto cleanup_host;

        /* ---- Update metadata ---- */
        tree->n_nodes = n_total;
        tree->n_internal = n_internal;
        tree->n_multipole_leaves = host_tree.n_multipole_leaves;
        tree->n_particle_leaves = host_tree.n_particle_leaves;
        tree->n_leaves = n_leaves;
        tree->max_depth_used = max_depth_used;
        tree->built = true;
    }

cleanup_host:
    free(coords);
    free(values);

    tree->build_in_flight = false;
    job->finished = true;
    cvl_cl_chain_destroy(&job->chain);
    return status;
}

cvl_cl_status_t cvl_cl_tree_build(cvl_cl_tree_t *tree, unsigned n_sources,
                                  const real3_t sources_coords[restrict n_sources],
                                  const real3_t sources_values[restrict n_sources], void *work, size_t work_size)
{
    cvl_cl_tree_build_job_t job;
    cvl_cl_status_t st = cvl_cl_tree_build_begin(tree, n_sources, sources_coords, sources_values, &job);
    if (st != CVL_CL_SUCCESS)
        return st;
    return cvl_cl_tree_build_finish(&job, work, work_size);
}

/* ------------------------------------------------------------------ */
/*  Eval                                                               */
/* ------------------------------------------------------------------ */

cvl_cl_status_t cvl_cl_tree_eval_begin(cvl_cl_tree_t *tree, unsigned n_targets, const real3_t targets[restrict],
                                       cvl_cl_tree_eval_mode_t mode, double theta, cvl_cl_tree_eval_job_t *job)
{
    assert(tree);
    assert(job);

    if (!tree->built)
        return CVL_CL_ERR_INVALID_PARAM;

    *job = (cvl_cl_tree_eval_job_t){
        .tree = tree,
        .n_targets = n_targets,
        .mode = mode,
        .theta = theta,
    };
    cvl_cl_chain_init(&job->chain, tree->compute->queue);
    cvl_cl_staging_buffer_init(&job->targets, tree->precision);
    cvl_cl_staging_buffer_init(&job->results, tree->precision);

    cvl_cl_status_t st;
    if ((st = cvl_cl_staging_buffer_reserve_chained(&job->targets, tree->compute->ctx, &job->chain, n_targets)) !=
            CVL_CL_SUCCESS ||
        (st = cvl_cl_staging_buffer_reserve_chained(&job->results, tree->compute->ctx, &job->chain, n_targets)) !=
            CVL_CL_SUCCESS ||
        (st = cvl_cl_staging_buffer_write_async(&job->targets, &job->chain, targets, NULL, n_targets, 0, NULL)) !=
            CVL_CL_SUCCESS)
    {
        cvl_cl_staging_buffer_destroy(&job->results);
        cvl_cl_staging_buffer_destroy(&job->targets);
        cvl_cl_chain_destroy(&job->chain);
        return st;
    }

    /* Enqueue the eval kernel. */
    if (mode == CVL_CL_TREE_EVAL_DIRECT)
    {
        cl_kernel k = cvl_cl_compute_kernel(tree->compute, CVL_CL_PACK_DIRECT_SUM, CVL_CL_DIRECT_SUM_KERNEL);
        if (!k)
        {
            cvl_cl_staging_buffer_destroy(&job->results);
            cvl_cl_staging_buffer_destroy(&job->targets);
            cvl_cl_chain_destroy(&job->chain);
            return CVL_CL_ERR_NOT_FOUND;
        }
        const size_t global = n_targets;
        st = cvl_cl_chain_ndrange(&job->chain, k, 1, &global, NULL,
                                  (cvl_cl_karg_t[]){
                                      {.type = CVL_CL_KARG_BUFFER, .index = 0, .mem = job->targets.device.mem},
                                      {.type = CVL_CL_KARG_BUFFER, .index = 1, .mem = tree->buf_src_pos.device.mem},
                                      {.type = CVL_CL_KARG_BUFFER, .index = 2, .mem = tree->buf_src_val.device.mem},
                                      {.type = CVL_CL_KARG_SCALAR_UINT, .index = 3, .scalar_uint = tree->n_sources},
                                      {.type = CVL_CL_KARG_SCALAR_UINT, .index = 4, .scalar_uint = n_targets},
                                      {.type = CVL_CL_KARG_BUFFER, .index = 5, .mem = job->results.device.mem},
                                      {},
                                  },
                                  0, NULL, NULL);
    }
    else
    {
        cl_kernel k = cvl_cl_compute_kernel(tree->compute, CVL_CL_PACK_BH_EVAL, CVL_CL_BH_EVAL_FLAT_EVAL);
        if (!k)
        {
            cvl_cl_staging_buffer_destroy(&job->results);
            cvl_cl_staging_buffer_destroy(&job->targets);
            cvl_cl_chain_destroy(&job->chain);
            return CVL_CL_ERR_NOT_FOUND;
        }
        const size_t global = n_targets;
        st =
            cvl_cl_chain_ndrange(&job->chain, k, 1, &global, NULL,
                                 (cvl_cl_karg_t[]){
                                     {.type = CVL_CL_KARG_BUFFER, .index = 0, .mem = tree->buf_nodes.mem},
                                     {.type = CVL_CL_KARG_BUFFER, .index = 1, .mem = tree->buf_particle_order.mem},
                                     {.type = CVL_CL_KARG_BUFFER, .index = 2, .mem = tree->buf_depth_offsets.mem},
                                     {.type = CVL_CL_KARG_BUFFER, .index = 3, .mem = tree->buf_src_pos.device.mem},
                                     {.type = CVL_CL_KARG_BUFFER, .index = 4, .mem = tree->buf_src_val.device.mem},
                                     {.type = CVL_CL_KARG_SCALAR_UINT, .index = 5, .scalar_uint = n_targets},
                                     {.type = CVL_CL_KARG_SCALAR_UINT, .index = 6, .scalar_uint = tree->settings.order},
                                     {.type = CVL_CL_KARG_SCALAR_DOUBLE, .index = 7, .scalar_double = theta},
                                     {.type = CVL_CL_KARG_BUFFER, .index = 8, .mem = tree->buf_coeffs.mem},
                                     {.type = CVL_CL_KARG_BUFFER, .index = 9, .mem = job->targets.device.mem},
                                     {.type = CVL_CL_KARG_BUFFER, .index = 10, .mem = job->results.device.mem},
                                     {},
                                 },
                                 0, NULL, NULL);
    }
    if (st != CVL_CL_SUCCESS)
    {
        cvl_cl_staging_buffer_destroy(&job->results);
        cvl_cl_staging_buffer_destroy(&job->targets);
        cvl_cl_chain_destroy(&job->chain);
        return st;
    }

    return CVL_CL_SUCCESS;
}

cvl_cl_status_t cvl_cl_tree_eval_finish(cvl_cl_tree_eval_job_t *job, real3_t out[restrict], float *scratch_f32)
{
    assert(job);
    assert(out);

    cvl_cl_status_t st =
        cvl_cl_staging_buffer_read_and_wait(&job->results, &job->chain, out, scratch_f32, job->n_targets, 0);
    job->finished = true;
    cvl_cl_staging_buffer_destroy(&job->results);
    cvl_cl_staging_buffer_destroy(&job->targets);
    cvl_cl_chain_destroy(&job->chain);
    return st;
}

void cvl_cl_tree_eval_cancel(cvl_cl_tree_eval_job_t *job)
{
    if (!job)
        return;
    cvl_cl_staging_buffer_destroy(&job->results);
    cvl_cl_staging_buffer_destroy(&job->targets);
    cvl_cl_chain_destroy(&job->chain);
    job->finished = true;
}

cvl_cl_status_t cvl_cl_tree_eval(cvl_cl_tree_t *tree, unsigned n_targets, const real3_t targets[restrict],
                                 cvl_cl_tree_eval_mode_t mode, double theta, real3_t out[restrict], float *scratch_f32)
{
    cvl_cl_tree_eval_job_t job;
    cvl_cl_status_t st = cvl_cl_tree_eval_begin(tree, n_targets, targets, mode, theta, &job);
    if (st != CVL_CL_SUCCESS)
        return st;
    return cvl_cl_tree_eval_finish(&job, out, scratch_f32);
}
