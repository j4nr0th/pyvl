/*
 * GPU-accelerated FMM evaluation (L2P) - host-side orchestration.
 *
 * Implements the pipeline declared in cvl_cl_fmm_eval.h:
 *
 *   1. Convert the CPU-built fmm_tree_t to a flat, GPU-friendly layout:
 *        - flat nodes (64 B each, geom_center + child_base + child_mask)
 *        - eval_centers (Γ-weighted centroids, 3 * n_nodes doubles)
 *        - particle_order (n_sources unsigned)
 *        - local_coeffs (3 * n_coeffs * n_nodes doubles, already populated
 *          by the CPU build's M2L + L2L sweep)
 *   2. Upload tree + source data + targets to the device.
 *   3. Launch the fmm_l2p_eval kernel (one work-item per target).
 *   4. Read back the induced field at every target.
 *
 * The kernel descends the flat octree to the leaf containing each target,
 * evaluates the leaf's precomputed local expansion, and adds the
 * near-field direct sum over the leaf's own particles.  This is the GPU
 * analogue of fmm_tree_eval(..., FMM_EVAL_FMM) for interior points.
 *
 * All device buffers are grown on demand via cvl_cl_buffer_reserve and
 * reused across runs (grow-only, never shrinks).  Cleanup frees buffers
 * in reverse allocation order.
 */

#include "cvl_cl_fmm_eval.h"
#include "../multipole.h"
#include "../octree.h"
#include "../opencl/cvl_cl_buffer.h"
#include "../opencl/cvl_cl_command.h"
#include "../opencl/cvl_cl_common.h"
#include "../opencl/cvl_cl_ctx.h"
#include "../opencl/cvl_cl_helpers.h"
#include "../opencl/cvl_cl_kernel.h"

#include <string.h>

/* ------------------------------------------------------------------ */
/*  Internal helpers                                                   */
/* ------------------------------------------------------------------ */

/**
 * @brief Flat node layout on the host (must match fmm_l2p_flat_node_t
 *        in fmm_l2p.cl.h and cvl_cl_flat_node_t in cvl_cl_flat_tree.h).
 *
 * 64 bytes: center (24) | half_size (8) | morton_code (8) |
 *           child_base (4) | particle_begin (4) | child_mask (1) |
 *           kind (1) | particle_count (2) | leaf_id (4) | pad (8).
 */
typedef struct
{
    real3_t center; /* geom_center - used for descent. */
    real_t half_size;
    uint64_t morton_code; /* unused by the kernel, zeroed here. */
    int32_t child_base;
    int32_t particle_begin;
    uint8_t child_mask;
    uint8_t kind;
    int16_t particle_count;
    int32_t leaf_id; /* leaf index for nflist lookup, -1 for internal. */
    uint8_t pad[8];
} fmm_eval_flat_node_t;

_Static_assert(sizeof(fmm_eval_flat_node_t) == 64, "fmm_eval_flat_node_t must be 64 bytes");

/**
 * @brief Compact flat node layout for FP32 kernels.
 *
 * Mirrors fmm_l2p_flat_node_t in fmm_l2p.cl.h compiled with real_t=float:
 *   center[3] (12) | half_size (4) | morton_code (8) | child_base (4) |
 *   particle_begin (4) | child_mask (1) | kind (1) | particle_count (2) |
 *   leaf_id (4) | pad (8)  → 48 bytes.
 *
 * The FP64 layout above is 64 bytes (real_t=double), so the host MUST
 * convert before upload in FP32 mode - otherwise every field offset
 * after `center` is wrong on the device and the kernel reads garbage
 * particle_begin/count/leaf_id values.
 */
typedef struct
{
    float center[3];        /* 12 bytes  */
    float half_size;        /*  4 bytes  */
    uint64_t morton_code;   /*  8 bytes  (unused by the kernel) */
    int32_t child_base;     /*  4 bytes  */
    int32_t particle_begin; /*  4 bytes  */
    uint8_t child_mask;     /*  1 byte   */
    uint8_t kind;           /*  1 byte   */
    int16_t particle_count; /*  2 bytes  */
    int32_t leaf_id;        /*  4 bytes  */
    uint8_t pad[8];         /*  8 bytes  → 48 total */
} fmm_eval_flat_node_f32_t;

_Static_assert(sizeof(fmm_eval_flat_node_f32_t) == 48, "fmm_eval_flat_node_f32_t must be 48 bytes");

/**
 * @brief Convert the FP64 flat-tree intermediate to the FP32 layout.
 *
 * @param tree         CPU-built FMM tree (dimensions only).
 * @param src          FP64 flat nodes (from flatten_fmm_tree).
 * @param src_centers  FP64 eval centres [3 * n_nodes].
 * @param out_nodes    Pre-allocated [n_nodes] FP32 nodes.
 * @param out_centers  Pre-allocated [3 * n_nodes] float centres.
 */
static void flatten_fmm_tree_f32(const fmm_tree_t *tree, const fmm_eval_flat_node_t *src, const real_t *src_centers,
                                 fmm_eval_flat_node_f32_t *out_nodes, float *out_centers)
{
    const unsigned n_nodes = tree->n_nodes;
    for (unsigned ni = 0; ni < n_nodes; ++ni)
    {
        const fmm_eval_flat_node_t *s = &src[ni];
        fmm_eval_flat_node_f32_t *d = &out_nodes[ni];
        d->center[0] = (float)s->center.x;
        d->center[1] = (float)s->center.y;
        d->center[2] = (float)s->center.z;
        d->half_size = (float)s->half_size;
        d->morton_code = s->morton_code;
        d->child_base = s->child_base;
        d->particle_begin = s->particle_begin;
        d->child_mask = s->child_mask;
        d->kind = s->kind;
        d->particle_count = s->particle_count;
        d->leaf_id = s->leaf_id;

        out_centers[3u * ni + 0u] = (float)src_centers[3u * ni + 0u];
        out_centers[3u * ni + 1u] = (float)src_centers[3u * ni + 1u];
        out_centers[3u * ni + 2u] = (float)src_centers[3u * ni + 2u];
    }
}

/**
 * @brief Convert a double coefficient array to float.
 */
static void convert_coeffs_f32(const real_t *src, size_t n, float *dst)
{
    for (size_t i = 0; i < n; ++i)
        dst[i] = (float)src[i];
}

/**
 * @brief Convert a CPU-built fmm_tree_t to the flat GPU layout.
 *
 * Walks the octree_node_t array once and produces:
 *   - flat_nodes[n_nodes]  (geom_center, child_base, child_mask, kind,
 *     particle_begin/count)
 *   - eval_centers[3 * n_nodes]  (Γ-weighted centroids = node.center)
 *
 * Child indices are computed by pointer subtraction
 * (child - tree->nodes).  Children are stored compactly in the flat
 * array, so child_base is the index of the first present child and
 * child_mask records which octants are occupied.  The kernel recovers
 * a child's flat index via popcount(child_mask & ((1<<oct)-1)).
 *
 * @param tree        CPU-built FMM tree.
 * @param flat_nodes  Pre-allocated [n_nodes] output array.
 * @param eval_centers Pre-allocated [3 * n_nodes] output array.
 */
static void flatten_fmm_tree(const fmm_tree_t *tree, fmm_eval_flat_node_t *flat_nodes, real_t *eval_centers,
                             int32_t *child_indices)
{
    const octree_node_t *nodes = tree->nodes;
    const unsigned n_nodes = tree->n_nodes;

    for (unsigned ni = 0; ni < n_nodes; ++ni)
    {
        const octree_node_t *node = &nodes[ni];
        fmm_eval_flat_node_t *flat = &flat_nodes[ni];

        /* Geometric centre for descent. */
        flat->center = node->geom_center;
        flat->half_size = node->half_size;
        flat->morton_code = 0; /* unused by the kernel. */
        flat->particle_begin = (int32_t)node->particle_begin;
        flat->particle_count = (int16_t)node->particle_count;
        flat->kind = (uint8_t)node->kind;
        flat->leaf_id = node->leaf_id;

        /* Γ-weighted centroid for local-expansion evaluation. */
        eval_centers[3u * ni + 0u] = node->center.x;
        eval_centers[3u * ni + 1u] = node->center.y;
        eval_centers[3u * ni + 2u] = node->center.z;

        if (node->kind == OCTREE_NODE_INTERNAL)
        {
            /* Store explicit child indices for all 8 octants. */
            for (int oct = 0; oct < 8; ++oct)
            {
                const octree_node_t *child = node->data.internal.children[oct];
                child_indices[8u * ni + oct] = child ? (int32_t)(child - nodes) : -1;
            }
            flat->child_base = 0;
            flat->child_mask = 0;
        }
        else
        {
            for (int oct = 0; oct < 8; ++oct)
                child_indices[8u * ni + oct] = -1;
            flat->child_base = -1;
            flat->child_mask = 0;
        }
    }
}

/* ------------------------------------------------------------------ */
/*  Public API                                                         */
/* ------------------------------------------------------------------ */

cvl_cl_status_t cvl_cl_fmm_eval_init(cvl_cl_fmm_eval_t *eval, cvl_cl_compute_t *compute, cvl_cl_precision_t precision,
                                     const allocator_t *allocator)
{
    if (!eval || !compute)
        return CVL_CL_ERR_INVALID_PARAM;

    memset(eval, 0, sizeof(*eval));
    eval->compute = compute;
    eval->precision = precision;
    eval->allocator = cl_resolve_allocator(allocator);

    /* Verify the required kernel is registered. */
    if (cvl_cl_compute_kernel(compute, "fmm_l2p_eval") == NULL)
        return CVL_CL_ERR_NOT_FOUND;

    eval->initialized = true;
    return CVL_CL_SUCCESS;
}

size_t cvl_cl_fmm_eval_work_size(const cvl_cl_fmm_eval_t *eval, const fmm_tree_t *tree)
{
    const unsigned n_nodes = tree->n_nodes;
    const unsigned order = tree->settings.order;
    const size_t n_coeffs = multipole_num_coeffs(order);
    const size_t n_real = (size_t)3 * n_coeffs * n_nodes;
    const bool is_f32 = (eval->precision == CVL_CL_PRECISION_FP32);

    size_t size = 0;
    /* Always allocated (FP64 intermediates): */
    size += (size_t)n_nodes * sizeof(fmm_eval_flat_node_t); /* flat_nodes */
    size += (size_t)3 * n_nodes * sizeof(real_t);           /* eval_centers */
    size += (size_t)8 * n_nodes * sizeof(int32_t);          /* child_indices */

    if (is_f32)
    {
        size += (size_t)n_nodes * sizeof(fmm_eval_flat_node_f32_t); /* flat_nodes_f32 */
        size += (size_t)3 * n_nodes * sizeof(float);                /* eval_centers_f32 */
        size += n_real * sizeof(float);                             /* local_coeffs_f32 */
        size += n_real * sizeof(float);                             /* mp_coeffs_f32 */
    }
    return size;
}

cvl_cl_status_t cvl_cl_fmm_eval_run(cvl_cl_fmm_eval_t *eval, cvl_cl_queue_t *queue, const cvl_cl_ctx_t *ctx,
                                    const fmm_tree_t *tree, const real3_t *sources_coords,
                                    const real3_t *sources_values, unsigned n_targets, const real3_t *targets,
                                    real3_t *results, void *work, size_t work_size)
{
    if (!eval || !eval->initialized || !queue || !ctx || !tree || !sources_coords || !sources_values || !targets ||
        !results || !work)
        return CVL_CL_ERR_INVALID_PARAM;
    if (tree->n_nodes == 0 || tree->n_sources == 0)
        return CVL_CL_ERR_INVALID_PARAM;
    if (n_targets == 0)
        return CVL_CL_SUCCESS;

    cvl_cl_status_t status = CVL_CL_SUCCESS;

    /* Cached tree dimensions. */
    const unsigned n_nodes = tree->n_nodes;
    const unsigned n_sources = tree->n_sources;
    const unsigned order = tree->settings.order;
    const unsigned max_depth = tree->max_depth_reached;
    const size_t n_coeffs = multipole_num_coeffs(order);

    /* The kernel's real_t is float in FP32 mode, so the flat node struct
     * shrinks from 64 to 48 bytes and every real_t array halves in size.
     * Buffer sizes and upload pointers must match the device precision. */
    const bool is_f32 = (eval->precision == CVL_CL_PRECISION_FP32);
    const size_t node_struct_bytes = is_f32 ? sizeof(fmm_eval_flat_node_f32_t) : sizeof(fmm_eval_flat_node_t);
    const size_t real_bytes = is_f32 ? sizeof(float) : sizeof(real_t);
    const size_t n_real = (size_t)3 * n_coeffs * n_nodes;

    /* Verify work buffer is large enough. */
    if (work_size < cvl_cl_fmm_eval_work_size(eval, tree))
        return CVL_CL_ERR_INVALID_PARAM;

    /* ---- 1. Partition work buffer ---- */
    uint8_t *bp = (uint8_t *)work;

    fmm_eval_flat_node_t *flat_nodes = (fmm_eval_flat_node_t *)bp;
    bp += (size_t)n_nodes * sizeof(fmm_eval_flat_node_t);

    real_t *eval_centers = (real_t *)bp;
    bp += (size_t)3 * n_nodes * sizeof(real_t);

    int32_t *child_indices = (int32_t *)bp;
    bp += (size_t)8 * n_nodes * sizeof(int32_t);

    /* FP32 upload mirrors (populated only in FP32 mode). */
    fmm_eval_flat_node_f32_t *flat_nodes_f32 = NULL;
    float *eval_centers_f32 = NULL;
    float *local_coeffs_f32 = NULL;
    float *mp_coeffs_f32 = NULL;

    if (is_f32)
    {
        flat_nodes_f32 = (fmm_eval_flat_node_f32_t *)bp;
        bp += (size_t)n_nodes * node_struct_bytes;
        eval_centers_f32 = (float *)bp;
        bp += (size_t)3 * n_nodes * sizeof(float);
        local_coeffs_f32 = (float *)bp;
        bp += n_real * sizeof(float);
        mp_coeffs_f32 = (float *)bp;
        bp += n_real * sizeof(float);
    }

    /* ---- 2. Flatten the tree on the host ---- */
    flatten_fmm_tree(tree, flat_nodes, eval_centers, child_indices);

    /* In FP32 mode the kernel reads a compact float layout - convert the
     * FP64 intermediate before upload. */
    if (is_f32)
    {
        flatten_fmm_tree_f32(tree, flat_nodes, eval_centers, flat_nodes_f32, eval_centers_f32);
        convert_coeffs_f32(tree->local_coeffs, n_real, local_coeffs_f32);
        convert_coeffs_f32(tree->multipole_coeffs, n_real, mp_coeffs_f32);
    }

    /* ---- 2. Ensure device buffers are large enough ---- */
    const size_t nodes_bytes = (size_t)n_nodes * node_struct_bytes;
    const size_t centers_bytes = (size_t)3 * n_nodes * real_bytes;
    const size_t child_idx_bytes = (size_t)8 * n_nodes * sizeof(int32_t);
    const size_t mp_bytes = n_real * real_bytes;
    const size_t order_bytes = (size_t)n_sources * sizeof(unsigned);
    const size_t local_bytes = n_real * real_bytes;
    const size_t nflist_off_bytes = (size_t)(tree->n_leaves + 1) * sizeof(unsigned);
    const size_t nflist_idx_bytes = (size_t)tree->nflist_count * sizeof(unsigned);
    const size_t leaf_idx_bytes = (size_t)tree->n_leaves * sizeof(unsigned);

    status = cl_ensure_buffer(&eval->buf_nodes, ctx, queue, nodes_bytes);
    if (status != CVL_CL_SUCCESS)
        goto cleanup_host;
    status = cl_ensure_buffer(&eval->buf_eval_centers, ctx, queue, centers_bytes);
    if (status != CVL_CL_SUCCESS)
        goto cleanup_host;
    status = cl_ensure_buffer(&eval->buf_particle_order, ctx, queue, order_bytes);
    if (status != CVL_CL_SUCCESS)
        goto cleanup_host;
    status = cl_ensure_buffer(&eval->buf_local_coeffs, ctx, queue, local_bytes);
    if (status != CVL_CL_SUCCESS)
        goto cleanup_host;
    status = cl_ensure_buffer(&eval->buf_nflist_offsets, ctx, queue, nflist_off_bytes);
    if (status != CVL_CL_SUCCESS)
        goto cleanup_host;
    status = cl_ensure_buffer(&eval->buf_nflist_indices, ctx, queue, nflist_idx_bytes > 0 ? nflist_idx_bytes : 4);
    if (status != CVL_CL_SUCCESS)
        goto cleanup_host;
    status = cl_ensure_buffer(&eval->buf_leaf_indices, ctx, queue, leaf_idx_bytes > 0 ? leaf_idx_bytes : 4);
    if (status != CVL_CL_SUCCESS)
        goto cleanup_host;
    status = cl_ensure_buffer(&eval->buf_child_indices, ctx, queue, child_idx_bytes);
    if (status != CVL_CL_SUCCESS)
        goto cleanup_host;
    status = cl_ensure_buffer(&eval->buf_mp_coeffs, ctx, queue, mp_bytes);
    if (status != CVL_CL_SUCCESS)
        goto cleanup_host;
    if (status != CVL_CL_SUCCESS)
        goto cleanup_host;

    /* Staging buffers for source/target/result real3_t arrays. */
    {
        const bool unified = cvl_cl_compute_unified_memory(eval->compute);
        if (!eval->buf_src_pos.device.mem)
        {
            status =
                cvl_cl_staging_buffer_init(&eval->buf_src_pos, eval->precision, unified, n_sources, eval->allocator);
            if (status != CVL_CL_SUCCESS)
                goto cleanup_host;
        }
        if (!eval->buf_src_val.device.mem)
        {
            status =
                cvl_cl_staging_buffer_init(&eval->buf_src_val, eval->precision, unified, n_sources, eval->allocator);
            if (status != CVL_CL_SUCCESS)
                goto cleanup_host;
        }
        if (!eval->buf_targets.device.mem)
        {
            status =
                cvl_cl_staging_buffer_init(&eval->buf_targets, eval->precision, unified, n_targets, eval->allocator);
            if (status != CVL_CL_SUCCESS)
                goto cleanup_host;
        }
        if (!eval->buf_results.device.mem)
        {
            status =
                cvl_cl_staging_buffer_init(&eval->buf_results, eval->precision, unified, n_targets, eval->allocator);
            if (status != CVL_CL_SUCCESS)
                goto cleanup_host;
        }

        status = cvl_cl_staging_buffer_reserve(&eval->buf_src_pos, ctx, queue, n_sources);
        if (status != CVL_CL_SUCCESS)
            goto cleanup_host;
        status = cvl_cl_staging_buffer_reserve(&eval->buf_src_val, ctx, queue, n_sources);
        if (status != CVL_CL_SUCCESS)
            goto cleanup_host;
        status = cvl_cl_staging_buffer_reserve(&eval->buf_targets, ctx, queue, n_targets);
        if (status != CVL_CL_SUCCESS)
            goto cleanup_host;
        status = cvl_cl_staging_buffer_reserve(&eval->buf_results, ctx, queue, n_targets);
        if (status != CVL_CL_SUCCESS)
            goto cleanup_host;
    }

    /* ---- 3. Upload tree + source data + targets ---- */
    {
        /* Flat tree (raw buffers) - use the precision-matched layout. */
        const void *upload_nodes = is_f32 ? (const void *)flat_nodes_f32 : (const void *)flat_nodes;
        const void *upload_centers = is_f32 ? (const void *)eval_centers_f32 : (const void *)eval_centers;
        const void *upload_local = is_f32 ? (const void *)local_coeffs_f32 : (const void *)tree->local_coeffs;
        const void *upload_mp = is_f32 ? (const void *)mp_coeffs_f32 : (const void *)tree->multipole_coeffs;

        status = cvl_cl_write_buffer(queue, &eval->buf_nodes, 0, nodes_bytes, upload_nodes, 0, NULL, NULL);
        if (status != CVL_CL_SUCCESS)
            goto cleanup_host;
        status = cvl_cl_write_buffer(queue, &eval->buf_eval_centers, 0, centers_bytes, upload_centers, 0, NULL, NULL);
        if (status != CVL_CL_SUCCESS)
            goto cleanup_host;
        status =
            cvl_cl_write_buffer(queue, &eval->buf_particle_order, 0, order_bytes, tree->particle_order, 0, NULL, NULL);
        if (status != CVL_CL_SUCCESS)
            goto cleanup_host;
        status = cvl_cl_write_buffer(queue, &eval->buf_local_coeffs, 0, local_bytes, upload_local, 0, NULL, NULL);
        if (status != CVL_CL_SUCCESS)
            goto cleanup_host;

        /* Explicit child indices for descent. */
        status = cvl_cl_write_buffer(queue, &eval->buf_child_indices, 0, child_idx_bytes, child_indices, 0, NULL, NULL);
        if (status != CVL_CL_SUCCESS)
            goto cleanup_host;

        /* Multipole coefficients (for fallback when target is outside bbox). */
        status = cvl_cl_write_buffer(queue, &eval->buf_mp_coeffs, 0, mp_bytes, upload_mp, 0, NULL, NULL);
        if (status != CVL_CL_SUCCESS)
            goto cleanup_host;

        /* Near-field interaction lists (CSR). */
        if (tree->nflist_offsets && tree->nflist_indices && tree->leaf_indices)
        {
            status = cvl_cl_write_buffer(queue, &eval->buf_nflist_offsets, 0, nflist_off_bytes, tree->nflist_offsets, 0,
                                         NULL, NULL);
            if (status != CVL_CL_SUCCESS)
                goto cleanup_host;
            if (nflist_idx_bytes > 0)
            {
                status = cvl_cl_write_buffer(queue, &eval->buf_nflist_indices, 0, nflist_idx_bytes,
                                             tree->nflist_indices, 0, NULL, NULL);
                if (status != CVL_CL_SUCCESS)
                    goto cleanup_host;
            }
            status = cvl_cl_write_buffer(queue, &eval->buf_leaf_indices, 0, leaf_idx_bytes, tree->leaf_indices, 0, NULL,
                                         NULL);
            if (status != CVL_CL_SUCCESS)
                goto cleanup_host;
        }

        /* Sources + targets (staging buffers, async). */
        cvl_cl_future_t f[3];
        unsigned nf = 0;
        status = cvl_cl_staging_buffer_write_async(&eval->buf_src_pos, queue, sources_coords, n_sources, 0, &f[nf++]);
        if (status != CVL_CL_SUCCESS)
            goto cleanup_host;
        status = cvl_cl_staging_buffer_write_async(&eval->buf_src_val, queue, sources_values, n_sources, 0, &f[nf++]);
        if (status != CVL_CL_SUCCESS)
            goto cleanup_host;
        status = cvl_cl_staging_buffer_write_async(&eval->buf_targets, queue, targets, n_targets, 0, &f[nf++]);
        if (status != CVL_CL_SUCCESS)
            goto cleanup_host;
        for (unsigned i = 0; i < nf; ++i)
        {
            status = cvl_cl_future_wait(&f[i]);
            if (status != CVL_CL_SUCCESS)
                goto cleanup_host;
        }
    }

    /* ---- 4. Launch the fmm_l2p_eval kernel ---- */
    {
        cvl_cl_kernel_t *k = cvl_cl_compute_kernel(eval->compute, "fmm_l2p_eval");
        if (!k)
        {
            status = CVL_CL_ERR_NOT_FOUND;
            goto cleanup_host;
        }

        status = cvl_cl_kernel_set_args(
            k, (cvl_cl_karg_t[]){
                   {.type = CVL_CL_KARG_BUFFER, .index = 0, .mem = eval->buf_nodes.mem},
                   {.type = CVL_CL_KARG_BUFFER, .index = 1, .mem = eval->buf_eval_centers.mem},
                   {.type = CVL_CL_KARG_BUFFER, .index = 2, .mem = eval->buf_particle_order.mem},
                   {.type = CVL_CL_KARG_BUFFER, .index = 3, .mem = eval->buf_local_coeffs.mem},
                   {.type = CVL_CL_KARG_BUFFER, .index = 4, .mem = eval->buf_src_pos.device.mem},
                   {.type = CVL_CL_KARG_BUFFER, .index = 5, .mem = eval->buf_src_val.device.mem},
                   {.type = CVL_CL_KARG_BUFFER, .index = 6, .mem = eval->buf_targets.device.mem},
                   {.type = CVL_CL_KARG_SCALAR_UINT, .index = 7, .scalar_uint = n_targets},
                   {.type = CVL_CL_KARG_SCALAR_UINT, .index = 8, .scalar_uint = n_nodes},
                   {.type = CVL_CL_KARG_SCALAR_UINT, .index = 9, .scalar_uint = max_depth},
                   {.type = CVL_CL_KARG_SCALAR_UINT, .index = 10, .scalar_uint = order},
                   {.type = CVL_CL_KARG_SCALAR_UINT, .index = 11, .scalar_uint = (unsigned)n_coeffs},
                   {.type = CVL_CL_KARG_BUFFER, .index = 12, .mem = eval->buf_nflist_offsets.mem},
                   {.type = CVL_CL_KARG_BUFFER, .index = 13, .mem = eval->buf_nflist_indices.mem},
                   {.type = CVL_CL_KARG_BUFFER, .index = 14, .mem = eval->buf_leaf_indices.mem},
                   {.type = CVL_CL_KARG_BUFFER, .index = 15, .mem = eval->buf_child_indices.mem},
                   {.type = CVL_CL_KARG_BUFFER, .index = 16, .mem = eval->buf_mp_coeffs.mem},
                   {.type = CVL_CL_KARG_BUFFER, .index = 17, .mem = eval->buf_results.device.mem},
                   {},
               });
        if (status != CVL_CL_SUCCESS)
            goto cleanup_host;

        const size_t global = n_targets;
        status = cvl_cl_ndrange(queue, k, 1, &global, NULL, NULL, 0, NULL, NULL);
        if (status != CVL_CL_SUCCESS)
            goto cleanup_host;
    }

    /* ---- 5. Read results back ---- */
    status = cvl_cl_staging_buffer_read_and_wait(&eval->buf_results, queue, results, n_targets, 0);
    if (status != CVL_CL_SUCCESS)
        goto cleanup_host;

    /* Cache dimensions for inspection. */
    eval->n_nodes = n_nodes;
    eval->n_sources = n_sources;
    eval->n_coeffs = (unsigned)n_coeffs;
    eval->order = order;
    eval->max_depth = max_depth;

cleanup_host:
    /* Work buffer is caller-owned - nothing to free here. */
    return status;
}

void cvl_cl_fmm_eval_destroy(cvl_cl_fmm_eval_t *eval)
{
    if (!eval)
        return;

    /* Release staging buffers first (allocated last), then raw buffers
     * in reverse allocation order. */
    cvl_cl_staging_buffer_destroy(&eval->buf_results);
    cvl_cl_staging_buffer_destroy(&eval->buf_targets);
    cvl_cl_staging_buffer_destroy(&eval->buf_src_val);
    cvl_cl_staging_buffer_destroy(&eval->buf_src_pos);
    cvl_cl_buffer_destroy(&eval->buf_mp_coeffs);
    cvl_cl_buffer_destroy(&eval->buf_child_indices);
    cvl_cl_buffer_destroy(&eval->buf_leaf_indices);
    cvl_cl_buffer_destroy(&eval->buf_nflist_indices);
    cvl_cl_buffer_destroy(&eval->buf_nflist_offsets);
    cvl_cl_buffer_destroy(&eval->buf_local_coeffs);
    cvl_cl_buffer_destroy(&eval->buf_particle_order);
    cvl_cl_buffer_destroy(&eval->buf_eval_centers);
    cvl_cl_buffer_destroy(&eval->buf_nodes);

    memset(eval, 0, sizeof(*eval));
}
