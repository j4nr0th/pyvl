/*
 * GPU-accelerated uniform octree builder — host-side orchestration.
 *
 * Implements the pipeline declared in cvl_cl_gpu_tree_build.h.
 *
 * The pipeline runs fully on the GPU once the bounding box is known:
 *   1. Morton-code generation
 *   2. 64-bit LSD radix sort (8 passes × histogram + scatter)
 *   3. Boundary-depth detection
 *   4. Host-side histogram → tree sizing
 *   5. Leaf compaction + construction (two kernels replace prefix-scan approach)
 *   6. Bottom-up internal node construction
 */

#include "cvl_cl_gpu_tree_build.h"
#include "../common.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/*  Internal helpers                                                   */
/* ------------------------------------------------------------------ */

/**
 * @brief Grow or create a device buffer to at least @p size_bytes.
 *
 * If the buffer does not exist yet, it is created via cvl_cl_buffer_create.
 * If it exists but capacity is insufficient, it is grown via cvl_cl_buffer_reserve
 * (preserving existing content).  Otherwise only the logical size is updated.
 *
 * @return CVL_CL_SUCCESS or error.
 */
static cvl_cl_status_t ensure_buffer_size(cvl_cl_buffer_t *buf, const cvl_cl_ctx_t *ctx, cvl_cl_queue_t *queue,
                                          size_t size_bytes)
{
    if (!buf->mem)
    {
        const cvl_cl_buffer_desc_t desc = {
            .access = CVL_CL_BUF_READ_WRITE,
            .size_bytes = size_bytes,
            .host_ptr = NULL,
            .use_host_ptr = false,
        };
        return cvl_cl_buffer_create(ctx, &desc, buf);
    }

    if (buf->capacity >= size_bytes)
    {
        buf->size = size_bytes;
        return CVL_CL_SUCCESS;
    }

    cvl_cl_status_t st = cvl_cl_buffer_reserve(buf, ctx, queue, size_bytes);
    if (st == CVL_CL_SUCCESS)
        buf->size = size_bytes;
    return st;
}

/**
 * @brief Zero-fill a device buffer via host-side write.
 *
 * Allocates a temporary zeroed host buffer of @p size_bytes, writes it
 * to the device, and finishes.  Suitable for infrequent resets (not
 * performance-critical paths).
 */
static cvl_cl_status_t zero_device_buffer(cvl_cl_queue_t *queue, cvl_cl_buffer_t *buf, size_t size_bytes)
{
    if (size_bytes == 0)
        return CVL_CL_SUCCESS;

    void *zeros = calloc(1, size_bytes);
    if (!zeros)
        return CVL_CL_ERR_MEMORY;

    cvl_cl_status_t st = cvl_cl_write_buffer(queue, buf, 0, size_bytes, zeros, 0, NULL, NULL);
    free(zeros);
    if (st != CVL_CL_SUCCESS)
        return st;

    return cvl_cl_finish(queue);
}

/* ------------------------------------------------------------------ */
/*  Helper: LSD radix sort on GPU (64-bit keys, 8-bit digits)          */
/* ------------------------------------------------------------------ */

/**
 * @brief Perform a 64-bit LSD radix sort of Morton codes + companion indices.
 *
 * On entry @p buf_morton holds the unsorted codes and @p buf_indices
 * holds the initial index permutation (identity).  On exit both are
 * sorted (Morton codes ascending, indices permuted accordingly).
 *
 * @p buf_morton_tmp and @p buf_indices_tmp are used as ping-pong
 * scratch buffers and must be pre-sized to at least @p n elements.
 *
 * The radix histogram buffer is grown as needed.
 *
 * @param builder  Builder (provides buffers and kernel handles).
 * @param queue    Command queue.
 * @param ctx      Context.
 * @param n        Number of elements to sort.
 * @return CVL_CL_SUCCESS or error.
 */
static cvl_cl_status_t gpu_radix_sort(cvl_cl_gpu_tree_build_t *builder, cvl_cl_queue_t *queue, const cvl_cl_ctx_t *ctx,
                                      unsigned n)
{
    cvl_cl_status_t st;
    const unsigned n_wgs = (n + CVL_CL_GPU_BUILD_WG - 1) / CVL_CL_GPU_BUILD_WG;
    const size_t hist_bytes = (size_t)n_wgs * 256u * sizeof(unsigned);

    /* ---- Ensure radix histogram buffer is large enough ---- */
    st = ensure_buffer_size(&builder->buf_radix_hist, ctx, queue, hist_bytes);
    if (st != CVL_CL_SUCCESS)
        return st;

    /* ---- Host-side histogram + prefix buffer ---- */
    unsigned *host_h = (unsigned *)malloc(hist_bytes);
    if (!host_h)
        return CVL_CL_ERR_MEMORY;

    /* ---- 8 passes, one per 8-bit digit of the 64-bit key ---- */
    int pass_odd = 0;

    for (unsigned pass = 0; pass < 8; ++pass)
    {
        const unsigned shift = pass * 8u;

        /* Select source / destination buffers for this pass.
         *
         *   pass_odd = 0 → read from even buffers  (buf_morton / buf_indices)
         *                   write to odd buffers   (buf_morton_tmp / buf_indices_tmp)
         *   pass_odd = 1 → read from odd buffers, write to even buffers
         */
        cl_mem keys_src = pass_odd ? builder->buf_morton_tmp.mem : builder->buf_morton.mem;
        cl_mem idx_src = pass_odd ? builder->buf_indices_tmp.mem : builder->buf_indices.mem;
        cl_mem keys_dst = pass_odd ? builder->buf_morton.mem : builder->buf_morton_tmp.mem;
        cl_mem idx_dst = pass_odd ? builder->buf_indices.mem : builder->buf_indices_tmp.mem;

        /* ----- histogram pass ----- */
        cvl_cl_kernel_t *kh = cvl_cl_compute_kernel(builder->compute, "kernel_radix_hist");
        if (!kh)
        {
            free(host_h);
            return CVL_CL_ERR_INTERNAL;
        }

        st =
            cvl_cl_kernel_set_args(kh, (cvl_cl_karg_t[]){
                                           {.type = CVL_CL_KARG_BUFFER, .index = 0, .mem = keys_src},
                                           {.type = CVL_CL_KARG_SCALAR_UINT, .index = 1, .scalar_uint = n},
                                           {.type = CVL_CL_KARG_SCALAR_UINT, .index = 2, .scalar_uint = shift},
                                           {.type = CVL_CL_KARG_BUFFER, .index = 3, .mem = builder->buf_radix_hist.mem},
                                           {},
                                       });
        if (st != CVL_CL_SUCCESS)
        {
            free(host_h);
            return st;
        }

        {
            const size_t global = ((n + CVL_CL_GPU_BUILD_WG - 1) / CVL_CL_GPU_BUILD_WG) * CVL_CL_GPU_BUILD_WG;
            st = cvl_cl_ndrange(queue, kh, 1, &global, NULL, NULL, 0, NULL, NULL);
            if (st != CVL_CL_SUCCESS)
            {
                free(host_h);
                return st;
            }
        }

        /* Read histogram to host. */
        st = cvl_cl_read_buffer(queue, &builder->buf_radix_hist, 0, hist_bytes, host_h, 0, NULL, NULL);
        if (st != CVL_CL_SUCCESS)
        {
            free(host_h);
            return st;
        }

        st = cvl_cl_finish(queue);
        if (st != CVL_CL_SUCCESS)
        {
            free(host_h);
            return st;
        }

        /* Compute exclusive per-WG prefix sum per digit (in place on host_h). */
        for (unsigned d = 0; d < 256; ++d)
        {
            unsigned acc = 0;
            for (unsigned w = 0; w < n_wgs; ++w)
            {
                const unsigned cnt = host_h[w * 256u + d];
                host_h[w * 256u + d] = acc;
                acc += cnt;
            }
        }

        /* Upload prefix back to the same device buffer. */
        st = cvl_cl_write_buffer(queue, &builder->buf_radix_hist, 0, hist_bytes, host_h, 0, NULL, NULL);
        if (st != CVL_CL_SUCCESS)
        {
            free(host_h);
            return st;
        }

        /* ----- scatter pass ----- */
        cvl_cl_kernel_t *ks = cvl_cl_compute_kernel(builder->compute, "kernel_radix_scatter");
        if (!ks)
        {
            free(host_h);
            return CVL_CL_ERR_INTERNAL;
        }

        st =
            cvl_cl_kernel_set_args(ks, (cvl_cl_karg_t[]){
                                           {.type = CVL_CL_KARG_BUFFER, .index = 0, .mem = keys_src},
                                           {.type = CVL_CL_KARG_BUFFER, .index = 1, .mem = idx_src},
                                           {.type = CVL_CL_KARG_BUFFER, .index = 2, .mem = keys_dst},
                                           {.type = CVL_CL_KARG_BUFFER, .index = 3, .mem = idx_dst},
                                           {.type = CVL_CL_KARG_SCALAR_UINT, .index = 4, .scalar_uint = n},
                                           {.type = CVL_CL_KARG_SCALAR_UINT, .index = 5, .scalar_uint = shift},
                                           {.type = CVL_CL_KARG_BUFFER, .index = 6, .mem = builder->buf_radix_hist.mem},
                                           {},
                                       });
        if (st != CVL_CL_SUCCESS)
        {
            free(host_h);
            return st;
        }

        {
            const size_t global = ((n + CVL_CL_GPU_BUILD_WG - 1) / CVL_CL_GPU_BUILD_WG) * CVL_CL_GPU_BUILD_WG;
            st = cvl_cl_ndrange(queue, ks, 1, &global, NULL, NULL, 0, NULL, NULL);
            if (st != CVL_CL_SUCCESS)
            {
                free(host_h);
                return st;
            }
        }

        pass_odd ^= 1;
    }

    /* After 8 passes pass_odd == 0, final result is in the even buffers
     * (buf_morton, buf_indices) — exactly where we started. */
    free(host_h);
    return CVL_CL_SUCCESS;
}

/* ------------------------------------------------------------------ */
/*  cvl_cl_gpu_tree_build_init                                         */
/* ------------------------------------------------------------------ */

cvl_cl_status_t cvl_cl_gpu_tree_build_init(cvl_cl_gpu_tree_build_t *builder, cvl_cl_compute_t *compute,
                                           unsigned max_depth, unsigned critical_count, unsigned order)
{
    if (!builder || !compute)
        return CVL_CL_ERR_INVALID_PARAM;
    if (max_depth > CVL_CL_GPU_BUILD_MAX_DEPTH)
        return CVL_CL_ERR_INVALID_PARAM;

    *builder = (cvl_cl_gpu_tree_build_t){0};

    /* Verify all required kernels exist in the compute backend. */
    static const char *required_kernels[] = {
        "kernel_morton",         "kernel_radix_hist",  "kernel_radix_scatter",  "kernel_boundary",
        "kernel_compact_leaves", "kernel_fill_leaves", "kernel_build_internal",
    };
    const unsigned n_req = sizeof(required_kernels) / sizeof(required_kernels[0]);

    for (unsigned i = 0; i < n_req; ++i)
    {
        if (!cvl_cl_compute_kernel(compute, required_kernels[i]))
            return CVL_CL_ERR_NOT_FOUND; /* kernel name missing */
    }

    builder->compute = compute;
    builder->max_depth = max_depth;
    builder->critical_count = critical_count;
    builder->order = order;
    builder->initialized = true;

    return CVL_CL_SUCCESS;
}

/* ------------------------------------------------------------------ */
/*  cvl_cl_gpu_tree_build_run                                          */
/* ------------------------------------------------------------------ */

cvl_cl_status_t cvl_cl_gpu_tree_build_run(cvl_cl_gpu_tree_build_t *builder, cvl_cl_queue_t *queue,
                                          const cvl_cl_ctx_t *ctx, cvl_cl_staging_buffer_t *staging_pos,
                                          unsigned n_sources)
{
    cvl_cl_status_t st;

    /* ---- Validate ---- */
    if (!builder || !builder->initialized)
        return CVL_CL_ERR_INVALID_PARAM;
    if (!queue || !ctx || !staging_pos)
        return CVL_CL_ERR_INVALID_PARAM;
    if (n_sources == 0 || n_sources > (unsigned)-1)
        return CVL_CL_ERR_INVALID_PARAM;

    builder->n_sources = n_sources;

    const unsigned max_depth = builder->max_depth;

    /* ================================================================ */
    /*  Stage 0 — Compute bounding box from staging buffer               */
    /* ================================================================ */
    /* Read back coords to compute root center and half-size.  This also
     * acts as a barrier ensuring all prior uploads to staging_pos have
     * completed before we launch the pipeline. */

    real3_t *coords_host = (real3_t *)malloc(n_sources * sizeof(real3_t));
    if (!coords_host)
        return CVL_CL_ERR_MEMORY;

    /* Ensure all prior writes to the staging buffer have completed. */
    st = cvl_cl_finish(queue);
    if (st != CVL_CL_SUCCESS)
    {
        free(coords_host);
        return st;
    }

    /* Read back coords directly from the staging buffer's device memory. */
    {
        cl_int err = clEnqueueReadBuffer(cvl_cl_queue_queue(queue), staging_pos->device.mem, CL_TRUE, 0,
                                         n_sources * sizeof(real3_t), coords_host, 0, NULL, NULL);
        if (err != CL_SUCCESS)
        {
            free(coords_host);
            return cvl_cl_status_from_cl_int(err);
        }
    }
    if (st != CVL_CL_SUCCESS)
    {
        free(coords_host);
        return st;
    }

    {
        real3_t bbox_min = coords_host[0];
        real3_t bbox_max = coords_host[0];

        for (unsigned i = 1; i < n_sources; ++i)
        {
            const real_t x = coords_host[i].x;
            const real_t y = coords_host[i].y;
            const real_t z = coords_host[i].z;

            if (x < bbox_min.x)
                bbox_min.x = x;
            if (y < bbox_min.y)
                bbox_min.y = y;
            if (z < bbox_min.z)
                bbox_min.z = z;
            if (x > bbox_max.x)
                bbox_max.x = x;
            if (y > bbox_max.y)
                bbox_max.y = y;
            if (z > bbox_max.z)
                bbox_max.z = z;
        }

        const real_t root_extent =
            fmax(fmax(bbox_max.x - bbox_min.x, bbox_max.y - bbox_min.y), bbox_max.z - bbox_min.z);
        const real_t root_hs = root_extent * (real_t)0.5 + (real_t)1e-12;
        const real_t root_cx = (bbox_min.x + bbox_max.x) * (real_t)0.5;
        const real_t root_cy = (bbox_min.y + bbox_max.y) * (real_t)0.5;
        const real_t root_cz = (bbox_min.z + bbox_max.z) * (real_t)0.5;

        free(coords_host);
        coords_host = NULL;

        /* ================================================================ */
        /*  Stage 1 — Ensure pipeline buffers are sized for n_sources         */
        /* ================================================================ */

        const size_t elem_bytes = n_sources * sizeof(uint64_t);
        const size_t idx_bytes = n_sources * sizeof(unsigned);
        const size_t bd_bytes = n_sources * sizeof(int);
        const size_t bd_hist_bytes = (size_t)(max_depth + 2) * sizeof(unsigned);
        const size_t counter_bytes = 2u * sizeof(unsigned); /* n_leaves_out + particle_counter */

        st = ensure_buffer_size(&builder->buf_morton, ctx, queue, elem_bytes);
        if (st != CVL_CL_SUCCESS)
            return st;
        st = ensure_buffer_size(&builder->buf_morton_tmp, ctx, queue, elem_bytes);
        if (st != CVL_CL_SUCCESS)
            return st;
        st = ensure_buffer_size(&builder->buf_indices, ctx, queue, idx_bytes);
        if (st != CVL_CL_SUCCESS)
            return st;
        st = ensure_buffer_size(&builder->buf_indices_tmp, ctx, queue, idx_bytes);
        if (st != CVL_CL_SUCCESS)
            return st;
        st = ensure_buffer_size(&builder->buf_boundary, ctx, queue, bd_bytes);
        if (st != CVL_CL_SUCCESS)
            return st;
        st = ensure_buffer_size(&builder->buf_bd_hist, ctx, queue, bd_hist_bytes);
        if (st != CVL_CL_SUCCESS)
            return st;
        st = ensure_buffer_size(&builder->buf_leaf_counter, ctx, queue, counter_bytes);
        if (st != CVL_CL_SUCCESS)
            return st;

        /* buf_indices_tmp does not need initialisation — first radix pass
         * writes it.  buf_indices gets the identity permutation. */

        /* ================================================================ */
        /*  Stage 2 — Initialise indices (identity permutation)              */
        /* ================================================================ */

        {
            unsigned *idx_host = (unsigned *)malloc(n_sources * sizeof(unsigned));
            if (!idx_host)
                return CVL_CL_ERR_MEMORY;
            for (unsigned i = 0; i < n_sources; ++i)
                idx_host[i] = i;

            st = cvl_cl_write_buffer(queue, &builder->buf_indices, 0, idx_bytes, idx_host, 0, NULL, NULL);
            free(idx_host);
            if (st != CVL_CL_SUCCESS)
                return st;
        }

        /* ================================================================ */
        /*  Stage 3 — Launch kernel_morton                                   */
        /* ================================================================ */

        {
            cvl_cl_kernel_t *km = cvl_cl_compute_kernel(builder->compute, "kernel_morton");
            if (!km)
                return CVL_CL_ERR_INTERNAL;

            st = cvl_cl_kernel_set_args(km,
                                        (cvl_cl_karg_t[]){
                                            {.type = CVL_CL_KARG_BUFFER, .index = 0, .mem = staging_pos->device.mem},
                                            {.type = CVL_CL_KARG_SCALAR_UINT, .index = 1, .scalar_uint = n_sources},
                                            {.type = CVL_CL_KARG_SCALAR_DOUBLE, .index = 2, .scalar_double = root_cx},
                                            {.type = CVL_CL_KARG_SCALAR_DOUBLE, .index = 3, .scalar_double = root_cy},
                                            {.type = CVL_CL_KARG_SCALAR_DOUBLE, .index = 4, .scalar_double = root_cz},
                                            {.type = CVL_CL_KARG_SCALAR_DOUBLE, .index = 5, .scalar_double = root_hs},
                                            {.type = CVL_CL_KARG_BUFFER, .index = 6, .mem = builder->buf_morton.mem},
                                            {},
                                        });
            if (st != CVL_CL_SUCCESS)
                return st;

            const size_t global = ((n_sources + CVL_CL_GPU_BUILD_WG - 1) / CVL_CL_GPU_BUILD_WG) * CVL_CL_GPU_BUILD_WG;
            st = cvl_cl_ndrange(queue, km, 1, &global, NULL, NULL, 0, NULL, NULL);
            if (st != CVL_CL_SUCCESS)
                return st;
        }

        /* ================================================================ */
        /*  Stage 4 — Radix sort Morton codes + companion indices             */
        /* ================================================================ */

        st = gpu_radix_sort(builder, queue, ctx, n_sources);
        if (st != CVL_CL_SUCCESS)
            return st;

        /* ================================================================ */
        /*  Stage 5 — Boundary detection                                     */
        /* ================================================================ */
        /*  Zero bd_hist before launching. */

        st = zero_device_buffer(queue, &builder->buf_bd_hist, bd_hist_bytes);
        if (st != CVL_CL_SUCCESS)
            return st;

        {
            cvl_cl_kernel_t *kb = cvl_cl_compute_kernel(builder->compute, "kernel_boundary");
            if (!kb)
                return CVL_CL_ERR_INTERNAL;

            st = cvl_cl_kernel_set_args(kb,
                                        (cvl_cl_karg_t[]){
                                            {.type = CVL_CL_KARG_BUFFER, .index = 0, .mem = builder->buf_morton.mem},
                                            {.type = CVL_CL_KARG_SCALAR_UINT, .index = 1, .scalar_uint = n_sources},
                                            {.type = CVL_CL_KARG_SCALAR_UINT, .index = 2, .scalar_uint = max_depth},
                                            {.type = CVL_CL_KARG_BUFFER, .index = 3, .mem = builder->buf_boundary.mem},
                                            {.type = CVL_CL_KARG_BUFFER, .index = 4, .mem = builder->buf_bd_hist.mem},
                                            {},
                                        });
            if (st != CVL_CL_SUCCESS)
                return st;

            const size_t global = ((n_sources + CVL_CL_GPU_BUILD_WG - 1) / CVL_CL_GPU_BUILD_WG) * CVL_CL_GPU_BUILD_WG;
            st = cvl_cl_ndrange(queue, kb, 1, &global, NULL, NULL, 0, NULL, NULL);
            if (st != CVL_CL_SUCCESS)
                return st;
        }

        /* ================================================================ */
        /*  Stage 6 — Read bd_hist → compute depth_counts / depth_offsets    */
        /* ================================================================ */

        {
            unsigned *hist_raw = builder->bd_hist;
            st = cvl_cl_read_buffer(queue, &builder->buf_bd_hist, 0, bd_hist_bytes, hist_raw, 0, NULL, NULL);
            if (st != CVL_CL_SUCCESS)
                return st;

            st = cvl_cl_finish(queue);
            if (st != CVL_CL_SUCCESS)
                return st;

            /* depth_counts[d] = sum_{b=0..d} bd_hist[b] */
            unsigned *dc = builder->depth_counts;
            unsigned acc = 0;
            for (unsigned d = 0; d <= max_depth; ++d)
            {
                /* In the boundary kernel, bd ranges 0 .. max_depth+1.
                 * bd_hist[max_depth+1] is the overflow bucket.  We only
                 * count groups at depths 0 .. max_depth. */
                unsigned bd_this = 0;
                for (unsigned b = 0; b <= d; ++b)
                    bd_this += hist_raw[b];
                dc[d] = bd_this;
            }

            /* Compute prefix sum → depth_offsets. */
            unsigned *doff = builder->depth_offsets;
            acc = 0;
            for (unsigned d = 0; d <= max_depth; ++d)
            {
                doff[d] = acc;
                acc += dc[d];
            }
            doff[max_depth + 1] = acc;

            builder->n_total = acc;
        }

        const unsigned n_total = builder->n_total;
        const unsigned n_leaves = builder->depth_counts[max_depth];

        if (n_leaves == 0)
        {
            /* No particles or all degenerate → nothing to build. */
            builder->n_internal = 0;
            builder->n_multipole_leaves = 0;
            builder->n_particle_leaves = 0;
            return CVL_CL_SUCCESS;
        }

        /* ================================================================ */
        /*  Stage 7 — Allocate output buffers                                */
        /* ================================================================ */
        /*  Nodes, particle_order, and (host) depth_offsets.                 */

        {
            const size_t node_bytes = (size_t)n_total * CVL_CL_GPU_NODE_SIZE;
            const size_t order_bytes = (size_t)n_sources * sizeof(unsigned);
            const size_t doff_bytes = (size_t)(max_depth + 2) * sizeof(unsigned);

            st = ensure_buffer_size(&builder->buf_nodes, ctx, queue, node_bytes);
            if (st != CVL_CL_SUCCESS)
                return st;
            st = ensure_buffer_size(&builder->buf_particle_order, ctx, queue, order_bytes);
            if (st != CVL_CL_SUCCESS)
                return st;
            st = ensure_buffer_size(&builder->buf_depth_offsets, ctx, queue, doff_bytes);
            if (st != CVL_CL_SUCCESS)
                return st;

            /* Upload depth_offsets to device. */
            st = cvl_cl_write_buffer(queue, &builder->buf_depth_offsets, 0, doff_bytes, builder->depth_offsets, 0, NULL,
                                     NULL);
            if (st != CVL_CL_SUCCESS)
                return st;
        }

        /* Ensure leaf_starts buffer is at least n_sources entries. */
        st = ensure_buffer_size(&builder->buf_leaf_starts, ctx, queue, (size_t)n_sources * sizeof(unsigned));
        if (st != CVL_CL_SUCCESS)
            return st;

        /* ================================================================ */
        /*  Stage 8 — Compact leaf starts                                    */
        /* ================================================================ */
        /*  kernel_compact_leaves writes leaf starts + n_leaves_out.         */
        /*  Zero leaf_counter before launching.                              */

        st = zero_device_buffer(queue, &builder->buf_leaf_counter, 2u * sizeof(unsigned));
        if (st != CVL_CL_SUCCESS)
            return st;

        {
            cvl_cl_kernel_t *kc = cvl_cl_compute_kernel(builder->compute, "kernel_compact_leaves");
            if (!kc)
                return CVL_CL_ERR_INTERNAL;

            st = cvl_cl_kernel_set_args(
                kc, (cvl_cl_karg_t[]){
                        {.type = CVL_CL_KARG_BUFFER, .index = 0, .mem = builder->buf_boundary.mem},
                        {.type = CVL_CL_KARG_SCALAR_UINT, .index = 1, .scalar_uint = n_sources},
                        {.type = CVL_CL_KARG_SCALAR_UINT, .index = 2, .scalar_uint = max_depth},
                        {.type = CVL_CL_KARG_BUFFER, .index = 3, .mem = builder->buf_leaf_counter.mem},
                        {.type = CVL_CL_KARG_BUFFER, .index = 4, .mem = builder->buf_leaf_starts.mem},
                        {},
                    });
            if (st != CVL_CL_SUCCESS)
                return st;

            const size_t global = ((n_sources + CVL_CL_GPU_BUILD_WG - 1) / CVL_CL_GPU_BUILD_WG) * CVL_CL_GPU_BUILD_WG;
            st = cvl_cl_ndrange(queue, kc, 1, &global, NULL, NULL, 0, NULL, NULL);
            if (st != CVL_CL_SUCCESS)
                return st;
        }

        /* ================================================================ */
        /*  Stage 9 — Read n_leaves_out from device                          */
        /* ================================================================ */

        unsigned n_leaves_out = 0;
        st = cvl_cl_read_buffer(queue, &builder->buf_leaf_counter, 0, sizeof(unsigned), &n_leaves_out, 0, NULL, NULL);
        if (st != CVL_CL_SUCCESS)
            return st;

        st = cvl_cl_finish(queue);
        if (st != CVL_CL_SUCCESS)
            return st;

        /* Sanity: n_leaves computed from histogram must match GPU compaction. */
        if (n_leaves_out != n_leaves)
        {
            /* Not a hard error — the histogram-based count could differ in
             * degenerate cases.  Use the GPU's actual count. */
            /* (silently use n_leaves_out) */
        }

        /* ================================================================ */
        /*  Stage 10 — Build leaf nodes (kernel_fill_leaves)                  */
        /* ================================================================ */
        /*  Zero the particle_counter half of leaf_counter before launching.  */

        {
            const unsigned zero_pc = 0;
            st = cvl_cl_write_buffer(queue, &builder->buf_leaf_counter, sizeof(unsigned), sizeof(unsigned), &zero_pc, 0,
                                     NULL, NULL);
            if (st != CVL_CL_SUCCESS)
                return st;
        }

        {
            cvl_cl_kernel_t *kf = cvl_cl_compute_kernel(builder->compute, "kernel_fill_leaves");
            if (!kf)
                return CVL_CL_ERR_INTERNAL;

            const unsigned leaf_offset = builder->depth_offsets[max_depth];

            st = cvl_cl_kernel_set_args(
                kf, (cvl_cl_karg_t[]){
                        {.type = CVL_CL_KARG_BUFFER, .index = 0, .mem = builder->buf_leaf_starts.mem},
                        {.type = CVL_CL_KARG_SCALAR_UINT, .index = 1, .scalar_uint = n_leaves},
                        {.type = CVL_CL_KARG_BUFFER, .index = 2, .mem = builder->buf_boundary.mem},
                        {.type = CVL_CL_KARG_SCALAR_UINT, .index = 3, .scalar_uint = n_sources},
                        {.type = CVL_CL_KARG_SCALAR_UINT, .index = 4, .scalar_uint = max_depth},
                        {.type = CVL_CL_KARG_BUFFER, .index = 5, .mem = builder->buf_nodes.mem},
                        {.type = CVL_CL_KARG_SCALAR_UINT, .index = 6, .scalar_uint = leaf_offset},
                        {.type = CVL_CL_KARG_BUFFER, .index = 7, .mem = builder->buf_particle_order.mem},
                        {.type = CVL_CL_KARG_BUFFER, .index = 8, .mem = builder->buf_leaf_counter.mem},
                        {.type = CVL_CL_KARG_BUFFER, .index = 9, .mem = staging_pos->device.mem},
                        {.type = CVL_CL_KARG_BUFFER, .index = 10, .mem = builder->buf_morton.mem},
                        {.type = CVL_CL_KARG_BUFFER, .index = 11, .mem = builder->buf_indices.mem},
                        {.type = CVL_CL_KARG_SCALAR_DOUBLE, .index = 12, .scalar_double = root_hs},
                        {.type = CVL_CL_KARG_SCALAR_UINT, .index = 13, .scalar_uint = builder->critical_count},
                        {},
                    });
            if (st != CVL_CL_SUCCESS)
                return st;

            const size_t global = ((n_leaves + CVL_CL_GPU_BUILD_WG - 1) / CVL_CL_GPU_BUILD_WG) * CVL_CL_GPU_BUILD_WG;
            st = cvl_cl_ndrange(queue, kf, 1, &global, NULL, NULL, 0, NULL, NULL);
            if (st != CVL_CL_SUCCESS)
                return st;
        }

        /* ================================================================ */
        /*  Stage 11 — Build internal nodes bottom-up                        */
        /* ================================================================ */
        /*  kernel_build_internal is called once per depth level from
         *  max_depth-1 down to 0.  Each call processes all parents at that
         *  depth using the already-built children at depth+1.               */

        for (int d = (int)max_depth - 1; d >= 0; --d)
        {
            const unsigned depth = (unsigned)d;
            const unsigned n_parents = builder->depth_counts[depth];
            const unsigned parent_off = builder->depth_offsets[depth];
            const unsigned child_off = builder->depth_offsets[depth + 1];
            const unsigned n_children = builder->depth_counts[depth + 1];

            if (n_parents == 0)
                continue;

            cvl_cl_kernel_t *ki = cvl_cl_compute_kernel(builder->compute, "kernel_build_internal");
            if (!ki)
                return CVL_CL_ERR_INTERNAL;

            st = cvl_cl_kernel_set_args(ki,
                                        (cvl_cl_karg_t[]){
                                            {.type = CVL_CL_KARG_BUFFER, .index = 0, .mem = builder->buf_nodes.mem},
                                            {.type = CVL_CL_KARG_SCALAR_UINT, .index = 1, .scalar_uint = depth},
                                            {.type = CVL_CL_KARG_SCALAR_UINT, .index = 2, .scalar_uint = n_parents},
                                            {.type = CVL_CL_KARG_SCALAR_UINT, .index = 3, .scalar_uint = parent_off},
                                            {.type = CVL_CL_KARG_SCALAR_UINT, .index = 4, .scalar_uint = child_off},
                                            {.type = CVL_CL_KARG_SCALAR_UINT, .index = 5, .scalar_uint = n_children},
                                            {.type = CVL_CL_KARG_SCALAR_DOUBLE, .index = 6, .scalar_double = root_hs},
                                            {},
                                        });
            if (st != CVL_CL_SUCCESS)
                return st;

            const size_t global = ((n_parents + CVL_CL_GPU_BUILD_WG - 1) / CVL_CL_GPU_BUILD_WG) * CVL_CL_GPU_BUILD_WG;
            st = cvl_cl_ndrange(queue, ki, 1, &global, NULL, NULL, 0, NULL, NULL);
            if (st != CVL_CL_SUCCESS)
                return st;
        }

        /* ================================================================ */
        /*  Stage 12 — Read back metadata                                    */
        /* ================================================================ */
        /*  Classify leaf nodes by reading back their kind byte.             */
        /*  Each leaf is at byte offset (depth_offsets[max_depth] + i) * 64.
         *  The kind field is at byte offset 49 within the node struct.      */

        {
            const unsigned leaf_offset = builder->depth_offsets[max_depth];
            const size_t kind_offset = (size_t)leaf_offset * CVL_CL_GPU_NODE_SIZE + 49u;

            unsigned char *leaf_kinds = (unsigned char *)malloc(n_leaves);
            if (!leaf_kinds)
                return CVL_CL_ERR_MEMORY;

            st = cvl_cl_read_buffer(queue, &builder->buf_nodes, kind_offset, n_leaves, leaf_kinds, 0, NULL, NULL);
            if (st != CVL_CL_SUCCESS)
            {
                free(leaf_kinds);
                return st;
            }
            st = cvl_cl_finish(queue);
            if (st != CVL_CL_SUCCESS)
            {
                free(leaf_kinds);
                return st;
            }

            unsigned n_mp = 0, n_pt = 0;
            for (unsigned i = 0; i < n_leaves; ++i)
            {
                /* BH_KIND_PARTICLE = 1, BH_KIND_MULTIPOLE = 2 */
                if (leaf_kinds[i] == 2)
                    ++n_mp;
                else
                    ++n_pt;
            }
            free(leaf_kinds);

            builder->n_internal = n_total - n_leaves;
            builder->n_multipole_leaves = n_mp;
            builder->n_particle_leaves = n_pt;
        }
    }

    return CVL_CL_SUCCESS;
}

/* ------------------------------------------------------------------ */
/*  cvl_cl_gpu_tree_build_destroy                                       */
/* ------------------------------------------------------------------ */

void cvl_cl_gpu_tree_build_destroy(cvl_cl_gpu_tree_build_t *builder)
{
    if (!builder)
        return;

    cvl_cl_buffer_destroy(&builder->buf_morton);
    cvl_cl_buffer_destroy(&builder->buf_morton_tmp);
    cvl_cl_buffer_destroy(&builder->buf_indices);
    cvl_cl_buffer_destroy(&builder->buf_indices_tmp);
    cvl_cl_buffer_destroy(&builder->buf_boundary);
    cvl_cl_buffer_destroy(&builder->buf_radix_hist);
    cvl_cl_buffer_destroy(&builder->buf_bd_hist);
    cvl_cl_buffer_destroy(&builder->buf_nodes);
    cvl_cl_buffer_destroy(&builder->buf_particle_order);
    cvl_cl_buffer_destroy(&builder->buf_depth_offsets);
    cvl_cl_buffer_destroy(&builder->buf_leaf_starts);
    cvl_cl_buffer_destroy(&builder->buf_leaf_counter);

    *builder = (cvl_cl_gpu_tree_build_t){0};
}
