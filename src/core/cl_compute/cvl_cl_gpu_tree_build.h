#pragma once
/*
 * GPU-accelerated uniform octree builder.
 *
 * Orchestrates the full tree build pipeline on the GPU:
 *   Morton codes → LSD radix sort → boundary detection → leaf build → internal build
 *
 * The result is a flat node array (+ particle_order + depth_offsets) directly
 * in device memory, ready for use by bh_flat_eval or similar kernels.
 *
 * Usage:
 *   cvl_cl_gpu_tree_build_t builder;
 *   cvl_cl_gpu_tree_build_init(&builder, &comp, max_depth, critical_count, order);
 *
 *   cvl_cl_gpu_tree_build_run(&builder, &queue, &ctx, &staging_pos, n_sources);
 *
 *   // builder.nodes, builder.particle_order, builder.depth_offsets are ready
 *   // builder.n_total, builder.n_internal etc. are populated
 *
 *   cvl_cl_gpu_tree_build_destroy(&builder);
 */

#include "../opencl/cvl_cl_buffer.h"
#include "../opencl/cvl_cl_command.h"
#include "../opencl/cvl_cl_common.h"
#include "../opencl/cvl_cl_ctx.h"
#include "../opencl/cvl_cl_kernel.h"
#include "cvl_cl_compute.h"
#include "cvl_cl_staging_buffer.h"

/** @brief Maximum tree depth for the GPU build (limited by Morton code to 21). */
enum
{
    CVL_CL_GPU_BUILD_MAX_DEPTH = 21
};

/** @brief Work-group size for radix sort and auxiliary kernels. */
enum
{
    CVL_CL_GPU_BUILD_WG = 256
};

/** @brief Node struct byte size on device (64 bytes, matches bh_build_node_t / cvl_cl_flat_node_t). */
enum
{
    CVL_CL_GPU_NODE_SIZE = 64
};

typedef struct
{
    /* Settings. */
    unsigned max_depth;
    unsigned critical_count;
    unsigned order;

    /* Compute backend (borrowed — kernels live here). */
    cvl_cl_compute_t *compute;

    /* Owned device buffers for the pipeline.
     * Sizes are determined at run time and grown on demand. */
    cvl_cl_buffer_t buf_morton;         /**< [n] Morton codes (uint64_t). */
    cvl_cl_buffer_t buf_morton_tmp;     /**< [n] temp for radix sort ping-pong. */
    cvl_cl_buffer_t buf_indices;        /**< [n] particle index permutation. */
    cvl_cl_buffer_t buf_indices_tmp;    /**< [n] temp for radix sort ping-pong. */
    cvl_cl_buffer_t buf_boundary;       /**< [n] boundary depth (int). */
    cvl_cl_buffer_t buf_radix_hist;     /**< [n_wgs * 256] per-pass histogram + prefix (unsigned). */
    cvl_cl_buffer_t buf_bd_hist;        /**< [max_depth+2] boundary depth histogram (unsigned). */
    cvl_cl_buffer_t buf_nodes;          /**< [n_total] flat node array (CVL_CL_GPU_NODE_SIZE each). */
    cvl_cl_buffer_t buf_particle_order; /**< [n] particle order (unsigned). */
    cvl_cl_buffer_t buf_depth_offsets;  /**< [max_depth+2] depth offsets (unsigned). */
    cvl_cl_buffer_t buf_leaf_starts;    /**< [n] leaf start positions (unsigned) — temp. */
    cvl_cl_buffer_t buf_leaf_counter;   /**< 2 × unsigned: [n_leaves_out, particle_counter]. */

    /* Host-side copies of metadata (read back from GPU after run). */
    unsigned bd_hist[CVL_CL_GPU_BUILD_MAX_DEPTH + 2];
    unsigned depth_counts[CVL_CL_GPU_BUILD_MAX_DEPTH + 1];
    unsigned depth_offsets[CVL_CL_GPU_BUILD_MAX_DEPTH + 2];
    unsigned n_total;
    unsigned n_internal;
    unsigned n_multipole_leaves;
    unsigned n_particle_leaves;
    unsigned n_sources;

    /* Valid flag. */
    bool initialized;
} cvl_cl_gpu_tree_build_t;

/**
 * @brief Initialise the GPU tree builder.
 *
 * Validates settings and verifies that all required kernels exist in the
 * compute backend.  Does not allocate device buffers (that happens on
 * the first run when @p n_sources is known).
 *
 * Required kernels (must be registered in @p compute):
 *   kernel_morton, kernel_radix_hist, kernel_radix_scatter,
 *   kernel_boundary, kernel_compact_leaves, kernel_fill_leaves,
 *   kernel_build_internal
 *
 * @param builder        Uninitialised builder.
 * @param compute        Compute backend (must have all bh_build kernels registered).
 * @param max_depth      Maximum octree depth (≤ CVL_CL_GPU_BUILD_MAX_DEPTH).
 * @param critical_count Subdivision threshold (particles above this → multipole leaf).
 * @param order          Multipole expansion order.
 * @return CVL_CL_SUCCESS or error.
 */
cvl_cl_status_t cvl_cl_gpu_tree_build_init(cvl_cl_gpu_tree_build_t *builder, cvl_cl_compute_t *compute,
                                           unsigned max_depth, unsigned critical_count, unsigned order);

/**
 * @brief Run the full GPU tree build pipeline.
 *
 * Source coordinates must already be uploaded to @p staging_pos.
 * The pipeline is:
 *   1. Read back coords → compute root bounding box
 *   2. Launch kernel_morton
 *   3. LSD radix sort (8 passes of kernel_radix_hist + kernel_radix_scatter)
 *   4. Launch kernel_boundary
 *   5. Read bd_hist → compute depth_counts / depth_offsets / n_total
 *   6. Allocate output buffers (nodes, particle_order, depth_offsets)
 *   7. Launch kernel_compact_leaves → read n_leaves
 *   8. Launch kernel_fill_leaves
 *   9. Loop kernel_build_internal from max_depth-1 down to 0
 *
 * After success, metadata fields (n_total, n_internal, n_multipole_leaves,
 * n_particle_leaves, n_sources) are populated on the builder.
 *
 * @param builder    Initialised builder.
 * @param queue      Queue for all commands.
 * @param ctx        Context.
 * @param staging_pos Source positions in a staging buffer (must be reserved to n_sources).
 * @param n_sources  Number of particles.
 * @return CVL_CL_SUCCESS or error.
 */
cvl_cl_status_t cvl_cl_gpu_tree_build_run(cvl_cl_gpu_tree_build_t *builder, cvl_cl_queue_t *queue,
                                          const cvl_cl_ctx_t *ctx, cvl_cl_staging_buffer_t *staging_pos,
                                          unsigned n_sources);

/**
 * @brief Destroy the GPU tree builder, releasing all device buffers.
 *
 * Safe to call on a zero-initialised or partially-initialised builder.
 *
 * @param builder Builder to destroy (may be NULL).
 */
void cvl_cl_gpu_tree_build_destroy(cvl_cl_gpu_tree_build_t *builder);
