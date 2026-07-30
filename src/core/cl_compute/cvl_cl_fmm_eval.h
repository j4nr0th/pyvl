#pragma once
/*
 * GPU-accelerated FMM evaluation (L2P) — host API.
 *
 * Phase 5 — offload the final FMM evaluation step to the GPU.
 *
 * The FMM tree is built on the CPU (fmm_tree_build), which performs the
 * full upward sweep (P2M + M2M) and downward sweep (M2L + L2L) and stores
 * the resulting local expansion coefficients in fmm_tree_t::local_coeffs.
 * This module uploads the tree (converted to a flat, GPU-friendly layout)
 * plus the source particle data and target points to the device, launches
 * the fmm_l2p_eval kernel, and reads back the induced field at every
 * target.
 *
 * The kernel does, per target:
 *   1. Descend the flat octree to the leaf containing the target.
 *   2. Evaluate the leaf's precomputed local expansion (L2P).
 *   3. Add the near-field direct sum over the leaf's own particles.
 *
 * This is the GPU analogue of fmm_tree_eval(..., FMM_EVAL_FMM) for points
 * inside the source bounding box.  As with the CPU FMM mode, the local
 * expansion diverges for targets outside the source bounding box —
 * callers must ensure targets lie well inside the domain.
 *
 * Usage:
 * @code
 *   cvl_cl_fmm_eval_t eval;
 *   cvl_cl_fmm_eval_init(&eval, &comp, CVL_CL_PRECISION_FP64);
 *
 *   cvl_cl_fmm_eval_run(&eval, &queue, &ctx, &tree,
 *                       n_targets, targets, results);
 *
 *   cvl_cl_fmm_eval_destroy(&eval);
 * @endcode
 *
 * The compute backend (comp) must have the "fmm_l2p_eval" kernel
 * registered.  The context, queue, and compute backend are borrowed —
 * they must outlive the eval handle.
 */

#include "../opencl/cvl_cl_buffer.h"
#include "../opencl/cvl_cl_command.h"
#include "../opencl/cvl_cl_common.h"
#include "../opencl/cvl_cl_ctx.h"
#include "../opencl/cvl_cl_kernel.h"
#include "cvl_cl_compute.h"
#include "cvl_cl_staging_buffer.h"

#include "../fmm_tree.h"

/* ------------------------------------------------------------------ */
/* Handle                                                             */
/* ------------------------------------------------------------------ */

/**
 * @brief GPU FMM evaluation state.
 *
 * Owns device buffers for the flat tree, source data, targets, and
 * results.  Buffers are grown on demand via cvl_cl_buffer_reserve and
 * reused across runs (grow-only, never shrinks).
 *
 * The compute backend, context, and queue are borrowed — the caller
 * must keep them alive for the lifetime of this handle.
 */
typedef struct
{
    /* Borrowed compute backend (kernel registry + cached caps). */
    cvl_cl_compute_t *compute;
    cvl_cl_precision_t precision;

    /* Device buffers for the flat tree (raw cvl_cl_buffer_t — not real3_t). */
    cvl_cl_buffer_t buf_nodes;          /**< [n_nodes] flat nodes (64 B each). */
    cvl_cl_buffer_t buf_eval_centers;   /**< [3 * n_nodes] Γ-weighted centroids. */
    cvl_cl_buffer_t buf_particle_order; /**< [n_sources] particle indices. */
    cvl_cl_buffer_t buf_local_coeffs;   /**< [3 * n_coeffs * n_nodes] doubles. */
    cvl_cl_buffer_t buf_nflist_offsets; /**< [n_leaves + 1] CSR offsets for near-field. */
    cvl_cl_buffer_t buf_nflist_indices; /**< [nflist_count] CSR near-field leaf IDs. */
    cvl_cl_buffer_t buf_leaf_indices;   /**< [n_leaves] leaf_id → node index map. */
    cvl_cl_buffer_t buf_child_indices;  /**< [8 * n_nodes] explicit child indices (-1 = none). */
    cvl_cl_buffer_t buf_mp_coeffs;      /**< [3 * n_coeffs * n_nodes] multipole coefficients (for fallback). */

    /* Staging buffers for source/target/result real3_t arrays. */
    cvl_cl_staging_buffer_t buf_src_pos; /**< [n_sources] source positions. */
    cvl_cl_staging_buffer_t buf_src_val; /**< [n_sources] source strengths. */
    cvl_cl_staging_buffer_t buf_targets; /**< [n_targets] target positions. */
    cvl_cl_staging_buffer_t buf_results; /**< [n_targets] output field. */

    /* Cached tree dimensions (from the last run). */
    unsigned n_nodes;   /**< Total flat nodes. */
    unsigned n_sources; /**< Total source particles. */
    unsigned n_coeffs;  /**< Coefficients per component per node. */
    unsigned order;     /**< Local expansion order. */
    unsigned max_depth; /**< Maximum tree depth. */

    /* Valid flag. */
    bool initialized;
} cvl_cl_fmm_eval_t;

/* ------------------------------------------------------------------ */
/* Lifecycle                                                          */
/* ------------------------------------------------------------------ */

/**
 * @brief Initialise the GPU FMM evaluator.
 *
 * Validates that the compute backend has the "fmm_l2p_eval" kernel
 * registered.  Does not allocate device buffers — that happens on the
 * first run when the tree size is known.
 *
 * @param eval       Uninitialised evaluator.
 * @param compute    Compute backend (borrowed; must have "fmm_l2p_eval").
 * @param precision  FP32 or FP64 (should match the compute backend).
 * @return CVL_CL_SUCCESS or error.
 */
cvl_cl_status_t cvl_cl_fmm_eval_init(cvl_cl_fmm_eval_t *eval, cvl_cl_compute_t *compute, cvl_cl_precision_t precision);

/**
 * @brief Run the GPU FMM evaluation.
 *
 * Converts the CPU-built @p tree to a flat layout, uploads the tree +
 * source data + targets to the device, launches the fmm_l2p_eval
 * kernel, and reads back the induced field at every target.
 *
 * The tree's local expansion coefficients (local_coeffs) must already
 * be populated — i.e. the tree must have been built with
 * fmm_tree_build (which runs M2L + L2L).  Targets should lie well
 * inside the source bounding box (FMM mode diverges outside).
 *
 * @param eval      Initialised evaluator.
 * @param queue     Command queue (borrowed).
 * @param ctx       Context (borrowed).
 * @param tree      CPU-built FMM tree with local expansions populated.
 * @param sources_coords  Source positions [tree->n_sources] (must match build).
 * @param sources_values  Source strengths [tree->n_sources] (must match build).
 * @param n_targets Number of target points.
 * @param targets   Target positions [n_targets].
 * @param results   Output field [n_targets] (caller-allocated).
 * @return CVL_CL_SUCCESS or error.
 */
cvl_cl_status_t cvl_cl_fmm_eval_run(cvl_cl_fmm_eval_t *eval, cvl_cl_queue_t *queue, const cvl_cl_ctx_t *ctx,
                                    const fmm_tree_t *tree, const real3_t *sources_coords,
                                    const real3_t *sources_values, unsigned n_targets, const real3_t *targets,
                                    real3_t *results);

/**
 * @brief Release all device buffers owned by the evaluator.
 *
 * Safe to call on a zero-initialised handle.  Does not release the
 * borrowed compute backend, context, or queue.
 *
 * @param eval Evaluator to destroy (may be zero-initialised).
 */
void cvl_cl_fmm_eval_destroy(cvl_cl_fmm_eval_t *eval);
