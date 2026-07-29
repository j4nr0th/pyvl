#pragma once

/*
 * Fast Multipole Method tree over vortex-particle sources.
 *
 * Reuses the same adaptive octree topology from @ref octree.h.
 * The key additions over Barnes-Hut are interaction lists (V-list /
 * near-field list) and local expansion operators (M2L, L2L, L2P)
 * for O(N) FMM evaluation.
 *
 * Build pipeline (shared stages from octree + FMM-specific):
 *   1-8. Shared octree stages (count pass, materialise, descend,
 *        metadata, fill, centroids, P2M, M2M).
 *   9.   Interaction lists (V-list + near-field).
 *   10.  M2L + L2L (FMM mode only).
 */

#include "common.h"
#include "multipole.h"
#include "octree.h"

/* ------------------------------------------------------------------ */
/* Settings                                                           */
/* ------------------------------------------------------------------ */

typedef octree_settings_t fmm_settings_t;

typedef enum
{
    FMM_EVAL_TREE_CODE = 0, /**< Tree-code mode: descend to leaf, evaluate each V-list multipole directly.  Works
                               anywhere in space.  Accuracy controlled by theta (MAC). */
    FMM_EVAL_FMM = 1,    /**< FMM mode: evaluate leaf's precomputed local expansion (M2L+L2L).  O(1) per point, interior
                            only.  Diverges outside source bounding box. */
    FMM_EVAL_HYBRID = 2, /**< HYBRID mode: stack-based traversal from root, accept first node where |r'| < hybrid_alpha
                            * half_size.  Uses precomputed local expansions like FMM mode, but traversal like BH.  Works
                            anywhere in space.  Falls back to tree-code if no node converges. */
} fmm_eval_mode_t;

typedef struct
{
    double theta;         /**< MAC opening-angle for tree-code mode (0 = neighbour criterion). */
    fmm_eval_mode_t mode; /**< Evaluation mode: TREE_CODE, FMM, or HYBRID. */
    double hybrid_alpha;  /**< Convergence safety factor for HYBRID mode (default 1.5).  Node accepted when |r'| < alpha
                             * half_size.  Smaller = deeper traversal (more accurate).  Larger = shallower (faster, may
                             diverge). */
} fmm_eval_settings_t;

#define FMM_EVAL_SETTINGS_DEFAULT ((fmm_eval_settings_t){.theta = 0.0, .mode = FMM_EVAL_TREE_CODE, .hybrid_alpha = 1.5})

/* ------------------------------------------------------------------ */
/* Types                                                              */
/* ------------------------------------------------------------------ */

/* Count and scratch types are aliased from octree. */
typedef octree_count_t fmm_count_res_t;
typedef octree_scratch_sizes_t fmm_scratch_sizes_t;
typedef octree_scratch_t fmm_scratch_t;

/** @brief Per-region work-buffer byte sizes (FMM-specific extra regions). */
typedef struct
{
    size_t nodes_bytes;
    size_t particle_order_bytes;
    size_t multipole_coeffs_bytes;
    size_t topo_to_real_bytes;
    size_t mp_slices_bytes;
    size_t leaf_indices_bytes;
    size_t local_coeffs_bytes;
    size_t local_slices_bytes;
    size_t interaction_lists_bytes;
    size_t m2l_shift_exp_bytes; /**< Per-thread shift_exp scratch for M2L sweep. */
    size_t m2l_pse_bytes;       /**< Per-thread pse scratch for M2L sweep.       */
    size_t morton_sort_bytes;   /**< Morton-code sort storage: codes[n_total] + sorted_indices[n_total] +
                                   depth_offsets[max_depth+2] + pairs_temp[2 * n_total * pair_size]. */
} fmm_work_sizes_t;

/* ------------------------------------------------------------------ */
/* Tree handle                                                        */
/* ------------------------------------------------------------------ */

typedef struct
{
    fmm_settings_t settings;
    real3_t root_center;
    real_t root_half_size;

    unsigned n_sources;
    unsigned n_nodes;
    unsigned n_internal;
    unsigned n_multipole_leaves;
    unsigned n_particle_leaves;
    unsigned max_depth_reached;

    uint8_t *buffer;
    size_t buffer_size;

    octree_node_t *nodes;
    unsigned *particle_order;
    real_t *multipole_coeffs;
    real_t **mp_slices;

    unsigned *leaf_indices;
    real_t *local_coeffs;
    real_t **local_slices;

    unsigned *vlist_offsets;  /**< CSR offsets [n_leaves+1] for V-list (well-separated node indices). */
    unsigned *vlist_indices;  /**< CSR flat V-list node indices [vlist_count]. */
    unsigned *nflist_offsets; /**< CSR offsets [n_leaves+1] for near-field list (neighbour leaf IDs). */
    unsigned *nflist_indices; /**< CSR flat near-field leaf IDs [nflist_count]. */
    unsigned n_leaves;        /**< Number of leaves (multipole + particle). */
    size_t vlist_count;       /**< Total V-list entries across all leaves. */
    size_t nflist_count;      /**< Total near-field entries across all leaves. */
} fmm_tree_t;

/* ------------------------------------------------------------------ */
/* Sizing                                                             */
/* ------------------------------------------------------------------ */

fmm_work_sizes_t fmm_size_work_buffer(unsigned n_sources, const fmm_settings_t settings[restrict],
                                      fmm_count_res_t count_pass_res, unsigned n_threads);
size_t fmm_total_work_size(fmm_work_sizes_t sizes);

/* ------------------------------------------------------------------ */
/* Staged build API                                                   */
/*                                                                   */
/*  1. fmm_scratch_size()  →  total scratch bytes                    */
/*  2. caller allocates scratch                                      */
/*  3. fmm_prepare_scratch()  →  count + partitioned scratch         */
/*  4. fmm_work_size()  →  total work bytes                          */
/*  5. caller allocates work buffer                                  */
/*  6. fmm_tree_insert()  →  full pipeline into tree handle          */
/*  7. caller (or build) releases scratch buffer                     */
/*                                                                   */
/*  fmm_tree_build() does all seven steps internally.                */
/* ------------------------------------------------------------------ */

/**
 * @brief Total scratch buffer bytes needed for count + build.
 *
 * The interaction-list capacities in the work buffer use tight bounds:
 * M2L per-node lists are capped at 189 entries (6^3 - 3^3), verified
 * by assert during interaction-list construction.
 */
size_t fmm_scratch_size(unsigned n_sources, const fmm_settings_t *settings, unsigned n_threads);

/**
 * @brief Partition scratch buffer, zero topo, run count pass.
 *
 * @param scratch_buffer  Buffer of at least @ref fmm_scratch_size bytes.
 * @param n_sources       Number of source particles.
 * @param n_threads       OpenMP thread count.
 * @param sources_coords  Source positions.
 * @param settings        Tree settings.
 * @param out_count       Filled with node counts from the pass.
 * @param out_scratch     Filled with partitioned scratch view.
 * @return true on success.
 */
bool fmm_prepare_scratch(void *scratch_buffer, size_t scratch_size, unsigned n_sources, unsigned n_threads,
                         const real3_t sources_coords[restrict n_sources], const fmm_settings_t settings[restrict],
                         octree_count_t *out_count, octree_scratch_t *out_scratch);

/** @brief Total work buffer bytes needed given a finished count pass. */
size_t fmm_work_size(unsigned n_sources, const fmm_settings_t *settings, const octree_count_t *count,
                     unsigned n_threads);

/**
 * @brief Insert pass — full build pipeline into pre-sized work buffer.
 *
 * Runs all pipeline stages: materialise, descend, metadata, fill, centroids,
 * P2M, M2M, Morton sort, leaf V-list + NF-list, per-node M2L interaction lists,
 * M2L sweep, L2L sweep.
 *
 * Pre-conditions:
 *   - count and scratch must come from a prior fmm_prepare_scratch call.
 *   - buffer must be at least fmm_work_size bytes.
 *   - sources_coords and sources_values must match those used in the count pass.
 *
 * Post-conditions:
 *   - out->nodes, out->particle_order, out->multipole_coeffs, out->mp_slices,
 *     out->leaf_indices, out->local_coeffs, out->local_slices populated.
 *   - out->vlist_offsets/indices, out->nflist_offsets/indices populated (CSR).
 *   - out->vlist_count, out->nflist_count, out->n_leaves set.
 *   - out->buffer = buffer, out->buffer_size = buffer_size.
 *   - Returns false on any failure (caller should not use out).
 *
 * @note The per-thread scratch buffers work.m2l_shift_exp and work.m2l_pse
 *       are zeroed internally to prevent stale-NaN propagation through
 *       multipole_add_poly_to_order's `if (c == 0.0) continue` path.
 *
 * @param n_sources       Number of source particles.
 * @param n_threads       OpenMP thread count.
 * @param sources_coords  Source positions [n_sources].
 * @param sources_values  Source strengths [n_sources].
 * @param settings        Tree settings.
 * @param count           Count-pass results (from fmm_prepare_scratch).
 * @param scratch         Partitioned scratch view (from fmm_prepare_scratch).
 * @param allocator       Allocator for leaf-multipole scratch (may be NULL for default).
 * @param buffer          Pre-sized work buffer.
 * @param buffer_size     Size of work buffer.
 * @param out             Output tree handle.
 * @return true on success.
 */
bool fmm_tree_insert(unsigned n_sources, unsigned n_threads, const real3_t sources_coords[restrict n_sources],
                     const real3_t sources_values[restrict n_sources], const fmm_settings_t settings[restrict],
                     const octree_count_t *count, const octree_scratch_t *scratch, const allocator_t *allocator,
                     void *buffer, size_t buffer_size, fmm_tree_t *out);

/**
 * @brief Allocate + build in one call.
 *
 * Convenience wrapper that runs all seven stages internally, allocating
 * both scratch and work buffers through allocator.
 *
 * Pre-conditions:
 *   - n_sources > 0, settings valid, sources_coords/sources_values non-NULL.
 *
 * Post-conditions:
 *   - out populated as per fmm_tree_insert.
 *   - out->buffer is owned by the caller (must be freed via allocator).
 *   - Returns false on allocation failure or build failure.
 *
 * @param n_sources       Number of source particles.
 * @param n_threads       OpenMP thread count.
 * @param sources_coords  Source positions [n_sources].
 * @param sources_values  Source strengths [n_sources].
 * @param settings        Tree settings.
 * @param allocator       Allocator for scratch + work buffers (may be NULL for default).
 * @param out             Output tree handle.
 * @return true on success.
 */
bool fmm_tree_build(unsigned n_sources, unsigned n_threads, const real3_t sources_coords[restrict n_sources],
                    const real3_t sources_values[restrict n_sources], const fmm_settings_t settings[restrict],
                    const allocator_t *allocator, fmm_tree_t *out);

/* ------------------------------------------------------------------ */
/* Inspection                                                         */
/* ------------------------------------------------------------------ */

unsigned fmm_tree_n_nodes(const fmm_tree_t *tree);
size_t fmm_tree_memory_bytes(const fmm_tree_t *tree);

/* ------------------------------------------------------------------ */
/* Evaluation                                                         */
/* ------------------------------------------------------------------ */

/**
 * @brief Evaluate at a single target point.
 *
 * Pre-conditions:
 *   - tree must have been successfully built (fmm_tree_build or fmm_tree_insert returned true).
 *   - sources_coords and sources_values must match those used at build time.
 *
 * Post-conditions:
 *   - Returns the induced vector at point.
 *   - For FMM_EVAL_TREE_CODE: works anywhere in space.  Accuracy depends on theta.
 *   - For FMM_EVAL_FMM: only accurate for points well inside source bounding box
 *     (|r'| < 0.3 * domain_radius).  Diverges outside.
 *   - For FMM_EVAL_HYBRID: works anywhere in space.  Falls back to tree-code
 *     if no node's local expansion converges (|r'| < hybrid_alpha * half_size).
 *
 * @param tree            Built FMM tree handle.
 * @param sources_coords  Source coordinates [n_sources] (must match build).
 * @param sources_values  Source strengths [n_sources] (must match build).
 * @param point           Target point.
 * @param eval_settings   Evaluation mode and parameters.
 * @return Induced vector at point.
 */
real3_t fmm_tree_eval(const fmm_tree_t *tree, const real3_t sources_coords[restrict],
                      const real3_t sources_values[restrict], real3_t point, fmm_eval_settings_t eval_settings);

/**
 * @brief Batched evaluation (OpenMP parallel over targets).
 *
 * Pre-conditions:
 *   - Same as fmm_tree_eval.
 *   - results must have space for n_targets.
 *
 * Post-conditions:
 *   - results[i] = fmm_tree_eval(tree, sources_coords, sources_values, targets[i], eval_settings).
 *   - Thread-safe: each target evaluated independently (no shared mutable state).
 *
 * @param tree            Built FMM tree handle.
 * @param sources_coords  Source coordinates [n_sources].
 * @param sources_values  Source strengths [n_sources].
 * @param n_targets       Number of target points.
 * @param targets         Target points [n_targets].
 * @param results         Output array [n_targets].
 * @param eval_settings   Evaluation mode and parameters.
 * @param n_threads       OpenMP thread count.
 */
void fmm_tree_eval_all(const fmm_tree_t *tree, const real3_t sources_coords[restrict],
                       const real3_t sources_values[restrict], unsigned n_targets,
                       const real3_t targets[restrict n_targets], real3_t results[restrict n_targets],
                       fmm_eval_settings_t eval_settings, unsigned n_threads);
