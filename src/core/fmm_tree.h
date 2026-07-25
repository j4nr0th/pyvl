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
    FMM_EVAL_TREE_CODE = 0,
    FMM_EVAL_FMM = 1,
} fmm_eval_mode_t;

typedef struct
{
    double theta;
    fmm_eval_mode_t mode;
} fmm_eval_settings_t;

#define FMM_EVAL_SETTINGS_DEFAULT ((fmm_eval_settings_t){.theta = 0.0, .mode = FMM_EVAL_TREE_CODE})

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

    unsigned *vlist_offsets;
    unsigned *vlist_indices;
    unsigned *nflist_offsets;
    unsigned *nflist_indices;
    unsigned n_leaves;
    size_t vlist_count;
    size_t nflist_count;
} fmm_tree_t;

/* ------------------------------------------------------------------ */
/* Sizing                                                             */
/* ------------------------------------------------------------------ */

fmm_work_sizes_t fmm_size_work_buffer(unsigned n_sources, const fmm_settings_t settings[restrict],
                                      fmm_count_res_t count_pass_res);
size_t fmm_total_work_size(fmm_work_sizes_t sizes);

/* ------------------------------------------------------------------ */
/* Build                                                              */
/* ------------------------------------------------------------------ */

bool fmm_tree_insert(unsigned n_sources, unsigned n_threads, const real3_t sources_coords[restrict n_sources],
                     const real3_t sources_values[restrict n_sources], const fmm_settings_t settings[restrict],
                     void *scratch_buffer, size_t scratch_size, const allocator_t *allocator, void *buffer,
                     size_t buffer_size, fmm_tree_t *out);

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

real3_t fmm_tree_eval(const fmm_tree_t *tree, const real3_t sources_coords[restrict],
                      const real3_t sources_values[restrict], real3_t point, fmm_eval_settings_t eval_settings);

void fmm_tree_eval_all(const fmm_tree_t *tree, const real3_t sources_coords[restrict],
                       const real3_t sources_values[restrict], unsigned n_targets,
                       const real3_t targets[restrict n_targets], real3_t results[restrict n_targets],
                       fmm_eval_settings_t eval_settings, unsigned n_threads);
