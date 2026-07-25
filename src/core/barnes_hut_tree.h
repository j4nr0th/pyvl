#pragma once

/*
 * Barnes-Hut octree over vortex-particle sources.
 *
 * The tree is built into a single caller-provided persistent buffer.
 * See the octree module (@ref octree.h) for shared sizing and build
 * pipeline functions.
 *
 * Default eval theta = 0.3 (opening-angle criterion).
 *
 * Public entry points:
 *  - barnes_hut_tree_insert     - insert pass into pre-allocated buffers
 *  - barnes_hut_tree_build      - count + insert in one call
 *  - barnes_hut_tree_eval / eval_all
 *  - barnes_hut_tree_*          - inspection helpers
 */

#include "common.h"
#include "multipole.h"
#include "octree.h"

/* ------------------------------------------------------------------ */
/* Backward-compatible typedefs                                        */
/* ------------------------------------------------------------------ */

typedef octree_settings_t barnes_hut_settings_t;
typedef octree_count_t barnes_hut_count_res_t;
typedef octree_scratch_sizes_t barnes_hut_scratch_sizes_t;
typedef octree_scratch_t barnes_hut_scratch_t;

/* ------------------------------------------------------------------ */
/* Tree handle                                                        */
/* ------------------------------------------------------------------ */

typedef struct
{
    barnes_hut_settings_t settings;
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
} barnes_hut_tree_t;

/* ------------------------------------------------------------------ */
/* Evaluation settings                                                */
/* ------------------------------------------------------------------ */

typedef struct
{
    double theta;
} barnes_hut_eval_settings_t;

#define BARNES_HUT_EVAL_SETTINGS_DEFAULT ((barnes_hut_eval_settings_t){.theta = 0.3})

/* ------------------------------------------------------------------ */
/* Work-buffer view                                                   */
/* ------------------------------------------------------------------ */

typedef struct
{
    octree_node_t *nodes;
    unsigned *particle_order;
    real_t *multipole_coeffs;
    uint32_t *topo_to_real;
    real_t **mp_slices;
} barnes_hut_work_t;

/* ------------------------------------------------------------------ */
/* Build                                                              */
/* ------------------------------------------------------------------ */

bool barnes_hut_tree_insert(unsigned n_sources, unsigned n_threads, const real3_t sources_coords[restrict n_sources],
                            const real3_t sources_values[restrict n_sources],
                            const barnes_hut_settings_t settings[restrict], void *scratch_buffer, size_t scratch_size,
                            const allocator_t *allocator, void *buffer, size_t buffer_size, barnes_hut_tree_t *out);

bool barnes_hut_tree_build(unsigned n_sources, unsigned n_threads, const real3_t sources_coords[restrict n_sources],
                           const real3_t sources_values[restrict n_sources],
                           const barnes_hut_settings_t settings[restrict], const allocator_t *allocator,
                           barnes_hut_tree_t *out);

/* ------------------------------------------------------------------ */
/* Inspection                                                         */
/* ------------------------------------------------------------------ */

unsigned barnes_hut_tree_n_nodes(const barnes_hut_tree_t *tree);
void barnes_hut_tree_depth_stats(const barnes_hut_tree_t *tree, unsigned *min_depth, unsigned *max_depth);
size_t barnes_hut_tree_memory_bytes(const barnes_hut_tree_t *tree);

/* ------------------------------------------------------------------ */
/* Evaluation                                                         */
/* ------------------------------------------------------------------ */

real3_t barnes_hut_tree_eval(const barnes_hut_tree_t *tree, const real3_t sources_coords[restrict],
                             const real3_t sources_values[restrict], real3_t point,
                             barnes_hut_eval_settings_t eval_settings);

void barnes_hut_tree_eval_all(const barnes_hut_tree_t *tree, const real3_t sources_coords[restrict],
                              const real3_t sources_values[restrict], unsigned n_targets,
                              const real3_t targets[restrict n_targets], real3_t results[restrict n_targets],
                              barnes_hut_eval_settings_t eval_settings, unsigned n_threads);
