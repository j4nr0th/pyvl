#pragma once

/*
 * Shared octree foundation for the pyvl tree methods.
 *
 * Both the Barnes-Hut tree and the FMM tree share the same adaptive-octree
 * build pipeline.  This module implements the pipeline once, operating
 * on a single @ref octree_node_t type used by both consumers.
 *
 * Buffer-passing convention
 * -------------------------
 * All functions take caller-owned buffers with explicit sizes.
 * Nothing is allocated internally.  The build proceeds in two passes:
 *
 *   1. **Count pass** (octree_count_pass) walks source coordinates,
 *      builds a transient topology array, and returns exact node counts.
 *   2. **Insert pass** (pipeline stages) uses the topology to
 *      materialise the tree into a caller-provided work buffer.
 *
 * A separate transient scratch buffer holds all build-time temporaries
 * (topo array, per-thread multipole scratch).  Both scratch and work
 * buffers have sizing helpers so callers can pre-allocate exactly what
 * is needed.
 */

#include "common.h"
#include "multipole.h"

#include <stddef.h>
#include <stdint.h>

/* ================================================================ */
/* Shared settings                                                  */
/* ================================================================ */

typedef struct
{
    unsigned order;
    unsigned critical_particle_count;
    unsigned max_depth;
    unsigned work_order;
    real_t alpha_centroid;
} octree_settings_t;

static inline unsigned octree_resolve_work_order(const octree_settings_t *settings)
{
    return settings->work_order ? settings->work_order : settings->order;
}

/* ================================================================ */
/* Count-pass result                                                */
/* ================================================================ */

typedef struct
{
    unsigned n_internal;
    unsigned n_multipole_leaves;
    unsigned n_particle_leaves;
    unsigned max_depth;
} octree_count_t;

/* ================================================================ */
/* Topology node (transient, used only during count pass)           */
/* ================================================================ */

typedef struct
{
    int32_t children[8];
    uint32_t particle_count;
    uint8_t is_internal;
    uint8_t depth;
    real3_t center;
    real_t half_size;
} topo_node_t;

/* ================================================================ */
/* Node type (shared between BH and FMM)                            */
/* ================================================================ */

typedef enum
{
    OCTREE_NODE_INTERNAL = 0,
    OCTREE_NODE_PARTICLE = 1,
    OCTREE_NODE_MULTIPOLE = 2,
} octree_node_kind_t;

typedef struct octree_node
{
    octree_node_kind_t kind;
    unsigned depth;
    real3_t center;
    real_t half_size;
    unsigned particle_begin;
    unsigned particle_count;
    int32_t leaf_id;
    union {
        struct
        {
            struct octree_node *children[8];
        } internal;
        multipole_t mp;
    } data;
} octree_node_t;

/* ================================================================ */
/* Per-thread scratch layout (transient)                            */
/* ================================================================ */

typedef struct
{
    size_t size_topo;
    size_t size_source_leaf_topo;
    size_t size_source_leaf_real;
    size_t size_leaf_buf_per_thread;
    size_t size_leaf_coords_per_thread;
    size_t size_leaf_values_per_thread;
    size_t size_shift_exp_per_thread;
    size_t size_pse_per_thread;
} octree_scratch_sizes_t;

typedef struct
{
    topo_node_t *topo;
    uint32_t *source_leaf_topo;
    unsigned *source_leaf_real;
    unsigned n_thread_partitions;
    real_t *leaf_cur;
    real_t *leaf_nxt;
    real_t *leaf_coords;
    real_t *leaf_values;
    real_t *shift_exp;
    real_t *pse;
} octree_scratch_t;

/* ================================================================ */
/* Base work sizes (common to BH and FMM)                           */
/* ================================================================ */

typedef struct
{
    size_t nodes_bytes;
    size_t particle_order_bytes;
    size_t multipole_coeffs_bytes;
    size_t topo_to_real_bytes;
    size_t mp_slices_bytes;
} octree_base_work_sizes_t;

/* ================================================================ */
/* Stride helpers (static inline)                                   */
/* ================================================================ */

static inline size_t octree_leaf_stride(unsigned order)
{
    const size_t n_coeffs = multipole_num_coeffs(order);
    const size_t mp_scratch = multipole_scratch_size(order);
    return n_coeffs * 3 > mp_scratch ? n_coeffs * 3 : mp_scratch;
}

static inline size_t octree_shift_stride(unsigned work_order)
{
    const size_t dim = (size_t)work_order + 1;
    return 3 * dim * dim;
}

static inline size_t octree_pse_stride(unsigned work_order)
{
    return 2 * multipole_num_coeffs(work_order);
}

/* ================================================================ */
/* Count pass                                                       */
/* ================================================================ */

/* ================================================================ */
/* Default allocator (shared across tree methods)                    */
/* ================================================================ */

extern const allocator_t CVL_DEFAULT_ALLOCATOR;

static inline const allocator_t *octree_resolve_allocator(const allocator_t *allocator)
{
    return allocator ? allocator : &CVL_DEFAULT_ALLOCATOR;
}

static inline void *octree_alloc(const allocator_t *allocator, size_t size)
{
    const allocator_t *a = octree_resolve_allocator(allocator);
    return a->allocate(a->state, size);
}

static inline void octree_free(const allocator_t *allocator, void *ptr)
{
    if (ptr == NULL)
        return;
    const allocator_t *a = octree_resolve_allocator(allocator);
    a->deallocate(a->state, ptr);
}

/* ================================================================ */
/* Count pass                                                       */
/* ================================================================ */

octree_count_t octree_count_pass(unsigned n_sources, const real3_t sources_coords[restrict n_sources],
                                 const octree_settings_t *settings, topo_node_t *topo, uint32_t *source_leaf);

/* ================================================================ */
/* Scratch sizing                                                  */
/* ================================================================ */

octree_scratch_sizes_t octree_size_scratch(unsigned n_sources, const octree_settings_t *settings);
size_t octree_total_scratch_size(octree_scratch_sizes_t sizes, unsigned n_threads);
size_t octree_scratch_size(unsigned n_sources, unsigned n_threads, const octree_settings_t *settings);
octree_scratch_t octree_scratch_partition(unsigned n_thread_partitions, void *buffer, octree_scratch_sizes_t sizes);

/* ================================================================ */
/* Work buffer sizing                                              */
/* ================================================================ */

octree_base_work_sizes_t octree_size_work_buffer(unsigned n_sources, const octree_settings_t settings[restrict],
                                                 octree_count_t count);
size_t octree_total_work_size(octree_base_work_sizes_t sizes);
size_t octree_buffer_size(unsigned n_sources, const octree_settings_t *settings);

/* ================================================================ */
/* Pipeline stages                                                  */
/* ================================================================ */

void octree_materialize(const topo_node_t topo[restrict], uint32_t n_topo_nodes, const octree_settings_t *settings,
                        uint32_t topo_to_real[restrict n_topo_nodes], octree_node_t *nodes, real_t *multipole_coeffs,
                        real_t *mp_slices[restrict n_topo_nodes], unsigned n_threads);

void octree_descend(unsigned n_sources, const real3_t sources_coords[restrict n_sources], octree_node_t *nodes,
                    unsigned *source_leaf_real, unsigned n_threads);

unsigned octree_compute_metadata(uint32_t n_nodes, octree_node_t *nodes, unsigned max_depth,
                                 unsigned depth_start[restrict], unsigned depth_end[restrict]);

void octree_fill_particle_order(unsigned n_sources, const unsigned *source_leaf_real, octree_node_t *nodes,
                                unsigned *particle_order, unsigned n_threads);

void octree_compute_leaf_centers(unsigned n_nodes, octree_node_t *nodes, const unsigned particle_order[restrict],
                                 const real3_t sources_coords[restrict], const real3_t sources_values[restrict],
                                 unsigned n_threads);

bool octree_build_leaf_multipoles(unsigned n_nodes, octree_node_t *nodes, const unsigned *restrict particle_order,
                                  const real3_t sources_coords[restrict], const real3_t sources_values[restrict],
                                  const octree_settings_t *settings, octree_scratch_t scratch, unsigned n_sources,
                                  unsigned n_threads, real_t *mp_slices[restrict n_nodes], unsigned n_mp_leaves,
                                  const allocator_t *allocator);

void octree_compute_depth_ranges(unsigned n_nodes, const octree_node_t nodes[restrict n_nodes], unsigned max_depth,
                                 unsigned depth_start[restrict max_depth + 2],
                                 unsigned depth_end[restrict max_depth + 2]);

void octree_upward_sweep_level(unsigned depth_start, unsigned depth_end, octree_node_t nodes[restrict], unsigned order,
                               size_t n_coeffs, real_t *mp_slices[restrict], unsigned work_order,
                               real_t shift_exp[restrict], real_t pse[restrict], size_t shift_stride, size_t pse_stride,
                               const unsigned particle_order[restrict], const real3_t sources_coords[restrict],
                               const real3_t sources_values[restrict], const octree_scratch_t *scratch,
                               size_t leaf_stride, unsigned n_threads);

/**
 * @brief Run the complete upward sweep (M2M), bottom-to-top.
 *
 * Iterates from @p max_depth down to 0, aggregating child multipoles
 * (or particle sources) into each internal node via @ref octree_upward_sweep_level.
 *
 * @param n_nodes          Number of nodes.
 * @param nodes            Node array.
 * @param max_depth        Maximum tree depth.
 * @param settings         Tree settings (order, work_order).
 * @param mp_slices        Per-node multipole coefficient slices.
 * @param scratch          Scratch buffer (per-thread shift_exp/pse/leaf buffers).
 * @param particle_order   Source → leaf particle ordering.
 * @param sources_coords   Source coordinates.
 * @param sources_values   Source strengths.
 * @param n_threads        Number of OpenMP threads.
 */
void octree_run_upward_sweep(unsigned n_nodes, octree_node_t nodes[restrict], unsigned max_depth,
                             const octree_settings_t settings[restrict], real_t **restrict mp_slices,
                             const octree_scratch_t *scratch, const unsigned particle_order[restrict],
                             const real3_t sources_coords[restrict], const real3_t sources_values[restrict],
                             unsigned n_threads);

/* OCTREE_H */
