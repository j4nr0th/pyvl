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
    double theta; /**< MAC opening-angle for interaction-list building (0 = neighbour criterion). */
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
    real3_t center;      /**< Node centre (may be updated to Γ-weighted centroid during centroids stage). */
    real3_t geom_center; /**< Geometric cell centre (set during materialize, never changed).  Used for Morton codes and
                            tree descent. */
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
    size_t size_radix_hist_per_thread; /**< Radix-sort histogram bins per thread (256 * sizeof(unsigned)). */
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
    unsigned *radix_hist; /**< [n_threads * 256] Per-thread radix-sort histogram bins. */
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

/**
 * @brief Count pass: build transient topology and count node types.
 *
 * The @p topo buffer must be at least @c (8 * n_sources + 1) * sizeof(topo_node_t) bytes.
 * Each source creates at most 7 internal nodes, so this is a tight upper bound.
 * Bounds are enforced via assert in debug builds.
 *
 * @param n_sources       Number of source particles.
 * @param sources_coords  Source positions [n_sources].
 * @param settings        Tree settings.
 * @param topo            Output topology buffer (pre-allocated, zeroed).
 * @param source_leaf     Output per-source leaf-node index [n_sources].
 * @return Node count breakdown.
 */
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

/**
 * @brief Compute node metadata: particle_begin prefix sums, reset particle_count, assign leaf_id.
 *
 * Pre-conditions:
 *   - nodes must be materialised and descended (particle_count populated).
 *   - depth_start/depth_end may be NULL (allocated internally if needed).
 *
 * Post-conditions:
 *   - particle_begin[i] = prefix sum of particle_count up to node i.
 *   - particle_count[i] reset to 0 for all nodes.
 *   - leaf_id assigned to each non-internal node (0..n_mp_leaves-1).
 *   - Returns number of multipole leaves.
 *
 * @param n_nodes     Number of nodes.
 * @param nodes       Node array.
 * @param max_depth   Maximum tree depth.
 * @param depth_start Output depth start offsets [max_depth+2] (may be NULL).
 * @param depth_end   Output depth end offsets [max_depth+2] (may be NULL).
 * @param n_threads   OpenMP thread count.
 * @return Number of multipole leaves.
 */
unsigned octree_compute_metadata(uint32_t n_nodes, octree_node_t *nodes, unsigned max_depth,
                                 unsigned depth_start[restrict], unsigned depth_end[restrict], unsigned n_threads);

void octree_fill_particle_order(unsigned n_sources, const unsigned *source_leaf_real, unsigned n_nodes,
                                octree_node_t *nodes, unsigned *particle_order, unsigned n_threads);

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
                               size_t leaf_stride, unsigned n_threads, unsigned depth);

/**
 * @brief Run the complete upward sweep (M2M), bottom-to-top.
 *
 * Iterates from @p max_depth down to 0, aggregating child multipoles
 * (or particle sources) into each internal node via @ref octree_upward_sweep_level.
 *
 * Pre-conditions:
 *   - nodes must be materialised, descended, metadata computed, centroids computed,
 *     and leaf multipoles built.
 *   - mp_slices must point to valid coefficient storage for all nodes.
 *
 * Post-conditions:
 *   - Every internal node's multipole coefficients contain the aggregated
 *     contribution of all descendants.
 *   - Internal node centres are updated to Γ-weighted centroids.
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

/* ================================================================ */
/* Morton 3D helpers (static inline)                                */
/* ================================================================ */

/**
 * @brief Split 21 low bits by interleaving with zeros (3D Morton code helper).
 *
 * Bit-parallel spread via successive shift-and-mask stages.
 * Each stage doubles the spacing between information bits:
 *
 * Mask constants (binary, grouped by stage):
 *   0x1fffff                - keep low 21 bits
 *   0x1f00000000ffff        - spread 7 → 14 bits
 *   0x1f0000ff0000ff        - spread 14 → 28 bits
 *   0x100f00f00f00f00f      - spread 28 → 42 bits
 *   0x10c30c30c30c30c3      - spread 42 → 56 bits
 *   0x1249249249249249      - final: bit at every 3rd position
 *                            (positions 0,3,6,… = X slot in 3D code)
 */
static inline uint64_t morton_split_21(uint64_t x)
{
    x &= 0x1fffffULL;
    x = (x | (x << 32)) & 0x1f00000000ffffULL;
    x = (x | (x << 16)) & 0x1f0000ff0000ffULL;
    x = (x | (x << 8)) & 0x100f00f00f00f00fULL;
    x = (x | (x << 4)) & 0x10c30c30c30c30c3ULL;
    x = (x | (x << 2)) & 0x1249249249249249ULL;
    return x;
}

/** @brief Compute 63-bit Morton code for a point within the root cell. */
static inline uint64_t morton_3d(real3_t p, real3_t root_center, real_t root_half_size)
{
    const real_t inv_cell = 1.0 / (2.0 * root_half_size);
    const real_t scale = (real_t)((1u << 21) - 1);
    real_t nx = (p.x - root_center.x) * inv_cell + 0.5;
    real_t ny = (p.y - root_center.y) * inv_cell + 0.5;
    real_t nz = (p.z - root_center.z) * inv_cell + 0.5;
    /* Clamp to [0, 1). */
    if (nx < 0)
        nx = 0;
    if (nx >= 1)
        nx = 0.999999;
    if (ny < 0)
        ny = 0;
    if (ny >= 1)
        ny = 0.999999;
    if (nz < 0)
        nz = 0;
    if (nz >= 1)
        nz = 0.999999;
    const uint64_t ix = (uint64_t)(nx * scale);
    const uint64_t iy = (uint64_t)(ny * scale);
    const uint64_t iz = (uint64_t)(nz * scale);
    return morton_split_21(ix) | (morton_split_21(iy) << 1) | (morton_split_21(iz) << 2);
}

/**
 * @brief Binary-search the Morton-sorted index array for a code range.
 */
static inline void morton_range_bounds(const unsigned *sorted_indices, const uint64_t *codes, size_t offset,
                                       size_t count, uint64_t code_min, uint64_t code_max, size_t *out_begin,
                                       size_t *out_end)
{
    if (count == 0)
    {
        *out_begin = *out_end = offset;
        return;
    }

    size_t lo = offset, hi = offset + count;
    while (lo < hi)
    {
        const size_t mid = lo + (hi - lo) / 2;
        const uint64_t mc = codes[sorted_indices[mid]];
        if (mc < code_min)
            lo = mid + 1;
        else
            hi = mid;
    }
    *out_begin = lo;

    lo = offset;
    hi = offset + count;
    while (lo < hi)
    {
        const size_t mid = lo + (hi - lo) / 2;
        const uint64_t mc = codes[sorted_indices[mid]];
        if (mc <= code_max)
            lo = mid + 1;
        else
            hi = mid;
    }
    *out_end = lo;
}

/* ================================================================ */
/* Morton sort (public API)                                         */
/* ================================================================ */

/**
 * @brief Build per-depth Morton-sorted index arrays using parallel LSD radix sort.
 *
 * Uses @c geom_center (the geometric cell centre, invariant across the build)
 * rather than the Γ-weighted centroid so that the spatial ordering is stable.
 *
 * Sorts (Morton code, node_index) pairs per depth level with a parallel
 * radix sort (8 passes, 8 bits/pass) that replaces the former qsort approach.
 * The histogram buffer (@p radix_hist) is caller-provided from the scratch buffer.
 *
 * On output:
 *   depth_offsets[d] for d = 0..max_depth_found stores the start offset for depth d.
 *   depth_offsets[max_depth_found + 1] = n_nodes (end sentinel).
 *   *out_max_depth_found = the number of depth levels found.
 *
 * @param n_nodes           Number of nodes.
 * @param nodes             Node array (geom_center must be populated).
 * @param codes             Output: Morton code for every node [n_nodes].
 * @param sorted_indices    Output: node indices sorted by Morton code (per depth) [n_nodes].
 * @param depth_offsets     Output: per-depth start offsets [(max_depth + 2)].
 * @param pairs_temp        Radix sort ping-pong buffer [2 * n_nodes * pair_size].
 * @param radix_hist        Per-thread histogram bins [n_threads * 256] (from scratch).
 * @param out_max_depth_found Output: actual max depth found.
 * @param n_threads         Number of OpenMP threads.
 * @return true on success.
 */
bool octree_build_morton_sorted(unsigned n_nodes, const octree_node_t nodes[restrict], uint64_t codes[restrict n_nodes],
                                unsigned sorted_indices[restrict n_nodes], unsigned depth_offsets[restrict],
                                uint8_t pairs_temp[restrict], unsigned radix_hist[restrict],
                                unsigned *out_max_depth_found, unsigned n_threads);

/* OCTREE_H */
