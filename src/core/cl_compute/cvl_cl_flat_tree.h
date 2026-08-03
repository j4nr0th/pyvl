#pragma once
/*
 * Uniform octree in flat, GPU-friendly format.
 *
 * The tree is built bottom-up from Morton-sorted particles, producing
 * a level-ordered flat node array.  All nodes at depth d are stored
 * contiguously; depth_offsets[d] gives the start index.
 *
 * Child access: an internal node stores child_base (index of first
 * child in the depth d+1 layer) and child_mask (8-bit mask of which
 * octants have children).  Children are stored compactly - no gaps.
 * To find child for octant o: if (mask & (1<<o)) then child = nodes[
 * child_base + popcount(mask & ((1<<o)-1)) ].
 *
 * This format is designed for GPU traversal: a single 64-byte struct
 * with no pointer chasing - only index arithmetic.
 */

#include "../common.h"
#include "../opencl/cvl_cl_common.h"

#include <stdint.h>

/* ------------------------------------------------------------------ */
/* Flat node (64 bytes, cache-line friendly)                           */
/* ------------------------------------------------------------------ */

typedef enum
{
    CVL_CL_FLAT_NODE_INTERNAL = 0,
    CVL_CL_FLAT_NODE_PARTICLE = 1,
    CVL_CL_FLAT_NODE_MULTIPOLE = 2,
} cvl_cl_flat_node_kind_t;

typedef struct
{
    real3_t center;         /**< Node centre (24 bytes). */
    real_t half_size;       /**< Cell half-side (8 bytes). */
    uint64_t morton_code;   /**< Morton code at this node's depth (8 bytes). */
    int32_t child_base;     /**< Index of first child in next depth layer (-1 = leaf). */
    int32_t particle_begin; /**< Start index in particle_order (leaves only). */
    uint8_t child_mask;     /**< Bitmask of present children (internal only). */
    uint8_t kind;           /**< INTERNAL / PARTICLE / MULTIPOLE. */
    int16_t particle_count; /**< Number of particles in this leaf. */
    uint8_t pad[12];        /**< Explicit padding to 64 bytes. */
} cvl_cl_flat_node_t;

_Static_assert(sizeof(cvl_cl_flat_node_t) == 64, "cvl_cl_flat_node_t must be 64 bytes");

/* ------------------------------------------------------------------ */
/* Builder settings                                                    */
/* ------------------------------------------------------------------ */

typedef struct
{
    unsigned max_depth;               /**< Maximum octree depth (≤ 21 for 64-bit Morton). */
    unsigned critical_particle_count; /**< Subdivision threshold. */
    unsigned order;                   /**< Multipole expansion order. */
} cvl_cl_flat_tree_settings_t;

/** @brief Default settings: depth=8, critical=8, order=4. */
#define CVL_CL_FLAT_TREE_SETTINGS_DEFAULT                                                                              \
    ((cvl_cl_flat_tree_settings_t){.max_depth = 8, .critical_particle_count = 8, .order = 4})

/* ------------------------------------------------------------------ */
/* Build result                                                       */
/* ------------------------------------------------------------------ */

typedef struct
{
    cvl_cl_flat_node_t *nodes;   /**< Flat node array (level-ordered). */
    unsigned *particle_order;    /**< Per-leaf particle indices (sorted). */
    unsigned *depth_offsets;     /**< [max_depth + 2] start of each depth level + sentinel. */
    unsigned n_nodes;            /**< Total nodes. */
    unsigned n_internal;         /**< Count of internal nodes. */
    unsigned n_multipole_leaves; /**< Count of multipole leaf nodes. */
    unsigned n_particle_leaves;  /**< Count of particle leaf nodes. */
    unsigned max_depth;          /**< Actual max depth used. */
} cvl_cl_flat_tree_t;

/* ------------------------------------------------------------------ */
/* Sizing helpers                                                     */
/* ------------------------------------------------------------------ */

/**
 * @brief Compute the shift amount to extract the depth-d key from a
 *        full 63-bit Morton code.
 *
 * The full Morton code packs 21 bits per axis (63 bits total).  At
 * depth d we only look at the top 3*d bits.
 */
static inline unsigned cvl_cl_flat_tree_depth_shift(unsigned depth)
{
    return 63u - 3u * depth;
}

/**
 * @brief Extract the depth-d key from a full Morton code.
 */
static inline uint64_t cvl_cl_flat_tree_key_at_depth(uint64_t code, unsigned depth)
{
    if (depth == 0)
        return 0;
    return code >> cvl_cl_flat_tree_depth_shift(depth);
}

/**
 * @brief Maximum depth representable with 64-bit Morton codes.
 */
enum
{
    CVL_CL_FLAT_TREE_MAX_DEPTH = 21
};

/* ------------------------------------------------------------------ */
/* Builder                                                            */
/* ------------------------------------------------------------------ */

/**
 * @brief Count the number of nodes per depth level for a uniform octree.
 *
 * Scans the sorted particle Morton codes and counts nodes at each depth
 * level, bottom-up.
 *
 * @param n_sources         Number of particles.
 * @param morton_codes      Particle Morton codes (sorted ascending).
 * @param settings          Tree settings.
 * @param out_depth_counts  Output [max_depth+2] - count of nodes at each depth.
 * @param out_n_total       Total node count.
 * @param out_max_depth_used Actual max depth used (may be < settings.max_depth).
 * @return CVL_CL_SUCCESS.
 */
cvl_cl_status_t cvl_cl_flat_tree_count(unsigned n_sources, const uint64_t morton_codes[restrict],
                                       const cvl_cl_flat_tree_settings_t settings[restrict],
                                       unsigned out_depth_counts[restrict], unsigned *out_n_total,
                                       unsigned *out_max_depth_used);

/**
 * @brief Build a uniform flat octree from Morton-sorted particles.
 *
 * Steps:
 *   1. Count nodes per level (cvl_cl_flat_tree_count).
 *   2. Allocate flat node array + particle_order.
 *   3. Fill leaf nodes (group by Morton key at max_depth).
 *   4. Fill internal nodes bottom-up.
 *   5. Populate depth_offsets.
 *
 * @param n_sources         Number of particles.
 * @param sources_coords    Particle positions [n_sources].
 * @param particle_indices  Particle index permutation sorted by Morton code [n_sources].
 * @param morton_codes      Particle Morton codes (sorted ascending) [n_sources].
 * @param settings          Tree settings.
 * @param out_tree          Filled with result (caller must destroy via cvl_cl_flat_tree_destroy).
 * @param work              Pre-allocated work buffer (size from cvl_cl_flat_tree_work_size).
 * @param work_size         Size of work buffer in bytes.
 * @return CVL_CL_SUCCESS or error.
 */
cvl_cl_status_t cvl_cl_flat_tree_build(unsigned n_sources, const real3_t sources_coords[restrict n_sources],
                                       const unsigned particle_indices[restrict n_sources],
                                       const uint64_t morton_codes[restrict n_sources],
                                       const cvl_cl_flat_tree_settings_t settings[restrict],
                                       cvl_cl_flat_tree_t *out_tree, void *work, size_t work_size);

/**
 * @brief Compute the work-buffer size needed for cvl_cl_flat_tree_build.
 *
 * @param n_total    Total node count (from cvl_cl_flat_tree_count).
 * @param n_sources  Number of particles.
 * @param max_depth  Maximum tree depth (from settings).
 * @return Required work-buffer size in bytes.
 */
static inline size_t cvl_cl_flat_tree_work_size(unsigned n_total, unsigned n_sources, unsigned max_depth)
{
    return (size_t)n_total * sizeof(cvl_cl_flat_node_t) + (size_t)n_sources * sizeof(unsigned) +
           (size_t)(max_depth + 2) * sizeof(unsigned);
}

/**
 * @brief Reset a flat-tree handle.
 *
 * The tree's buffers alias the caller's work buffer, so nothing is
 * freed - this just clears the handle fields.
 *
 * @param tree Tree to reset (may be NULL).
 */
void cvl_cl_flat_tree_destroy(cvl_cl_flat_tree_t *tree);
