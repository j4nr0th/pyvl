#pragma once

/*
 * Barnes-Hut octree over vortex-particle sources.
 *
 * Each tree node holds either a tagged-union of raw particle indices (small
 * leaves) or a multipole expansion (compressed leaves). The build phase
 * recursively subdivides a leaf whenever its source count exceeds an
 * order-coupled threshold. Empty subtrees are pruned so that the buffer only
 * stores octants that contain at least one source.
 *
 * The tree is built into a single caller-provided persistent buffer that the
 * kernel partitions into nodes | particle_order | multipole_coeffs |
 * topo_to_real | mp_slices. The output tree lives in the first three regions;
 * the remaining regions are transient scratch used during the build and
 * discarded after barnes_hut_tree_insert / barnes_hut_tree_build returns. A
 * separate caller-provided scratch buffer holds all the input-sized transient
 * scratch (topo array, source-leaf maps, per-thread multipole build scratch,
 * per-thread shift_exp and pse). Nothing is allocated by the build
 *
 * Default eval theta = 0.3 (opening-angle criterion).  See
 * `barnes_hut_eval_settings_t` for tuning recommendations.
 * except through an optional allocator callback that handles residual
 * fragments; the typical configuration passes NULL to use libc only when
 * really needed.
 *
 * Public entry points:
 *  - barnes_hut_buffer_size     - size the persistent buffer without building
 *  - barnes_hut_scratch_size    - size the transient scratch buffer
 *  - barnes_hut_size_work_buffer      - count pass only (no buffer write)
 *  - barnes_hut_tree_insert     - insert pass into a pre-allocated buffer
 *  - barnes_hut_tree_build      - count + insert in one call
 *  - barnes_hut_tree_*          - inspection helpers
 */

#include "common.h"
#include "multipole.h"

/* ------------------------------------------------------------------ */
/* Transient types exposed for function signatures.                    */
/* ------------------------------------------------------------------ */

/** @brief Internal topology node used during the count phase of tree construction. */
typedef struct
{
    int32_t children[8];     /**< Child topo-node indices; -1 indicates an empty child. */
    uint32_t particle_count; /**< Number of source particles in this cell. */
    uint8_t is_internal;     /**< 1 if this cell has been subdivided, 0 otherwise. */
    uint8_t depth;           /**< Octree depth of this cell (root = 0). */
    real3_t center;          /**< Centre of the cell. */
    real_t half_size;        /**< Half-extent of the cell along each axis. */
} topo_node_t;

/** @brief Partitioned view of the transient scratch buffer for the tree build. */
typedef struct
{
    topo_node_t *topo;            /**< Topology array (count-pass output). */
    uint32_t *source_leaf_topo;   /**< Per-source leaf index into the topology array. */
    unsigned *source_leaf_real;   /**< Per-source leaf index into the @c bh_node_t array. */
    unsigned n_thread_partitions; /**< Number of OpenMP thread partitions. */
    real_t *leaf_cur;             /**< Per-thread multipole build scratch (current). */
    real_t *leaf_nxt;             /**< Per-thread multipole build scratch (next). */
    real_t *leaf_coords;          /**< Per-thread source-coordinate scratch. */
    real_t *leaf_values;          /**< Per-thread source-values scratch. */
    real_t *shift_exp;            /**< Base of per-thread shift_exp regions. */
    real_t *pse;                  /**< Base of per-thread pse regions. */
} barnes_hut_scratch_t;

/**
 * @brief Discriminator for a Barnes-Hut node.
 */
typedef enum
{
    BH_NODE_INTERNAL = 0,
    BH_NODE_PARTICLE = 1,
    BH_NODE_MULTIPOLE = 2,
} bh_node_kind_t;

/**
 * @brief Build settings for the Barnes-Hut tree.
 *
 * User-tunable hyperparameters for tree construction.
 */
typedef struct
{
    unsigned order;                   /**< Multipole expansion order (>= 1). */
    unsigned critical_particle_count; /**< Minimum number of sources before a leaf becomes a multipole cell. */
    unsigned max_depth;               /**< Maximum octree depth (>= 1). */
    unsigned work_order;              /**< Internal expansion order for work buffers. */
    real_t alpha_centroid;            /**< Centroid-based subdivision threshold (0.0 = disabled).
                                           When > 0, a leaf is subdivided if any source is more than
                                           `alpha_centroid * half_size` from the cell geometric center.
                                           This guarantees sources are tightly clustered around the cell
                                           center, improving multipole convergence.  Default 0.0. */
} barnes_hut_settings_t;

/**
 * @brief Single node of the Barnes-Hut tree.
 */
typedef struct bh_node
{
    bh_node_kind_t kind;
    unsigned depth;
    real3_t center;
    real_t half_size;
    unsigned particle_begin;
    unsigned particle_count;
    union {
        struct
        {
            struct bh_node *children[8];
        } internal;
        multipole_t mp;
    } data;
} bh_node_t;

/**
 * @brief Tree handle. Holds views into the single caller-provided buffer.
 */
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

    bh_node_t *nodes;
    unsigned *particle_order;
    real_t *multipole_coeffs;
    real_t **mp_slices;
} barnes_hut_tree_t;

typedef struct
{
    unsigned n_internal;
    unsigned n_multipole_leaves;
    unsigned n_particle_leaves;
    unsigned max_depth;
} barnes_hut_count_res_t;

/**
 * @brief Settings for tree evaluation.
 *
 * Members:
 *  - `theta` — multipole acceptance criterion (MAC).
 *    - `<= 0` (default): **neighbour criterion**.  A cell's multipole is
 *      accepted whenever the target point lies outside the cell's 3×3×3
 *      neighbourhood (i.e.
 *      :math:`|\Delta x| > 2h \lor |\Delta y| > 2h \lor |\Delta z| > 2h`,
 *      where :math:`h` is the cell half-size).  This is the safest setting
 *      for general use — it never accepts a multipole for a target inside
 *      or immediately adjacent to the cell.
 *    - `> 0`: **opening-angle criterion**.  A cell's multipole is accepted
 *      whenever :math:`h / d < \theta`, where :math:`d` is the distance
 *      to the cell center.  Smaller values force deeper descent (more
 *      accurate, slower).
 *
 *      Recommended values:
 *      - ``0.3`` — far-field optimum (similar accuracy to neighbour
 *        criterion, :math:`2\!-\!10\times` faster eval).
 *      - ``0.01`` — high mid-field accuracy for clustered sources
 *        (forces near-direct evaluation, :math:`100\!-\!300\times` slower).
 *      - ``0.1`` — moderate accuracy at modest speed cost.
 *
 *      The multipole expansion of :math:`1/|r|^2` converges only when the
 *      eval point is outside the source bounding sphere.  For mid-field
 *      targets inside or just outside the source cloud, use a small
 *      opening angle (\< 0.03) or the neighbour criterion to force the
 *      evaluator down to small-enough cells.
 */
typedef struct
{
    double theta;
} barnes_hut_eval_settings_t;

#define BARNES_HUT_EVAL_SETTINGS_DEFAULT ((barnes_hut_eval_settings_t){.theta = 0.3})

/**
 * @brief Run the sequential count pass to determine tree topology.
 *
 * Walks the source coordinates and builds the topology array, counting how
 * many internal nodes, multipole leaves, and particle leaves the tree will
 * contain. No multipole coefficients are computed; this pass only determines
 * the structural sizes needed to allocate the work buffer.
 *
 * @param n_sources         Number of source points.
 * @param sources_coords    Source coordinates (read-only).
 * @param settings          Build settings (order, critical_particle_count, max_depth).
 * @param topo              Output topology array (pre-allocated from scratch).
 * @param source_leaf       Output per-source leaf index in the topology array.
 * @return Count results: n_internal, n_multipole_leaves, n_particle_leaves, max_depth.
 */
barnes_hut_count_res_t barnes_hut_count_pass(unsigned n_sources,
                                             const real3_t CVL_ARRAY_ARG(sources_coords, restrict n_sources),
                                             const barnes_hut_settings_t *settings, topo_node_t *topo,
                                             uint32_t *source_leaf);

typedef struct
{
    size_t size_topo;
    size_t size_source_leaf_topo;
    size_t size_source_leaf_real;
    size_t size_leaf_buf_per_thread;
    size_t size_leaf_coords_per_thread;
    size_t size_leaf_values_per_thread;
    size_t size_shift_exp_per_thread; /**< shift_exp bytes per thread */
    size_t size_pse_per_thread;       /**< pse bytes per thread */
} barnes_hut_scratch_sizes_t;

/**
 * @brief Compute per-region scratch sizes from input parameters alone.
 *
 * Unlike `barnes_hut_scratch_size` which returns total bytes for a given
 * thread count, this returns the individual per-region sizes. Use
 * `barnes_hut_total_scratch_size` to convert to total bytes.
 *
 * @param n_sources  Number of source points.
 * @param settings   Build settings.
 * @return Per-region scratch sizes.
 */
barnes_hut_scratch_sizes_t barnes_hut_size_scratch(unsigned n_sources, const barnes_hut_settings_t *settings);

/**
 * @brief Total scratch buffer size from per-region sizes and thread count.
 *
 * @param sizes      Per-region sizes from `barnes_hut_size_scratch`.
 * @param n_threads  Number of OpenMP threads (>= 1).
 * @return Total scratch buffer size in bytes.
 */
size_t barnes_hut_total_scratch_size(barnes_hut_scratch_sizes_t sizes, unsigned n_threads);

/**
 * @brief Compute the buffer size required to hold the tree.
 *
 * This is a pessimistic upper bound derived from input parameters alone.
 *
 * @param n_sources  Number of source points (must be > 0).
 * @param settings   Build settings.
 * @return Required buffer size in bytes, or 0 on invalid input.
 */
size_t barnes_hut_buffer_size(unsigned n_sources, const barnes_hut_settings_t *settings);

/**
 * @brief Compute the size of the transient scratch buffer required by the
 *        count and insert passes.
 *
 * The scratch buffer holds everything the build needs but does not keep:
 * the topology array used during the count pass, the per-source
 * leaf-index map (used twice), and the per-thread multipole build scratch
 * (`leaf_cur`, `leaf_nxt`, `leaf_coords`, `leaf_values`).
 * All scratch sizes are derivable from @p n_sources and @p settings; the
 * returned value is independent of the actual source distribution.
 *
 * @param n_sources  Number of source points (must be > 0).
 * @param n_threads  Number of OpenMP threads (>= 1).
 * @param settings   Build settings.
 * @return Required scratch size in bytes, or `0` on invalid input.
 */
size_t barnes_hut_scratch_size(unsigned n_sources, unsigned n_threads, const barnes_hut_settings_t *settings);

/** @brief Per-region byte sizes of the work buffer layout. */
typedef struct
{
    size_t nodes_bytes;            /**< Bytes for the @c bh_node_t array. */
    size_t particle_order_bytes;   /**< Bytes for the particle permutation array. */
    size_t multipole_coeffs_bytes; /**< Bytes for multipole coefficient storage. */
    size_t topo_to_real_bytes;     /**< Bytes for topology-to-real-node index map. */
    size_t mp_slices_bytes;        /**< Bytes for multipole slice pointers. */
} barnes_hut_work_sizes_t;

/**
 * @brief Compute the exact work buffer sizes from count-pass results.
 *
 * After running `barnes_hut_count_pass`, use this with the returned
 * `barnes_hut_count_res_t` to compute the exact layout of the work
 * buffer. Call `barnes_hut_total_work_size` to get the total bytes.
 *
 * @param n_sources              Number of source points.
 * @param settings               Build settings.
 * @param count_pass_res         Result from `barnes_hut_count_pass`.
 * @return A struct with per-region byte sizes.
 */
barnes_hut_work_sizes_t barnes_hut_size_work_buffer(unsigned n_sources,
                                                    const barnes_hut_settings_t CVL_ARRAY_ARG(settings, restrict),
                                                    barnes_hut_count_res_t count_pass_res);

/**
 * @brief Total work buffer size in bytes from per-region sizes.
 *
 * @param sizes  Per-region sizes from `barnes_hut_size_work_buffer`.
 * @return Total buffer size in bytes.
 */
size_t barnes_hut_total_work_size(barnes_hut_work_sizes_t sizes);

/** @brief Partitioned view of the work buffer, providing direct pointers to each region. */
typedef struct
{
    bh_node_t *nodes;         /**< Pointer to the @c bh_node_t array. */
    unsigned *particle_order; /**< Pointer to the particle-order permutation. */
    real_t *multipole_coeffs; /**< Pointer to multipole coefficient storage. */
    uint32_t *topo_to_real;   /**< Pointer to the topology-to-real-node index map. */
    real_t **mp_slices;       /**< Pointer to the multipole slice-pointer array. */
} barnes_hut_work_t;

/**
 * @brief Partition a scratch buffer into regions needed by the build.
 *
 * @param n_threads         Number of OpenMP threads (>= 1).
 * @param scratch_buffer    Scratch buffer.
 * @param scratch_sizes     Pre-computed scratch sizes.
 * @return A partitioned scratch view.
 */
barnes_hut_scratch_t barnes_hut_scratch_partition(unsigned n_threads, void *scratch_buffer,
                                                  barnes_hut_scratch_sizes_t scratch_sizes);

/**
 * @brief Run the insert pass into pre-allocated buffers.
 *
 * @param n_sources         Number of source points (must be > 0).
 * @param n_threads         Number of OpenMP threads (>= 1).
 * @param sources_coords    Coordinates of the source points.
 * @param sources_values    Vector source strengths.
 * @param settings          Build settings.
 * @param scratch_buffer    Transient scratch buffer.
 * @param scratch_size      Size of @p scratch_buffer in bytes.
 * @param allocator         Allocator callbacks. Pass NULL for libc default.
 * @param buffer            Persistent storage for the output tree.
 * @param buffer_size       Size of @p buffer in bytes.
 * @param out               Out-parameter for the populated tree handle.
 * @return true on success, false if buffers are too small.
 */
bool barnes_hut_tree_insert(unsigned n_sources, unsigned n_threads,
                            const real3_t CVL_ARRAY_ARG(sources_coords, restrict n_sources),
                            const real3_t CVL_ARRAY_ARG(sources_values, restrict n_sources),
                            const barnes_hut_settings_t CVL_ARRAY_ARG(settings, restrict), void *scratch_buffer,
                            size_t scratch_size, const allocator_t *allocator, void *buffer, size_t buffer_size,
                            barnes_hut_tree_t *out);

/**
 * @brief Convenience wrapper: run count + insert in one call.
 *
 * @param n_sources         Number of source points.
 * @param n_threads         Number of OpenMP threads (>= 1).
 * @param sources_coords    Coordinates of the source points.
 * @param sources_values    Vector source strengths.
 * @param settings          Build settings.
 * @param allocator         Allocator callbacks. Pass NULL for libc default.
 * @param out               Out-parameter for the populated tree handle.
 * @return true on success.
 */
bool barnes_hut_tree_build(unsigned n_sources, unsigned n_threads,
                           const real3_t CVL_ARRAY_ARG(sources_coords, restrict n_sources),
                           const real3_t CVL_ARRAY_ARG(sources_values, restrict n_sources),
                           const barnes_hut_settings_t CVL_ARRAY_ARG(settings, restrict), const allocator_t *allocator,
                           barnes_hut_tree_t *out);

/**
 * @brief Total number of nodes stored in @p tree.
 */
unsigned barnes_hut_tree_n_nodes(const barnes_hut_tree_t *tree);

/**
 * @brief Populate @p min_depth and @p max_depth with the depth range observed
 *        in @p tree.
 */
void barnes_hut_tree_depth_stats(const barnes_hut_tree_t *tree, unsigned *min_depth, unsigned *max_depth);

/**
 * @brief Total bytes occupied by the tree inside its buffer.
 *
 * Equal to `tree->buffer_size` for trees built through this module.
 */
size_t barnes_hut_tree_memory_bytes(const barnes_hut_tree_t *tree);

/**
 * @brief Evaluate the tree at a single target point.
 *
 * Walks the octree from the root, deciding at each internal node whether to
 * accept its multipole (via the MAC) or descend into its children.
 *
 * @param tree               Built tree handle.
 * @param sources_coords     Source coordinates (the same array used to build
 *                           the tree).
 * @param sources_values     Source strengths (same array used to build the
 *                           tree).
 * @param point              Target evaluation point.
 * @param eval_settings      Multipole acceptance settings.
 * @return Induced velocity @f$ \vec{v}(\mathrm{point}) @f$.
 */
real3_t barnes_hut_tree_eval(const barnes_hut_tree_t *tree, const real3_t CVL_ARRAY_ARG(sources_coords, restrict),
                             const real3_t CVL_ARRAY_ARG(sources_values, restrict), real3_t point,
                             barnes_hut_eval_settings_t eval_settings);

/**
 * @brief Evaluate the tree at multiple target points (batched, OpenMP).
 *
 * Thread-safe: each target is evaluated independently with a local stack.
 *
 * @param tree               Built tree handle.
 * @param sources_coords     Source coordinates (same arrays used to build).
 * @param sources_values     Source strengths.
 * @param n_targets          Number of target points.
 * @param targets            Array of target points.
 * @param results            Output array for induced velocities.
 * @param eval_settings      Multipole acceptance settings.
 * @param n_threads          OpenMP thread count.
 */
void barnes_hut_tree_eval_all(const barnes_hut_tree_t *tree, const real3_t CVL_ARRAY_ARG(sources_coords, restrict),
                              const real3_t CVL_ARRAY_ARG(sources_values, restrict), unsigned n_targets,
                              const real3_t CVL_ARRAY_ARG(targets, restrict n_targets),
                              real3_t CVL_ARRAY_ARG(results, restrict n_targets),
                              barnes_hut_eval_settings_t eval_settings, unsigned n_threads);
