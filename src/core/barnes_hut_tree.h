#pragma once

/**
 * @file barnes_hut_tree.h
 *
 * Barnes-Hut octree over vortex-particle sources.
 *
 * Each tree node holds either a tagged-union of raw `particle` indices (small
 * leaves) or a `multipole` expansion (compressed leaves). The build phase
 * recursively subdivides a leaf whenever its source count exceeds an
 * order-coupled threshold. Empty subtrees are pruned so that the buffer only
 * stores octants that contain at least one source.
 *
 * The tree is built into a single caller-provided persistent buffer that the
 * kernel partitions into `nodes | particle_order | multipole_coeffs |
 * shift_exp | pse | topo_to_real | mp_slices`. Only the first three regions
 * form the output tree; the remaining regions are transient scratch used
 * during the build and discarded after `barnes_hut_tree_insert` /
 * `barnes_hut_tree_build` returns. A separate caller-provided scratch buffer
 * holds all the input-sized transient scratch (topo array, source-leaf maps,
 * per-thread multipole build scratch). Nothing is allocated by the build
 * except through an optional allocator callback that handles residual
 * fragments; the typical configuration passes `NULL` to use libc only when
 * really needed. This matches the buffer-passing convention used by the rest
 * of `src/core/` and lets callers reuse both buffers across frames in a tight
 * solver loop.
 *
 * Public entry points:
 *  - `barnes_hut_buffer_size`     - size the persistent buffer without building
 *  - `barnes_hut_scratch_size`    - size the transient scratch buffer
 *  - `barnes_hut_size_work_buffer`      - count pass only (no buffer write)
 *  - `barnes_hut_tree_insert`     - insert pass into a pre-allocated buffer
 *  - `barnes_hut_tree_build`      - count + insert in one call
 *  - `barnes_hut_tree_*`          - inspection helpers
 *
 * The far-field induction query API is intentionally not exposed here. It is
 * a separate consumer of this tree and lives in a follow-up module.
 */

#include "common.h"
#include "multipole.h"

/* ------------------------------------------------------------------ */
/* Transient types exposed for function signatures.                    */
/* ------------------------------------------------------------------ */

/**
 * @brief Topology-only node used during the count phase.
 *
 * This struct lives in a temporary, function-local buffer and is *not* what
 * callers see. The real tree's `bh_node_t` is laid out in the caller-provided
 * buffer by the insert pass. The two are kept distinct so the kernel's hot
 * path never touches this scratch type.
 */
typedef struct
{
    int32_t children[8]; /* indices into topo[]; -1 if pruned/empty */
    uint32_t particle_count;
    uint8_t is_internal; /* 1 once converted to internal */
    uint8_t depth;       /* 0 at root */
    real3_t center;
    real_t half_size;
} topo_node_t;

/**
 * @brief Internal scratch view. Each pointer aliases a partition of the
 *        caller-provided `scratch_buffer`.
 */
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
} barnes_hut_scratch_t;

/**
 * @brief Discriminator for a Barnes-Hut node.
 *
 * - `BH_NODE_INTERNAL`: an octree internal node with up to eight children and
 *   its own aggregated multipole.
 * - `BH_NODE_PARTICLE`: a leaf whose cost would exceed a single multipole
 *   evaluation, kept as raw source indices.
 * - `BH_NODE_MULTIPOLE`: a leaf compressed into a multipole expansion.
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
 * Members:
 *  - `order`                    multipole order @f$ P @f$ used for every
 *                               compressed leaf and the internal-node
 *                               aggregation.
 *  - `critical_particle_count`  base threshold @f$ N_0 @f$. The effective
 *                               threshold per leaf is
 *                               @f$ \mathrm{critical\_particle\_count} \cdot
 *                                   (P + 1)^3 / 8 @f$
 *                               (clamped to @f$ \geq 1 @f$). A leaf is
 *                               subdivided whenever its source count exceeds
 *                               this value (subject to `max_depth`).
 *  - `max_depth`                hard cap on recursion. The root sits at depth
 *                               `0`; a leaf at `max_depth` is kept even when
 *                               empty so children-array bounds stay
 *                               predictable.
 *  - `work_order`               series expansion order used by
 *                               `multipole_add_shift` when aggregating child
 *                               multipoles into a parent. Set to `0` to use
 *                               `order` (the default).
 *  - `n_threads`                (removed — passed as a separate parameter to
 *                               build functions instead of being stored here)
 */
typedef struct
{
    unsigned order;
    unsigned critical_particle_count;
    unsigned max_depth;
    unsigned work_order;
} barnes_hut_settings_t;

/**
 * @brief Single node of the Barnes-Hut tree.
 *
 * The `data` field is a tagged union selected by `kind`:
 *  - `BH_NODE_INTERNAL`: `data.internal.children[8]` is read. Any slot may be
 *    `NULL` if that octant was empty (pruned during the count pass).
 *  - `BH_NODE_PARTICLE`: no field beyond `particle_begin`/`particle_count` is
 *    meaningful.
 *  - `BH_NODE_MULTIPOLE`: `data.mp` is valid. Its `coeffs_x/y/z` point into
 *    the tree's `multipole_coeffs` partition.
 *
 * The `center` is the geometric centre of the cell. The multipole at a leaf
 * carries its own `|Γ|`-weighted centroid; the two may differ for compressed
 * leaves, by design.
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
 *
 * The buffer layout is, in order:
 *  1. `nodes`           - `n_nodes` entries of `bh_node_t`.
 *  2. `particle_order`  - `n_sources` entries of `unsigned`. Each entry is an
 *                         index into the original `sources_coords` /
 *                         `sources_values` arrays.
 *  3. `multipole_coeffs`- flat coefficient storage, partitioned by
 *                         multipole-bearing leaf. Slice size is
 *                         `3 * multipole_num_coeffs(order)` per such node.
 *  4–6. `shift_exp` / `pse` / `topo_to_real` — transient scratch regions used
 *       during the build. After the build returns, only regions 1–3 and the
 *       side-table `mp_slices` (on the tree handle) carry the output tree.
 *
 * `work_order` is resolved at build time as
 * `settings.work_order ? settings.work_order : settings.order`.
 *
 * The struct owns no heap memory of its own. `buffer` is the only storage and
 * its lifetime must outlive the tree.
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
 *  - `theta` — opening angle @f$ \theta @f$. When `<= 0` (default), the
 *    neighbour criterion is used: the multipole of a cell is accepted
 *    whenever the target point lies outside the cell's 3×3×3 neighbourhood.
 *    When `> 0`, the opening-angle criterion applies: a cell's multipole is
 *    accepted whenever `half_size / distance < theta`.
 */
typedef struct
{
    double theta;
} barnes_hut_eval_settings_t;

/** @brief Default eval settings: neighbour criterion. */
#define BARNES_HUT_EVAL_SETTINGS_DEFAULT ((barnes_hut_eval_settings_t){.theta = 0.0})

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
 * @brief Compute the buffer size required to hold the tree for @p n_sources
 *        sources and the given @p settings.
 *
 * This is a pessimistic upper bound derived from input parameters alone,
 * without running the count pass. After the count pass, the exact size can
 * be obtained from `barnes_hut_size_work_buffer`. The returned size is the
 * minimum `buffer_size` argument that `barnes_hut_tree_insert` will accept.
 *
 * The buffer layout is `nodes | particle_order | multipole_coeffs |
 * shift_exp | pse | topo_to_real | mp_slices`. Only nodes, particle_order,
 * and multipole_coeffs form the output tree; the remaining regions are
 * transient build scratch and are not exposed on `barnes_hut_tree_t`.
 *
 * @param n_sources  Number of source points (must be > 0).
 * @param settings   Build settings (must have `order >= 1`,
 *                   `critical_particle_count >= 1`, `max_depth >= 1`).
 * @return Required buffer size in bytes, or `0` on invalid input.
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

typedef struct
{
    size_t nodes_bytes;
    size_t particle_order_bytes;
    size_t multipole_coeffs_bytes;
    size_t shift_exp_bytes;
    size_t pse_bytes;
    size_t topo_to_real_bytes;
    size_t mp_slices_bytes;
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

size_t barnes_hut_total_work_size(barnes_hut_work_sizes_t sizes);

typedef struct
{
    bh_node_t *nodes;
    unsigned *particle_order;
    real_t *multipole_coeffs;
    real_t *shift_exp;
    real_t *pse;
    uint32_t *topo_to_real;
    real_t **mp_slices;
} barnes_hut_work_t;

/**
 * @brief Partition a scratch buffer into the regions needed by the build.
 *
 * The partition order follows `barnes_hut_scratch_sizes_t`:
 *   topo | source_leaf_topo | source_leaf_real | leaf_cur | leaf_nxt
 *   | leaf_coords | leaf_values
 *
 * This is a convenience for callers that want to run the count pass
 * separately before allocating the work buffer.
 *
 * @param n_threads         Number of OpenMP threads (>= 1).
 * @param scratch_buffer    Scratch buffer of at least
 *                          `barnes_hut_total_scratch_size(sizes, n_threads)` bytes.
 * @param scratch_sizes     Pre-computed scratch sizes from
 *                          `barnes_hut_size_scratch`.
 * @return A partitioned scratch view.
 */
barnes_hut_scratch_t barnes_hut_scratch_partition(unsigned n_threads, void *scratch_buffer,
                                                  barnes_hut_scratch_sizes_t scratch_sizes);

/**
 * @brief Run the insert pass into pre-allocated buffers.
 *
 * The caller must have allocated at least `barnes_hut_buffer_size(n_sources,
 * settings)` bytes in @p buffer and at least
 * `barnes_hut_scratch_size(n_sources, n_threads, settings)` bytes in
 * @p scratch_buffer. The function partitions both buffers internally.
 *
 * To size the work buffer exactly (without the pessimistic bound), run
 * `barnes_hut_scratch_partition` + `barnes_hut_count_pass`, then
 * `barnes_hut_size_work_buffer` + `barnes_hut_total_work_size`.
 *
 * @param n_sources         Number of source points (must be > 0).
 * @param n_threads         Number of OpenMP threads (>= 1).
 * @param sources_coords    Coordinates of the source points.
 * @param sources_values    Vector source strengths @f$ \vec{\Gamma}_i @f$.
 * @param settings          Build settings (must have been validated by
 *                          `barnes_hut_buffer_size`).
 * @param scratch_buffer    Transient scratch buffer (must be valid).
 * @param scratch_size      Size of @p scratch_buffer in bytes.
 * @param allocator         Allocator callbacks for residual allocations.
 *                          Pass `NULL` for the libc `malloc`/`free`
 *                          fallback.
 * @param buffer            Persistent storage for the output tree
 *                          (must be valid).
 * @param buffer_size       Size of @p buffer in bytes.
 * @param out               Out-parameter receiving the populated tree handle.
 * @return `true` on success, `false` if buffers are too small or inputs
 *         are invalid.
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
 * Allocates scratch and work buffers internally through the @p allocator,
 * so the caller only needs sources, settings, and an output handle. The
 * work buffer remains owned by the returned tree (see `out->buffer`) and
 * must be freed via `allocator->deallocate()` when the tree is no longer
 * needed (or via `free()` when using the default libc allocator by passing
 * `NULL` for @p allocator).
 *
 * Use `barnes_hut_tree_insert` when you want to manage buffer lifetimes
 * yourself (e.g. reuse buffers across solver frames).
 *
 * @param n_sources         Number of source points.
 * @param n_threads         Number of OpenMP threads (>= 1).
 * @param sources_coords    Coordinates of the source points.
 * @param sources_values    Vector source strengths.
 * @param settings          Build settings.
 * @param allocator         Allocator callbacks for all allocations.
 *                          Pass `NULL` for the libc `malloc`/`free`
 *                          fallback.
 * @param out               Out-parameter receiving the populated tree handle.
 *                          The caller must eventually free `out->buffer`.
 * @return `true` on success.
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
