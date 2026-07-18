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
 * kernel partitions into `nodes | particle_order | multipole_coeffs | scratch |
 * shift_exp | pse`. A separate caller-provided scratch buffer holds all the
 * input-sized transient scratch (topo array, source-leaf maps, per-thread
 * multipole build scratch). Nothing is allocated by the build except through
 * an optional allocator callback that handles residual fragments; the typical
 * configuration passes `NULL` to use libc only when really needed. This
 * matches the buffer-passing convention used by the rest of `src/core/` and
 * lets callers reuse both buffers across frames in a tight solver loop.
 *
 * Public entry points:
 *  - `barnes_hut_buffer_size`     - size the persistent buffer without building
 *  - `barnes_hut_scratch_size`    - size the transient scratch buffer
 *  - `barnes_hut_tree_count`      - count pass only (no buffer write)
 *  - `barnes_hut_tree_insert`     - insert pass into a pre-allocated buffer
 *  - `barnes_hut_tree_build`      - count + insert in one call
 *  - `barnes_hut_tree_*`          - inspection helpers
 *
 * The far-field induction query API is intentionally not exposed here. It is
 * a separate consumer of this tree and lives in a follow-up module.
 */

#include "common.h"
#include "multipole.h"

#ifdef __cplusplus
extern "C"
{
#endif

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
     *  - `n_threads`                OpenMP thread count for the parallel count and
     *                               insert passes. Set to `0` for the OpenMP
     *                               runtime default.
     */
    typedef struct
    {
        unsigned order;
        unsigned critical_particle_count;
        unsigned max_depth;
        unsigned work_order;
        unsigned n_threads;
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
     *  4. `scratch`         - `multipole_scratch_size(work_order)` `real_t`s used
     *                         by `multipole_create` per leaf.
     *  5. `shift_exp`       - `3 * (work_order + 1)^2` `real_t`s used by
     *                         `multipole_add_shift` per internal-node aggregation.
     *  6. `pse`             - `2 * multipole_num_coeffs(work_order)` `real_t`s,
     *                         same use as `shift_exp`.
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
        real_t *scratch;
        real_t *shift_exp;
        real_t *pse;
    } barnes_hut_tree_t;

    /**
     * @brief Compute the buffer size required to hold the tree for @p n_sources
     *        sources and the given @p settings.
     *
     * This is a pure function: it walks the same subdivision logic as the count
     * pass but writes nothing. The returned size is the minimum `buffer_size`
     * argument that `barnes_hut_tree_insert` will accept.
     *
     * @param n_sources  Number of source points (must be > 0).
     * @param settings   Build settings (must have `order >= 1`,
     *                   `critical_particle_count >= 1`, `max_depth >= 1`).
     * @return Required buffer size in bytes, or `0` on invalid input.
     */
    size_t barnes_hut_buffer_size(unsigned n_sources, const barnes_hut_settings_t CVL_ARRAY_ARG(settings, restrict));

    /**
     * @brief Compute the size of the transient scratch buffer required by the
     *        count and insert passes.
     *
     * The scratch buffer holds everything the build needs but does not keep:
     * the topology array used during the count pass, the per-source
     * leaf-index map (used twice), and the worst-case per-thread multipole
     * build scratch (`leaf_cur`, `leaf_nxt`, `leaf_coords`, `leaf_values`).
     * All scratch sizes are derivable from @p n_sources and @p settings; the
     * returned value is independent of the actual source distribution.
     *
     * The scratch is sized for a single thread. When
     * `settings.n_threads > 1`, each parallel region reuses the same scratch
     * (different threads cooperate on disjoint work units), so no extra
     * allocation is required.
     *
     * @param n_sources  Number of source points (must be > 0).
     * @param settings   Build settings.
     * @return Required scratch size in bytes, or `0` on invalid input.
     */
    size_t barnes_hut_scratch_size(unsigned n_sources, const barnes_hut_settings_t CVL_ARRAY_ARG(settings, restrict));

    /**
     * @brief Run the count pass only.
     *
     * Equivalent to a build, but writes nothing and computes the total node count
     * and the buffer size. Useful when a caller wants to allocate the buffer with
     * a custom strategy before calling `barnes_hut_tree_insert`.
     *
     * @param n_sources              Number of source points.
     * @param sources_coords         Coordinates of the source points.
     * @param settings               Build settings.
     * @param scratch_buffer         Caller-provided scratch buffer of at least
     *                               `barnes_hut_scratch_size(n_sources,
     *                               settings)` bytes. The contents are not
     *                               preserved across calls.
     * @param scratch_size           Size of @p scratch_buffer in bytes.
     * @param required_buffer_size   Out-parameter receiving the required size.
     * @return `true` on success, `false` on invalid input (no output written).
     */
    bool barnes_hut_tree_count(unsigned n_sources, const real3_t CVL_ARRAY_ARG(sources_coords, restrict n_sources),
                               const barnes_hut_settings_t CVL_ARRAY_ARG(settings, restrict), void *scratch_buffer,
                               size_t scratch_size, size_t *required_buffer_size);

    /**
     * @brief Run the insert pass into a pre-allocated buffer.
     *
     * The caller must have allocated at least `barnes_hut_buffer_size(n_sources,
     * settings)` bytes in @p buffer and at least
     * `barnes_hut_scratch_size(n_sources, settings)` bytes in @p scratch_buffer.
     * Both buffers are partitioned in-place; the allocator callback is only
     * used for residual fragments that cannot be sized from inputs alone.
     *
     * @param n_sources         Number of source points.
     * @param sources_coords    Coordinates of the source points.
     * @param sources_values    Vector source strengths @f$ \vec{\Gamma}_i @f$.
     * @param settings          Build settings (the same struct passed to
     *                          `barnes_hut_buffer_size`).
     * @param scratch_buffer    Caller-provided transient scratch buffer.
     * @param scratch_size      Size of @p scratch_buffer in bytes.
     * @param allocator         Allocator callbacks for residual allocations.
     *                          Pass `NULL` for the libc `malloc`/`free`
     *                          fallback.
     * @param buffer            Pre-allocated persistent storage of at least
     *                          `required_buffer_size` bytes.
     * @param buffer_size       Size of @p buffer in bytes.
     * @param out               Out-parameter receiving the populated tree handle.
     * @return `true` on success, `false` if @p buffer or @p scratch_buffer is
     *         too small or inputs are invalid.
     */
    bool barnes_hut_tree_insert(unsigned n_sources, const real3_t CVL_ARRAY_ARG(sources_coords, restrict n_sources),
                                const real3_t CVL_ARRAY_ARG(sources_values, restrict n_sources),
                                const barnes_hut_settings_t CVL_ARRAY_ARG(settings, restrict), void *scratch_buffer,
                                size_t scratch_size, const allocator_t *allocator, void *buffer, size_t buffer_size,
                                barnes_hut_tree_t *out);

    /**
     * @brief Convenience: run count and insert in one call.
     *
     * @param n_sources         Number of source points.
     * @param sources_coords    Coordinates of the source points.
     * @param sources_values    Vector source strengths.
     * @param settings          Build settings.
     * @param scratch_buffer    Caller-provided transient scratch (see
     *                          `barnes_hut_tree_insert`).
     * @param scratch_size      Size of @p scratch_buffer in bytes.
     * @param allocator         Allocator callbacks for residual allocations
     *                          (see `barnes_hut_tree_insert`).
     * @param buffer            Caller-provided persistent storage.
     * @param buffer_size       Size of @p buffer in bytes.
     * @param out               Out-parameter receiving the populated tree handle.
     * @return `true` on success.
     */
    bool barnes_hut_tree_build(unsigned n_sources, const real3_t CVL_ARRAY_ARG(sources_coords, restrict n_sources),
                               const real3_t CVL_ARRAY_ARG(sources_values, restrict n_sources),
                               const barnes_hut_settings_t CVL_ARRAY_ARG(settings, restrict), void *scratch_buffer,
                               size_t scratch_size, const allocator_t *allocator, void *buffer, size_t buffer_size,
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

#ifdef __cplusplus
}
#endif
