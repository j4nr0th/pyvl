#include "barnes_hut_tree.h"

#include <omp.h>

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* Default allocator (libc malloc/free fallback)                      */
/* ------------------------------------------------------------------ */

/**
 * @brief Default allocator backed by libc `malloc`/`free`.
 *
 * Used when the caller passes `NULL` for the allocator argument. The
 * `reallocate` slot is a thin `realloc` wrapper so callers can use the
 * allocator as a growable arena if desired.
 */
static void *bh_default_allocate(void *state, size_t size)
{
    (void)state;
    return malloc(size);
}

static void bh_default_deallocate(void *state, void *ptr)
{
    (void)state;
    free(ptr);
}

static void *bh_default_reallocate(void *state, void *ptr, size_t new_size)
{
    (void)state;
    return realloc(ptr, new_size);
}

static const allocator_t BH_DEFAULT_ALLOCATOR = {
    .allocate = bh_default_allocate,
    .deallocate = bh_default_deallocate,
    .reallocate = bh_default_reallocate,
    .state = NULL,
};

/**
 * @brief Resolve the effective allocator. Falls back to the libc-backed
 *        default when the caller passes `NULL`.
 */
static inline const allocator_t *bh_allocator(const allocator_t *allocator)
{
    return allocator ? allocator : &BH_DEFAULT_ALLOCATOR;
}

static inline void *bh_alloc(const allocator_t *allocator, size_t size)
{
    const allocator_t *a = bh_allocator(allocator);
    return a->allocate(a->state, size);
}

static inline void *bh_calloc(const allocator_t *allocator, size_t count, size_t elem_size)
{
    /* We don't expose a calloc slot on the allocator, so allocate and zero
     * manually. Used only for the topo scratch (small, called once). */
    const size_t total = count * elem_size;
    void *p = bh_alloc(allocator, total);
    if (p != NULL)
        memset(p, 0, total);
    return p;
}

static inline void bh_free(const allocator_t *allocator, void *ptr)
{
    if (ptr == NULL)
        return;
    const allocator_t *a = bh_allocator(allocator);
    a->deallocate(a->state, ptr);
}

/* ------------------------------------------------------------------ */
/* Internal types                                                     */
/* ------------------------------------------------------------------ */

/**
 * @brief Per-thread view of the leaf scratch region: `leaf_cur[t]`,
 *        `leaf_nxt[t]`, `leaf_coords[t]`, `leaf_values[t]` for thread `t`.
 *
 * The OpenMP parallel region is pinned to `n_thread_partitions` via
 * `num_threads(n_threads)`, so the thread id is always in range.
 */
typedef struct
{
    real_t *leaf_cur;
    real_t *leaf_nxt;
    real_t *leaf_coords;
    real_t *leaf_values;
} barnes_hut_leaf_scratch_t;

/* ------------------------------------------------------------------ */
/* Static helpers                                                     */
/* ------------------------------------------------------------------ */

/** Return the octant index (0..7) of @p p relative to @p center. */
static inline unsigned octant_of(const real3_t p, const real3_t center)
{
    return (unsigned)(p.x >= center.x) * 1u + (unsigned)(p.y >= center.y) * 2u + (unsigned)(p.z >= center.z) * 4u;
}

/**
 * @brief Resolve the effective work order. `settings.work_order == 0` falls
 *        back to `settings.order`.
 */
static inline unsigned resolve_work_order(const barnes_hut_settings_t *settings)
{
    return settings->work_order ? settings->work_order : settings->order;
}

/**
 * @brief Compute the per-leaf critical thresholds:
 *   - `multipole_threshold` — leaves with n > this become MULTIPOLE leaves.
 *   - `subdivide_threshold` — leaves with n > this AND depth < max_depth subdivide.
 *
 * The thresholds control two different things:
 *   - `multipole_threshold` — sources in a leaf > this → compress to multipole.
 *   - `subdivide_threshold` — sources in a leaf > this AND depth < max_depth
 *     → split the cell into 8 children.
 *
 * The subdivision threshold uses octant-factor 8 (one child per octant):
 * a leaf is subdivided when it has more than critical\cdot 8 particles,
 * so each child receives roughly critical particles on average.  This gives
 * deeper trees with smaller cells, improving mid-field accuracy at modest
 * build-cost increase.
 *
 * :math:`\mathrm{mp\_threshold} = \max(1, \mathrm{critical\_particle\_count})`
 * :math:`\mathrm{subdivide\_threshold} = \mathrm{critical\_particle\_count} \cdot 8`
 */
static unsigned multipole_threshold(const barnes_hut_settings_t *settings)
{
    return settings->critical_particle_count < 1u ? 1u : settings->critical_particle_count;
}

static unsigned subdivide_threshold(const barnes_hut_settings_t *settings)
{
    /* Octree splits into 8 children — subdivide when a leaf has more than
     * 8× the critical count, so each child gets ~critical on average. */
    const uint64_t num = (uint64_t)settings->critical_particle_count * 8u;
    return (unsigned)(num < 1u ? 1u : num);
}

/**
 * @brief Decide whether a leaf with `n` particles at `depth` should subdivide.
 */
static inline bool should_subdivide(uint32_t n, unsigned depth, const barnes_hut_settings_t *settings)
{
    return n > subdivide_threshold(settings) && depth < settings->max_depth;
}

/**
 * @brief Decide whether a non-subdividing leaf should be compressed into a
 *        multipole.
 */
static inline bool should_be_multipole(uint32_t n, const barnes_hut_settings_t *settings)
{
    return n > multipole_threshold(settings);
}

/**
 * @brief Decide whether a source at @p pos triggers centroid-based subdivision
 *        of the cell at @p center with half-size @p half_size.
 *
 * Subdivides when |pos - center| > alpha_centroid * half_size.
 * The default 0.0 disables this criterion.
 */
static inline bool should_subdivide_centroid(const real3_t pos, const real3_t center, real_t half_size,
                                             const barnes_hut_settings_t *settings)
{
    if (settings->alpha_centroid <= 0.0)
        return false;
    const real3_t diff = real3_sub(pos, center);
    const real_t dist = real3_mag(diff);
    return dist > settings->alpha_centroid * half_size;
}

/* ------------------------------------------------------------------ */
/* Validation                                                         */
/* ------------------------------------------------------------------ */

static bool settings_valid(const barnes_hut_settings_t *settings)
{
    if (settings == NULL)
        return false;
    if (settings->order < 1)
        return false;
    if (settings->critical_particle_count < 1)
        return false;
    if (settings->max_depth < 1)
        return false;
    if (settings->alpha_centroid < 0.0)
        return false;
    return true;
}

/* ------------------------------------------------------------------ */
/* Count phase                                                        */
/* ------------------------------------------------------------------ */

/**
 * @brief Run the count pass over @p sources_coords using a topo-scratch
 *        array. On return, `topo[]` holds the full tree topology and the
 *        classification out-parameters are populated.
 *
 * Each source is *recorded* at the leaf it finally rests in. When a leaf
 * converts to internal during the walk, the existing sources attributed to
 * it are NOT redistributed — that requires a fresh walk. We instead mark
 * the conversion but reset the count; sources that arrived before the
 * conversion are dropped from this pass. The insert pass then re-walks
 * all sources against the rebuilt topology, which is correct because:
 *   - The topology built by this walk IS the final topology (we convert
 *     only when `n > threshold`, and the conversions trigger fresh
 *     children that further sources fill).
 *   - The insert pass uses the actual source coordinates (not counts) to
 *     decide subdivision, so the leaf each source ends up in is
 *     deterministic.
 */
barnes_hut_count_res_t barnes_hut_count_pass(unsigned n_sources,
                                             const real3_t CVL_ARRAY_ARG(sources_coords, restrict n_sources),
                                             const barnes_hut_settings_t *settings, topo_node_t *topo,
                                             uint32_t *source_leaf)
{
    /* Initialize root. */
    topo[0] = (topo_node_t){
        .children = {-1, -1, -1, -1, -1, -1, -1, -1},
        .particle_count = 0,
        .is_internal = 1,
        .depth = 0,
        .center = {.x = 0, .y = 0, .z = 0},
        .half_size = 0,
    };

    /* Root bounding box — grown by every source. */
    real3_t bbox_min = {.x = sources_coords[0].x, .y = sources_coords[0].y, .z = sources_coords[0].z};
    real3_t bbox_max = bbox_min;
    for (unsigned i = 1; i < n_sources; ++i)
    {
        const real3_t p = sources_coords[i];
        if (p.x < bbox_min.x)
            bbox_min.x = p.x;
        if (p.y < bbox_min.y)
            bbox_min.y = p.y;
        if (p.z < bbox_min.z)
            bbox_min.z = p.z;
        if (p.x > bbox_max.x)
            bbox_max.x = p.x;
        if (p.y > bbox_max.y)
            bbox_max.y = p.y;
        if (p.z > bbox_max.z)
            bbox_max.z = p.z;
    }

    /* Pad by eps so points exactly on the face land in a deterministic octant. */
    const real_t eps = 1e-12;
    const real3_t root_center = {
        .x = 0.5 * (bbox_min.x + bbox_max.x),
        .y = 0.5 * (bbox_min.y + bbox_max.y),
        .z = 0.5 * (bbox_min.z + bbox_max.z),
    };
    real_t root_half = 0.5 * real3_max(real3_sub(bbox_max, bbox_min));
    if (root_half < eps)
        root_half = eps;
    root_half += eps;
    topo[0].center = root_center;
    topo[0].half_size = root_half;

    uint32_t next_node = 1; /* next free topo slot; root is 0 */
    uint32_t max_depth = 0;

    /* Insertion loop. */
    for (unsigned i = 0; i < n_sources; ++i)
    {
        const real3_t p = sources_coords[i];
        uint32_t node_idx = 0;
        unsigned depth = 0;

        /* Descend, subdividing along the way if needed. */
        while (topo[node_idx].is_internal)
        {
            const unsigned oct = octant_of(p, topo[node_idx].center);
            int32_t child = topo[node_idx].children[oct];
            if (child < 0)
            {
                if (next_node >= 8u * (uint32_t)n_sources + 1u)
                {
                    /* Should never happen given our upper bound; bail safely. */
                    source_leaf[i] = 0;
                    return (barnes_hut_count_res_t){0};
                }
                child = (int32_t)next_node++;
                topo[child] = (topo_node_t){
                    .children = {-1, -1, -1, -1, -1, -1, -1, -1},
                    .particle_count = 0,
                    .is_internal = 0,
                    .depth = (uint8_t)(depth + 1),
                    .center = {.x = 0, .y = 0, .z = 0},
                    .half_size = topo[node_idx].half_size * 0.5,
                };
                topo[node_idx].children[oct] = child;

                /* Geometry: child octant centre = parent_centre +/- parent_half/2. */
                const real_t h = topo[node_idx].half_size * 0.5;
                topo[child].center.x = topo[node_idx].center.x + ((oct & 1u) ? h : -h);
                topo[child].center.y = topo[node_idx].center.y + ((oct & 2u) ? h : -h);
                topo[child].center.z = topo[node_idx].center.z + ((oct & 4u) ? h : -h);
            }
            node_idx = (uint32_t)child;
            depth = (unsigned)topo[node_idx].depth;
        }

        /* Landed in a leaf — increment count. */
        topo[node_idx].particle_count += 1;

        /* Subdivide if count or centroid threshold exceeded. */
        if (should_subdivide(topo[node_idx].particle_count, depth, settings) ||
            should_subdivide_centroid(p, topo[node_idx].center, topo[node_idx].half_size, settings))
        {
            topo[node_idx].is_internal = 1;
            for (int k = 0; k < 8; ++k)
                topo[node_idx].children[k] = -1;
            topo[node_idx].particle_count = 0;

            /* Re-descend into the freshly-converted internal node for
             * the *current* source. The same descent loop above handles
             * child allocation as needed. */
            while (topo[node_idx].is_internal)
            {
                const unsigned oct = octant_of(p, topo[node_idx].center);
                int32_t child = topo[node_idx].children[oct];
                if (child < 0)
                {
                    if (next_node >= 8u * (uint32_t)n_sources + 1u)
                    {
                        source_leaf[i] = 0;
                        return (barnes_hut_count_res_t){0};
                    }
                    child = (int32_t)next_node++;
                    topo[child] = (topo_node_t){
                        .children = {-1, -1, -1, -1, -1, -1, -1, -1},
                        .particle_count = 0,
                        .is_internal = 0,
                        .depth = (uint8_t)(depth + 1),
                        .center = {.x = 0, .y = 0, .z = 0},
                        .half_size = topo[node_idx].half_size * 0.5,
                    };
                    topo[node_idx].children[oct] = child;

                    const real_t h = topo[node_idx].half_size * 0.5;
                    topo[child].center.x = topo[node_idx].center.x + ((oct & 1u) ? h : -h);
                    topo[child].center.y = topo[node_idx].center.y + ((oct & 2u) ? h : -h);
                    topo[child].center.z = topo[node_idx].center.z + ((oct & 4u) ? h : -h);
                }
                node_idx = (uint32_t)child;
                depth = (unsigned)topo[node_idx].depth;
            }
            /* Now node_idx is a leaf again. Attribute this source to it. */
            topo[node_idx].particle_count += 1;
        }

        source_leaf[i] = node_idx;
        if (depth > max_depth)
            max_depth = depth;
    }

    /* Walk topo and classify. After all conversions, leaves are final:
     * a leaf with > multipole_threshold particles is a multipole leaf;
     * <= multipole_threshold is a particle leaf. */
    unsigned n_internal = 0;
    unsigned n_multipole = 0;
    unsigned n_particle = 0;
    for (uint32_t i = 0; i < next_node; ++i)
    {
        if (topo[i].is_internal)
        {
            n_internal += 1;
        }
        else if (should_be_multipole(topo[i].particle_count, settings))
        {
            n_multipole += 1;
        }
        else
        {
            n_particle += 1;
        }
    }

    return (barnes_hut_count_res_t){
        .n_internal = n_internal,
        .n_multipole_leaves = n_multipole,
        .n_particle_leaves = n_particle,
        .max_depth = max_depth,
    };
}

/* ------------------------------------------------------------------ */
/* Public API — buffer sizing                                         */
/* ------------------------------------------------------------------ */

size_t barnes_hut_buffer_size(unsigned n_sources, const barnes_hut_settings_t *settings)
{
    if (n_sources == 0 || !settings_valid(settings))
        return 0;

    /* Pessimistic upper bound on node count: n_sources leaves + ~ n_sources/7
     * internal nodes. We add a small constant for alignment slack. */
    const size_t max_nodes = (size_t)n_sources + (size_t)((n_sources + 6u) / 7u) + 16u;
    const size_t nodes_bytes = max_nodes * sizeof(bh_node_t);
    const size_t particle_order_bytes = (size_t)n_sources * sizeof(unsigned);
    /* Worst case: every node carries a multipole. */
    const size_t multipole_coeffs_bytes = max_nodes * 3u * multipole_num_coeffs(settings->order) * sizeof(real_t);

    return nodes_bytes + particle_order_bytes + multipole_coeffs_bytes;
}

/* ------------------------------------------------------------------ */
/* Public API — scratch sizing                                        */
/* ------------------------------------------------------------------ */

/**
 * @brief Compute the size of the transient scratch buffer required by the
 *        count and insert passes.
 */
barnes_hut_scratch_sizes_t barnes_hut_size_scratch(unsigned n_sources, const barnes_hut_settings_t *settings)
{
    /* Layout (one contiguous byte stream, all offsets are byte offsets):
     *   1. topo                — (8 * n_sources + 1) * sizeof(topo_node_t)
     *   2. source_leaf_topo    — n_sources * sizeof(uint32_t)
     *   3. source_leaf_real    — n_sources * sizeof(unsigned)
     *   4. leaf_cur, leaf_nxt  — 2 * leaf_buf_size * sizeof(real_t) per thread
     *   5. leaf_coords         — n_sources * 3 * sizeof(real_t) per thread
     *   6. leaf_values         — n_sources * 3 * sizeof(real_t) per thread
     *   7. shift_exp           — 3 * (work_order+1)^2 * sizeof(real_t) per thread
     *   8. pse                 — 2 * n_coeffs(work_order) * sizeof(real_t) per thread
     */
    const size_t topo_bytes = (8u * (size_t)n_sources + 1u) * sizeof(topo_node_t);
    const size_t source_leaf_topo_bytes = (size_t)n_sources * sizeof(uint32_t);
    const size_t source_leaf_real_bytes = (size_t)n_sources * sizeof(unsigned);

    const size_t n_coeffs = multipole_num_coeffs(settings->order);
    const size_t multipole_scratch = multipole_scratch_size(settings->order);
    const size_t leaf_buf_size = (3u * n_coeffs > multipole_scratch ? 3u * n_coeffs : multipole_scratch);
    const size_t leaf_buf_per_thread = 2u * leaf_buf_size * sizeof(real_t);
    const size_t leaf_coords_per_thread = (size_t)n_sources * 3u * sizeof(real_t);
    const size_t leaf_values_per_thread = (size_t)n_sources * 3u * sizeof(real_t);

    const unsigned work_order = resolve_work_order(settings);
    const size_t size_shift_exp_per_thread = 3u * (size_t)(work_order + 1) * (size_t)(work_order + 1) * sizeof(real_t);
    const size_t size_pse_per_thread = 2u * multipole_num_coeffs(work_order) * sizeof(real_t);

    return (barnes_hut_scratch_sizes_t){
        .size_topo = topo_bytes,
        .size_source_leaf_topo = source_leaf_topo_bytes,
        .size_source_leaf_real = source_leaf_real_bytes,
        .size_leaf_buf_per_thread = leaf_buf_per_thread,
        .size_leaf_coords_per_thread = leaf_coords_per_thread,
        .size_leaf_values_per_thread = leaf_values_per_thread,
        .size_shift_exp_per_thread = size_shift_exp_per_thread,
        .size_pse_per_thread = size_pse_per_thread,
    };
}

size_t barnes_hut_total_scratch_size(barnes_hut_scratch_sizes_t sizes, unsigned n_threads)
{
    if (n_threads < 1)
        return 0;

    const size_t leaf_buf_total = sizes.size_leaf_buf_per_thread * (size_t)n_threads;
    const size_t leaf_coords_total = sizes.size_leaf_coords_per_thread * (size_t)n_threads;
    const size_t leaf_values_total = sizes.size_leaf_values_per_thread * (size_t)n_threads;
    const size_t shift_exp_total = sizes.size_shift_exp_per_thread * (size_t)n_threads;
    const size_t pse_total = sizes.size_pse_per_thread * (size_t)n_threads;

    return sizes.size_topo + sizes.size_source_leaf_topo + sizes.size_source_leaf_real + leaf_buf_total +
           leaf_coords_total + leaf_values_total + shift_exp_total + pse_total;
}

// /**
//  * @brief Sum of the per-region sizes of the scratch layout returned by
//  *        `barnes_hut_scratch_size`. Kept as a static helper so the layout
//  *        computation is shared between the public sizer and the internal
//  *        partition routine.
//  *
//  * `n_thread_partitions` is the number of identical per-thread multipole
//  * scratch regions to allocate. Callers must set `settings.n_threads >= 1`
//  * (validated by `settings_valid`); the scratch is sized for exactly that
//  * many thread partitions, and the OpenMP parallel region is pinned to the
//  * same thread count via `num_threads(n_threads)`.
//  */
// static size_t barnes_hut_scratch_layout(unsigned n_sources, const barnes_hut_settings_t *settings,
//                                         unsigned n_thread_partitions, barnes_hut_scratch_sizes_t *out)
// {
//     if (n_sources == 0 || !settings_valid(settings))
//         return 0;
//
//     /* Layout (one contiguous byte stream, all offsets are byte offsets):
//      *   1. topo                — (8 * n_sources + 1) * sizeof(topo_node_t)
//      *   2. source_leaf_topo    — n_sources * sizeof(uint32_t)
//      *   3. source_leaf_real    — n_sources * sizeof(unsigned)
//      *   4. leaf_cur, leaf_nxt  — 2 * leaf_buf_size * sizeof(real_t) per thread
//      *   5. leaf_coords         — n_sources * 3 * sizeof(real_t) per thread
//      *   6. leaf_values         — n_sources * 3 * sizeof(real_t) per thread
//      */
//     const size_t topo_bytes = (8u * (size_t)n_sources + 1u) * sizeof(topo_node_t);
//     const size_t source_leaf_topo_bytes = (size_t)n_sources * sizeof(uint32_t);
//     const size_t source_leaf_real_bytes = (size_t)n_sources * sizeof(unsigned);
//
//     const size_t n_coeffs = multipole_num_coeffs(settings->order);
//     const size_t multipole_scratch = multipole_scratch_size(settings->order);
//     const size_t leaf_buf_size = (3u * n_coeffs > multipole_scratch ? 3u * n_coeffs : multipole_scratch);
//     const size_t leaf_buf_per_thread = 2u * leaf_buf_size * sizeof(real_t);
//     const size_t leaf_coords_per_thread = (size_t)n_sources * 3u * sizeof(real_t);
//     const size_t leaf_values_per_thread = (size_t)n_sources * 3u * sizeof(real_t);
//
//     const size_t leaf_buf_total = leaf_buf_per_thread * n_thread_partitions;
//     const size_t leaf_coords_total = leaf_coords_per_thread * n_thread_partitions;
//     const size_t leaf_values_total = leaf_values_per_thread * n_thread_partitions;
//
//     if (out)
//     {
//         out->size_topo = topo_bytes;
//         out->size_source_leaf_topo = source_leaf_topo_bytes;
//         out->size_source_leaf_real = source_leaf_real_bytes;
//         out->size_leaf_buf_per_thread = leaf_buf_per_thread;
//         out->size_leaf_buf_total = leaf_buf_total;
//         out->size_leaf_coords_total = leaf_coords_total;
//         out->size_leaf_values_total = leaf_values_total;
//     }
//
//     return topo_bytes + source_leaf_topo_bytes + source_leaf_real_bytes + leaf_buf_total + leaf_coords_total +
//            leaf_values_total;
// }

size_t barnes_hut_scratch_size(unsigned n_sources, unsigned n_threads, const barnes_hut_settings_t *settings)
{
    if (n_sources == 0 || !settings_valid(settings) || n_threads < 1)
        return 0;

    const barnes_hut_scratch_sizes_t sizes = barnes_hut_size_scratch(n_sources, settings);
    /* `n_threads` is guaranteed >= 1 by `settings_valid`. */
    return barnes_hut_total_scratch_size(sizes, n_threads);
}

/**
 * @brief Partition @p scratch_buffer into the regions described by
 *        the scratch layout.
 *
 * The partition order follows `barnes_hut_scratch_sizes_t`:
 *   topo | source_leaf_topo | source_leaf_real | leaf_cur | leaf_nxt
 *   | leaf_coords | leaf_values | shift_exp | pse
 */
barnes_hut_scratch_t barnes_hut_scratch_partition(unsigned n_thread_partitions, void *scratch_buffer,
                                                  barnes_hut_scratch_sizes_t scratch_sizes)
{
    const size_t s_topo = scratch_sizes.size_topo;
    const size_t s_leaf_topo = scratch_sizes.size_source_leaf_topo;
    const size_t s_leaf_real = scratch_sizes.size_source_leaf_real;
    const size_t s_coords = scratch_sizes.size_leaf_coords_per_thread * n_thread_partitions;
    const size_t s_values = scratch_sizes.size_leaf_values_per_thread * n_thread_partitions;
    const size_t s_shift_exp = scratch_sizes.size_shift_exp_per_thread * n_thread_partitions;
    const size_t s_pse = scratch_sizes.size_pse_per_thread * n_thread_partitions;
    /* size_leaf_buf_per_thread covers leaf_cur + leaf_nxt combined per thread.
     * Each individual buffer (leaf_cur or leaf_nxt) is half that per thread. */
    const size_t leaf_buf_each = scratch_sizes.size_leaf_buf_per_thread / 2;
    const size_t s_cur = (size_t)n_thread_partitions * leaf_buf_each;
    const size_t s_nxt = s_cur;

    uint8_t *bp = (uint8_t *)scratch_buffer;
    barnes_hut_scratch_t out = {0};
    out.topo = (topo_node_t *)bp;
    bp += s_topo;
    out.source_leaf_topo = (uint32_t *)bp;
    bp += s_leaf_topo;
    out.source_leaf_real = (unsigned *)bp;
    bp += s_leaf_real;
    out.leaf_cur = (real_t *)bp;
    bp += s_cur;
    out.leaf_nxt = (real_t *)bp;
    bp += s_nxt;
    out.leaf_coords = (real_t *)bp;
    bp += s_coords;
    out.leaf_values = (real_t *)bp;
    bp += s_values;
    out.shift_exp = (real_t *)bp;
    bp += s_shift_exp;
    out.pse = (real_t *)bp;
    bp += s_pse;
    out.n_thread_partitions = n_thread_partitions;
    return out;
}

/**
 * @brief Slice the per-thread leaf scratch into a single thread's view.
 */
static barnes_hut_leaf_scratch_t barnes_hut_leaf_scratch_for(const barnes_hut_scratch_t *scratch, unsigned thread_id,
                                                             unsigned n_sources, unsigned order)
{
    const size_t n_coeffs = multipole_num_coeffs(order);
    const size_t multipole_scratch = multipole_scratch_size(order);
    const size_t leaf_buf_size = (3u * n_coeffs > multipole_scratch ? 3u * n_coeffs : multipole_scratch);
    const size_t coords_per_thread = (size_t)n_sources * 3u;
    barnes_hut_leaf_scratch_t out = {0};
    out.leaf_cur = scratch->leaf_cur + (size_t)thread_id * leaf_buf_size;
    out.leaf_nxt = scratch->leaf_nxt + (size_t)thread_id * leaf_buf_size;
    out.leaf_coords = scratch->leaf_coords + (size_t)thread_id * coords_per_thread;
    out.leaf_values = scratch->leaf_values + (size_t)thread_id * coords_per_thread;
    return out;
}

/* ------------------------------------------------------------------ */
/* Public API — work buffer sizing                                    */
/* ------------------------------------------------------------------ */

barnes_hut_work_sizes_t barnes_hut_size_work_buffer(unsigned n_sources,
                                                    const barnes_hut_settings_t CVL_ARRAY_ARG(settings, restrict),
                                                    barnes_hut_count_res_t count_pass_res)
{
    const unsigned n_internal = count_pass_res.n_internal;
    const unsigned n_multipole = count_pass_res.n_multipole_leaves;
    const unsigned n_particle = count_pass_res.n_particle_leaves;
    const unsigned n_total = n_internal + n_multipole + n_particle;

    const size_t nodes_bytes = (size_t)n_total * sizeof(bh_node_t);
    const size_t particle_order_bytes = (size_t)n_sources * sizeof(unsigned);
    const size_t multipole_coeffs_bytes =
        (size_t)(n_internal + n_multipole) * 3u * multipole_num_coeffs(settings->order) * sizeof(real_t);
    const size_t topo_to_real_bytes = (size_t)n_total * sizeof(uint32_t);
    const size_t mp_slices_bytes = (size_t)n_total * sizeof(real_t *);

    return (barnes_hut_work_sizes_t){
        .nodes_bytes = nodes_bytes,
        .particle_order_bytes = particle_order_bytes,
        .multipole_coeffs_bytes = multipole_coeffs_bytes,
        .topo_to_real_bytes = topo_to_real_bytes,
        .mp_slices_bytes = mp_slices_bytes,
    };
}

/**
 * @brief Sum all per-region byte sizes to obtain the total work buffer size.
 *
 * @param sizes  Per-region sizes from `barnes_hut_size_work_buffer`.
 * @return Total work buffer size in bytes.
 */
size_t barnes_hut_total_work_size(barnes_hut_work_sizes_t sizes)
{
    return sizes.nodes_bytes + sizes.particle_order_bytes + sizes.multipole_coeffs_bytes + sizes.topo_to_real_bytes +
           sizes.mp_slices_bytes;
}

/* ------------------------------------------------------------------ */
/* Public API — insert pass                                          */
/* ------------------------------------------------------------------ */

/**
 * @brief Walk the topo array in DFS pre-order, assigning each topo node a
 *        bh_node_t index in `nodes[]` and recording the mapping in
 *        `topo_to_real`. Also lays out the multipole coefficient slice for
 *        multipole-bearing nodes.
 *
 * Because `bh_node_t::data` is a union between `internal` and `mp`,
 * `data.internal.children[k]` and `data.mp.coeffs_x/y/z` overlap. For
 * internal nodes we therefore stash the multipole slice pointer in the
 * parallel `mp_slices_out[]` array rather than in the node struct itself.
 * Multipole leaves use `data.mp` directly (no children to keep).
 *
 * Fills `nodes[]` for the indices it touches.
 */
static void materialize_tree(const topo_node_t CVL_ARRAY_ARG(topo, restrict), uint32_t n_topo_nodes,
                             const barnes_hut_settings_t *settings,
                             uint32_t CVL_ARRAY_ARG(topo_to_real, restrict n_topo_nodes),
                             bh_node_t CVL_ARRAY_ARG(nodes, restrict n_topo_nodes), real_t *multipole_coeffs,
                             real_t *CVL_ARRAY_ARG(mp_slices_out, restrict n_topo_nodes))
{
    real_t *cursor = multipole_coeffs;
    // unsigned n_multipole_slices = 0;
    uint32_t next_real = 0;

    /* Iterative DFS using a small stack (bounded by depth * 8 * 2 = safe). */
    enum
    {
        DFS_STACK_MAX = 1024
    };
    uint32_t stack[DFS_STACK_MAX];
    unsigned depth_stack[DFS_STACK_MAX];
    int sp = 0;
    stack[sp] = 0;
    depth_stack[sp] = 0;
    sp += 1;

    while (sp > 0)
    {
        sp -= 1;
        const uint32_t ti = stack[sp];
        const unsigned depth = depth_stack[sp];
        const topo_node_t *tn = &topo[ti];

        topo_to_real[ti] = next_real;
        bh_node_t *node = &nodes[next_real];
        next_real += 1;

        node->depth = depth;
        node->center = tn->center;
        node->half_size = tn->half_size;
        node->particle_begin = 0;
        node->particle_count = 0;
        mp_slices_out[next_real - 1] = NULL; /* default */

        if (tn->is_internal)
        {
            node->kind = BH_NODE_INTERNAL;
            /* Allocate the internal node's multipole slice via the side
             * table so we don't clobber children[] pointers. */
            real_t *slice = cursor;
            cursor += 3u * multipole_num_coeffs(settings->order);
            mp_slices_out[next_real - 1] = slice;
            for (int k = 0; k < 8; ++k)
            {
                if (tn->children[k] < 0)
                {
                    node->data.internal.children[k] = NULL;
                }
                else
                {
                    if ((size_t)sp + 1 > DFS_STACK_MAX)
                        return;
                    stack[sp] = (uint32_t)tn->children[k];
                    depth_stack[sp] = depth + 1;
                    sp += 1;
                    /* Placeholder; resolved by the second pass below. */
                    node->data.internal.children[k] = (bh_node_t *)(uintptr_t)((uint32_t)tn->children[k] + 1u);
                }
            }
        }
        else if (should_be_multipole(tn->particle_count, settings))
        {
            node->kind = BH_NODE_MULTIPOLE;
            node->data.mp.order = settings->order;
            node->data.mp.center = tn->center;
            node->data.mp.coeffs_x = cursor;
            cursor += multipole_num_coeffs(settings->order);
            node->data.mp.coeffs_y = cursor;
            cursor += multipole_num_coeffs(settings->order);
            node->data.mp.coeffs_z = cursor;
            cursor += multipole_num_coeffs(settings->order);
            mp_slices_out[next_real - 1] = node->data.mp.coeffs_x;
        }
        else
        {
            node->kind = BH_NODE_PARTICLE;
        }
    }

    /* Second pass: resolve child pointers that hold topo-index sentinels.
     * Parallelised: each internal node writes to its own children[] slot,
     * so threads touch disjoint cache lines. */
#pragma omp parallel for default(none) shared(n_topo_nodes, topo, topo_to_real, nodes) schedule(static, 256)
    for (uint32_t ti = 0; ti < n_topo_nodes; ++ti)
    {
        const topo_node_t *tn = &topo[ti];
        if (!tn->is_internal)
            continue;
        bh_node_t *node = &nodes[topo_to_real[ti]];
        for (int k = 0; k < 8; ++k)
        {
            if (tn->children[k] < 0)
                continue;
            const uint32_t child_real = topo_to_real[tn->children[k]];
            node->data.internal.children[k] = &nodes[child_real];
        }
    }

    // if (out_n_multipole_slices != NULL)
    //     *out_n_multipole_slices = n_multipole_slices;
    // return next_real;
}

/**
 * @brief Resolve the leaf-id (in real-tree form) for each source, using a
 *        depth-bounded descent through the bh_node_t tree.
 *
 * Parallelised via OpenMP: each thread updates distinct leaf counters using
 * an atomic increment on `nodes[idx].particle_count`. The
 * `source_leaf_real[]` writes are per-thread (different `i`).
 */
static void descend_for_each_source(unsigned n_sources, const real3_t CVL_ARRAY_ARG(sources_coords, restrict n_sources),
                                    uint32_t n_real_nodes, bh_node_t CVL_ARRAY_ARG(nodes, restrict n_real_nodes),
                                    unsigned *source_leaf_real, unsigned n_threads)
{
#pragma omp parallel for default(none) shared(n_sources, sources_coords, n_real_nodes, nodes, source_leaf_real)        \
    schedule(static) num_threads(n_threads)
    for (unsigned i = 0; i < n_sources; ++i)
    {
        uint32_t idx = 0;
        while (nodes[idx].kind == BH_NODE_INTERNAL)
        {
            const real3_t p = sources_coords[i];
            const real3_t c = nodes[idx].center;
            const unsigned oct =
                (unsigned)(p.x >= c.x) * 1u + (unsigned)(p.y >= c.y) * 2u + (unsigned)(p.z >= c.z) * 4u;
            const bh_node_t *child = nodes[idx].data.internal.children[oct];
            if (child == NULL)
            {
                /* Topo/bh-node inconsistency: an octant that count_pass
                 * descended into has no materialized child. The safest
                 * recovery is to land the source at the deepest node we
                 * reached (an internal node with no children for this
                 * octant is itself effectively a missing leaf).
                 * compute_particle_begins will skip internals, so this
                 * source will be ignored in particle_order. */
                break;
            }
            idx = (uint32_t)(child - nodes);
        }
        source_leaf_real[i] = idx;
        if (nodes[idx].kind != BH_NODE_INTERNAL)
        {
#pragma omp atomic
            nodes[idx].particle_count += 1;
        }
    }
}

/**
 * @brief Combined pass: compute particle_begin prefix sums with count reset,
 *        count multipole leaves, and (optionally) build depth-range table.
 *
 * Replaces three separate O(nodes) walks with one. The depth range output
 * is used by the upward sweep for slice-based iteration.
 *
 * @param n_nodes     Number of nodes.
 * @param nodes       Node array.
 * @param max_depth   Maximum depth (from count_res), or 0 to skip ranges.
 * @param depth_start Output array [max_depth+2], or NULL.
 * @param depth_end   Output array [max_depth+2], or NULL.
 * @return Number of MULTIPOLE leaves.
 */
static unsigned compute_node_metadata(uint32_t n_nodes, bh_node_t CVL_ARRAY_ARG(nodes, restrict n_nodes),
                                      unsigned max_depth, unsigned CVL_ARRAY_ARG(depth_start, restrict),
                                      unsigned CVL_ARRAY_ARG(depth_end, restrict))
{
    unsigned n_mp_leaves = 0;
    unsigned cursor = 0;

    if (depth_start && depth_end)
    {
        for (unsigned d = 0; d <= max_depth + 1; ++d)
        {
            depth_start[d] = n_nodes;
            depth_end[d] = n_nodes;
        }
    }

    for (uint32_t i = 0; i < n_nodes; ++i)
    {
        const unsigned d = nodes[i].depth;
        if (depth_start)
        {
            if (i < depth_start[d])
                depth_start[d] = i;
            depth_end[d] = i + 1;
        }

        if (nodes[i].kind != BH_NODE_INTERNAL)
        {
            nodes[i].particle_begin = cursor;
            cursor += nodes[i].particle_count;
            nodes[i].particle_count = 0;

            if (nodes[i].kind == BH_NODE_MULTIPOLE)
                n_mp_leaves += 1;
        }
    }
    return n_mp_leaves;
}

static barnes_hut_work_t partition_work_buffer(barnes_hut_work_sizes_t work_sizes, void *buffer)
{
    uint8_t *bp = (uint8_t *)buffer;
    bh_node_t *nodes = (bh_node_t *)bp;
    bp += work_sizes.nodes_bytes;
    unsigned *particle_order = (unsigned *)bp;
    bp += work_sizes.particle_order_bytes;
    real_t *multipole_coeffs = (real_t *)bp;
    bp += work_sizes.multipole_coeffs_bytes;
    uint32_t *topo_to_real = (uint32_t *)bp;
    bp += work_sizes.topo_to_real_bytes;
    real_t **mp_slices = (real_t **)bp;
    // bp += work_sizes.mp_slices_bytes; // not needed
    return (barnes_hut_work_t){
        .nodes = nodes,
        .particle_order = particle_order,
        .multipole_coeffs = multipole_coeffs,
        .topo_to_real = topo_to_real,
        .mp_slices = mp_slices,
    };
}

static bool barnes_hut_tree_build_multipoles(
    const unsigned n_topo_nodes, bh_node_t CVL_ARRAY_ARG(nodes, restrict n_topo_nodes),
    const unsigned *restrict particle_order, const real3_t CVL_ARRAY_ARG(sources_coords, restrict),
    const real3_t CVL_ARRAY_ARG(sources_values, restrict), const barnes_hut_settings_t *settings,
    barnes_hut_scratch_t scratch, size_t leaf_buf_size, unsigned n_sources, unsigned n_threads,
    real_t *CVL_ARRAY_ARG(mp_slices, restrict n_topo_nodes), const unsigned n_mp_leaves,
    uint32_t CVL_ARRAY_ARG(mp_leaf_indices, restrict n_mp_leaves))
{
    bool leaf_ok = true;

    // TODO: check if this can be parallelized (and if it is worth it)
    {
        unsigned k = 0;
        for (uint32_t i = 0; i < n_topo_nodes; ++i)
        {
            if (nodes[i].kind == BH_NODE_MULTIPOLE)
            {
                mp_leaf_indices[k] = i;
                k += 1;
            }
        }
    }

    unsigned worker_id = 0;
    (void)worker_id;

#pragma omp parallel default(none)                                                                                     \
    shared(n_mp_leaves, mp_leaf_indices, nodes, particle_order, sources_coords, sources_values, settings, scratch,     \
               leaf_buf_size, n_sources, leaf_ok, mp_slices, worker_id) num_threads(n_threads)
    {
        // const int tid = omp_get_thread_num();
        unsigned tid;
#pragma omp atomic capture
        tid = worker_id++;
        const barnes_hut_leaf_scratch_t leaf =
            barnes_hut_leaf_scratch_for(&scratch, (unsigned)tid, n_sources, settings->order);
#pragma omp for reduction(&& : leaf_ok) schedule(static)
        for (unsigned ml = 0; ml < n_mp_leaves; ++ml)
        {
            const uint32_t i = mp_leaf_indices[ml];
            const size_t n_particles = nodes[i].particle_count;

            /* Pack sources into the leaf buffer. */
#pragma omp simd
            for (unsigned k = 0; k < n_particles; ++k)
            {
                const unsigned src = particle_order[nodes[i].particle_begin + k];
                leaf.leaf_coords[3u * k + 0] = sources_coords[src].x;
                leaf.leaf_coords[3u * k + 1] = sources_coords[src].y;
                leaf.leaf_coords[3u * k + 2] = sources_coords[src].z;
                leaf.leaf_values[3u * k + 0] = sources_values[src].x;
                leaf.leaf_values[3u * k + 1] = sources_values[src].y;
                leaf.leaf_values[3u * k + 2] = sources_values[src].z;
            }

            const bool mp_ok =
                multipole_create(settings->order, (unsigned)leaf_buf_size, mp_slices[i], nodes[i].center,
                                 (unsigned)n_particles, (const real3_t *)leaf.leaf_coords,
                                 (const real3_t *)leaf.leaf_values, leaf.leaf_cur, leaf.leaf_nxt, &nodes[i].data.mp);
            /* Force gcc to keep nodes[i].kind live across the call —
             * the strict-aliasing rules can otherwise let it spill a
             * temporary into the kind slot via the inactive-union
             * access patterns. */
            if (nodes[i].kind != BH_NODE_MULTIPOLE)
                leaf_ok = false;

            if (!mp_ok)
                leaf_ok = false;
        }
    }
    return leaf_ok;
}

/* ------------------------------------------------------------------ */
/* Leaf centroid computation                                          */
/* ------------------------------------------------------------------ */

/**
 * @brief Compute |Γ|-weighted centroids for all leaf nodes.
 *
 * Replaces each leaf's geometric center (set from topo subdivision) with
 * the source-magnitude-weighted centroid.  Internal-node centroids are
 * computed later in the upward sweep (see `upward_sweep_level`).
 *
 * For leaves where all sources have zero magnitude, falls back to the
 * geometric center (which is fine: if all |Γ| = 0 the field is zero).
 *
 * Updates `nodes[i].data.mp.center` alongside `nodes[i].center` for
 * MULTIPOLE leaves so both are consistent.
 *
 * @param n_nodes         Number of nodes.
 * @param nodes           Node array (read/write).
 * @param particle_order  Source permutation (leaf -> source index).
 * @param sources_coords  Source coordinates.
 * @param sources_values  Source strengths.
 * @param n_threads       OpenMP thread count.
 */
static void barnes_hut_compute_leaf_centers(unsigned n_nodes, bh_node_t CVL_ARRAY_ARG(nodes, restrict n_nodes),
                                            const unsigned CVL_ARRAY_ARG(particle_order, restrict),
                                            const real3_t CVL_ARRAY_ARG(sources_coords, restrict),
                                            const real3_t CVL_ARRAY_ARG(sources_values, restrict), unsigned n_threads)
{
#pragma omp parallel for default(none) shared(n_nodes, nodes, particle_order, sources_coords, sources_values)          \
    schedule(static) num_threads(n_threads)
    for (uint32_t i = 0; i < n_nodes; ++i)
    {
        if (nodes[i].kind == BH_NODE_INTERNAL)
            continue;

        const unsigned begin = nodes[i].particle_begin;
        const unsigned n_particles = nodes[i].particle_count;
        if (n_particles == 0)
            continue;
        const unsigned end = begin + n_particles;

        real_t cx = 0, cy = 0, cz = 0;
        real_t total_weight = 0.0;

        for (unsigned k = begin; k < end; ++k)
        {
            const unsigned src = particle_order[k];
            const real_t w = real3_mag(sources_values[src]);
            cx += sources_coords[src].x * w;
            cy += sources_coords[src].y * w;
            cz += sources_coords[src].z * w;
            total_weight += w;
        }

        if (total_weight > 0.0)
        {
            nodes[i].center.x = cx / total_weight;
            nodes[i].center.y = cy / total_weight;
            nodes[i].center.z = cz / total_weight;
        }
        /* else: all-zero strength — geometric center is fine, keep it. */

        /* Keep multipole leaf's mp.center in sync with node center. */
        if (nodes[i].kind == BH_NODE_MULTIPOLE)
        {
            nodes[i].data.mp.center = nodes[i].center;
        }
    }
}

/* ------------------------------------------------------------------ */
/* Upward sweep helpers                                               */
/* ------------------------------------------------------------------ */

/**
 * @brief Build a compact depth-range table for the node array.
 *
 * Nodes are in DFS pre-order, so each depth forms a contiguous index range
 * [depth_start[d], depth_end[d]). The table has @p max_depth + 2 entries
 * (entries max_depth+1 and max_depth+2 are sentinel-zeroed for safety).
 *
 * @param n_nodes      Number of nodes in @p nodes.
 * @param nodes        Node array (DFS pre-order).
 * @param max_depth    The maximum depth in the tree.
 * @param depth_start  Output array of size max_depth+2, populated with start
 *                     indices.
 * @param depth_end    Output array of size max_depth+2, populated with
 *                     one-past-end indices.
 */
static void compute_depth_ranges(unsigned n_nodes, const bh_node_t CVL_ARRAY_ARG(nodes, restrict n_nodes),
                                 unsigned max_depth, unsigned CVL_ARRAY_ARG(depth_start, restrict max_depth + 2),
                                 unsigned CVL_ARRAY_ARG(depth_end, restrict max_depth + 2))
{
    for (unsigned d = 0; d <= max_depth + 1; ++d)
    {
        depth_start[d] = n_nodes; /* sentinel: empty range */
        depth_end[d] = n_nodes;
    }

    for (unsigned i = 0; i < n_nodes; ++i)
    {
        const unsigned d = nodes[i].depth;
        if (i < depth_start[d])
            depth_start[d] = i;
        depth_end[d] = i + 1;
    }
}

/**
 * @brief Process one depth level of the upward sweep for internal nodes.
 *
 * Shifts child multipoles into their parent and (for particle-leaf children)
 * updates the parent multipole directly. Only nodes in [depth_start,
 * depth_end) with kind == BH_NODE_INTERNAL are processed — no full-scan
 * filtering needed.
 *
 * @param depth_start   First node index at this depth.
 * @param depth_end     One-past-last node index at this depth.
 * @param nodes         Node array.
 * @param order         Multipole order.
 * @param n_coeffs      Number of coefficients per component.
 * @param mp_slices     Per-node multipole slice pointers.
 * @param work_order    Work order for add_shift.
 * @param shift_exp     Shift expansion scratch (per-thread regions inside).
 * @param pse           Polynomial scratch (per-thread regions inside).
 * @param shift_stride  Per-thread byte stride in shift_exp.
 * @param pse_stride    Per-thread byte stride in pse.
 * @param particle_order  Source-index order array.
 * @param sources_coords  Source coordinates.
 * @param sources_values  Source strengths.
 * @param scratch       Scratch buffer (for per-thread leaf_cur/nxt).
 * @param leaf_stride   Per-thread stride in leaf_cur/nxt.
 * @param n_threads     OpenMP thread count.
 */
static void upward_sweep_level(unsigned depth_start, unsigned depth_end, bh_node_t CVL_ARRAY_ARG(nodes, restrict),
                               unsigned order, size_t n_coeffs, real_t *CVL_ARRAY_ARG(mp_slices, restrict),
                               unsigned work_order, real_t CVL_ARRAY_ARG(shift_exp, restrict),
                               real_t CVL_ARRAY_ARG(pse, restrict), size_t shift_stride, size_t pse_stride,
                               const unsigned CVL_ARRAY_ARG(particle_order, restrict),
                               const real3_t CVL_ARRAY_ARG(sources_coords, restrict),
                               const real3_t CVL_ARRAY_ARG(sources_values, restrict),
                               const barnes_hut_scratch_t *scratch, size_t leaf_stride, unsigned n_threads)
{
#pragma omp parallel for default(none)                                                                                 \
    shared(depth_start, depth_end, nodes, order, n_coeffs, mp_slices, work_order, shift_exp, pse, shift_stride,        \
               pse_stride, particle_order, sources_coords, sources_values, scratch, leaf_stride) schedule(dynamic, 16) \
    num_threads(n_threads)
    for (uint32_t i = depth_start; i < depth_end; ++i)
    {
        if (nodes[i].kind != BH_NODE_INTERNAL)
            continue;

        /* --------------------------------------------------------------- */
        /* Step 1: Compute |Γ|-weighted centroid from children.            */
        /* --------------------------------------------------------------- */
        {
            real_t cx = 0, cy = 0, cz = 0;
            real_t total_weight = 0.0;
            unsigned n_children = 0;

            for (int oct = 0; oct < 8; ++oct)
            {
                const bh_node_t *child = nodes[i].data.internal.children[oct];
                if (child == NULL)
                    continue;
                n_children += 1;
                /* Weight = |first coefficient| for MULTIPOLE/INTERNAL (first
                 * coeff = Σ|Γ|) or Σ|Γ_i| from particle data. */
                real_t w;
                if (child->kind == BH_NODE_PARTICLE)
                {
                    w = 0.0;
                    for (unsigned k = child->particle_begin; k < child->particle_begin + child->particle_count; ++k)
                    {
                        const unsigned src = particle_order[k];
                        w += real3_mag(sources_values[src]);
                    }
                }
                else
                {
                    /* INTERNAL or MULTIPOLE child: first coefficient = total vector. */
                    real_t *cs = mp_slices[child - nodes];
                    if (cs == NULL)
                        continue;
                    w = real3_mag((real3_t){.x = cs[0], .y = cs[n_coeffs], .z = cs[2u * n_coeffs]});
                }
                cx += child->center.x * w;
                cy += child->center.y * w;
                cz += child->center.z * w;
                total_weight += w;
            }

            if (total_weight > 0.0)
            {
                nodes[i].center.x = cx / total_weight;
                nodes[i].center.y = cy / total_weight;
                nodes[i].center.z = cz / total_weight;
            }
            else if (n_children > 0)
            {
                /* Uniform average of child centers (geometric fallback). */
                real_t ax = 0, ay = 0, az = 0;
                for (int oct = 0; oct < 8; ++oct)
                {
                    const bh_node_t *child = nodes[i].data.internal.children[oct];
                    if (child == NULL)
                        continue;
                    ax += child->center.x;
                    ay += child->center.y;
                    az += child->center.z;
                }
                const real_t inv_n = 1.0 / (real_t)n_children;
                nodes[i].center.x = ax * inv_n;
                nodes[i].center.y = ay * inv_n;
                nodes[i].center.z = az * inv_n;
            }
            /* else: no children (degenerate) — geometric center stays. */
        }

        /* Zero the internal node's slice and prepare a multipole_t. */
        real_t *slice = mp_slices[i];
        memset(slice, 0, 3u * n_coeffs * sizeof(real_t));
        const multipole_t internal_mp = {.order = order,
                                         .center = nodes[i].center,
                                         .coeffs_x = slice,
                                         .coeffs_y = slice + n_coeffs,
                                         .coeffs_z = slice + 2u * n_coeffs};

        /* Per-thread scratch for multipole_update and multipole_add_shift. */
        const int tid = omp_get_thread_num();
        real_t *particle_cur = scratch->leaf_cur + (size_t)tid * leaf_stride;
        real_t *particle_nxt = scratch->leaf_nxt + (size_t)tid * leaf_stride;
        real_t *my_shift_exp = shift_exp + (size_t)tid * shift_stride;
        real_t *my_pse = pse + (size_t)tid * pse_stride;

        for (int oct = 0; oct < 8; ++oct)
        {
            bh_node_t *child = nodes[i].data.internal.children[oct];
            if (child == NULL)
                continue;

            if (child->kind == BH_NODE_PARTICLE)
            {
                for (unsigned k = child->particle_begin; k < child->particle_begin + child->particle_count; ++k)
                {
                    const unsigned src = particle_order[k];
                    multipole_update(&internal_mp, nodes[i].center, sources_coords[src], sources_values[src],
                                     particle_cur, particle_nxt);
                }
            }
            else
            {

                /* For BH_NODE_INTERNAL and BH_NODE_MULTIPOLE children the
                 * multipole slice is in mp_slices[] (side table avoids
                 * clobbering children[] in the union). */
                real_t *child_slice = mp_slices[child - nodes];
                if (child_slice == NULL)
                    continue;
                multipole_t child_mp = {.order = order,
                                        .center = child->center,
                                        .coeffs_x = child_slice,
                                        .coeffs_y = child_slice + n_coeffs,
                                        .coeffs_z = child_slice + 2u * n_coeffs};
                multipole_add_shift(&child_mp, &internal_mp, work_order, my_shift_exp, my_pse);
            }
        }
    }
}

/**
 * @brief Finalise the tree handle after the downward pass and multipole build.
 *
 * Runs the upward sweep (aggregating child multipoles into parents via
 * multipole_add_shift), then fills and returns a barnes_hut_tree_t handle
 * pointing into the caller-provided work buffers.
 *
 * @param count_res        Count-pass results (node topology counts).
 * @param n_sources        Number of source points.
 * @param n_threads        Number of OpenMP threads (>= 1).
 * @param settings         Build settings.
 * @param work_buffers     Partitioned work buffer views.
 * @param work_order       Internal expansion order for multipole_add_shift.
 * @param sources_coords   Source coordinates.
 * @param sources_values   Source strengths.
 * @param nodes            Materialised bh_node_t array.
 * @param particle_order   Per-leaf source index ordering.
 * @param multipole_coeffs Contiguous multipole coefficient storage.
 * @param mp_slices        Per-node multipole slice pointers.
 * @param scratch          Scratch buffer (per-thread temp storage).
 * @return Populated tree handle (views into caller buffers, no allocation).
 */
static barnes_hut_tree_t barnes_hut_tree_complete(
    const barnes_hut_count_res_t count_res, const unsigned n_sources, const barnes_hut_settings_t *settings,
    unsigned work_order, const real3_t CVL_ARRAY_ARG(sources_coords, restrict),
    const real3_t CVL_ARRAY_ARG(sources_values, restrict), bh_node_t CVL_ARRAY_ARG(nodes, restrict),
    unsigned CVL_ARRAY_ARG(particle_order, restrict), real_t CVL_ARRAY_ARG(multipole_coeffs, restrict),
    real_t *CVL_ARRAY_ARG(mp_slices, restrict), const barnes_hut_scratch_t *scratch)
{
    const unsigned n_internal = count_res.n_internal;
    const unsigned n_multipole = count_res.n_multipole_leaves;
    const unsigned n_particle = count_res.n_particle_leaves;
    const unsigned max_depth = count_res.max_depth;
    const size_t n_topo_nodes = (size_t)n_internal + (size_t)n_multipole + (size_t)n_particle;
    const unsigned order = settings->order;
    const size_t n_coeffs = multipole_num_coeffs(order);

    real_t *shift_exp = scratch->shift_exp;
    real_t *pse = scratch->pse;

    /* Per-thread scratch stride for multipole_update. */
    const size_t leaf_stride =
        (3u * n_coeffs > multipole_scratch_size(order)) ? (3u * n_coeffs) : multipole_scratch_size(order);

    /* Per-thread stride for shift_exp and pse (sized for n_threads in the
     * work buffer — each thread writes its own region). */
    const size_t shift_stride = 3u * (size_t)(work_order + 1) * (size_t)(work_order + 1);
    const size_t pse_stride = 2u * multipole_num_coeffs(work_order);

    /* Build depth-range table for slice-based iteration (avoids O(all-nodes)
     * scan at each depth level — only the internal node at each depth is
     * visited). Tree depth is stored in uint8_t, so 256 covers all. */
    unsigned depth_start[256], depth_end[256];
    compute_depth_ranges((unsigned)n_topo_nodes, nodes, max_depth, depth_start, depth_end);

    /* --- Upward sweep: aggregate all children into internal nodes. --- */
    for (unsigned d = max_depth;; --d)
    {
        upward_sweep_level(depth_start[d], depth_end[d], nodes, order, n_coeffs, mp_slices, work_order, shift_exp, pse,
                           shift_stride, pse_stride, particle_order, sources_coords, sources_values, scratch,
                           leaf_stride, scratch->n_thread_partitions);
        if (d == 0)
            break;
    }

    barnes_hut_tree_t out = {0};
    out.settings = *settings;
    out.root_center = nodes[0].center;
    out.root_half_size = nodes[0].half_size;
    out.n_sources = n_sources;
    out.n_nodes = n_topo_nodes;
    out.n_internal = n_internal;
    out.n_multipole_leaves = n_multipole;
    out.n_particle_leaves = n_particle;
    out.max_depth_reached = max_depth;
    out.nodes = nodes;
    out.particle_order = particle_order;
    out.multipole_coeffs = multipole_coeffs;
    out.mp_slices = mp_slices;

    return out;
}

/**
 * TODO: split this into several functions so none need to allocate and just ask for buffer sizes.
 */
bool barnes_hut_tree_insert(unsigned n_sources, unsigned n_threads,
                            const real3_t CVL_ARRAY_ARG(sources_coords, restrict n_sources),
                            const real3_t CVL_ARRAY_ARG(sources_values, restrict n_sources),
                            const barnes_hut_settings_t CVL_ARRAY_ARG(settings, restrict), void *scratch_buffer,
                            size_t scratch_size, const allocator_t *allocator, void *buffer, size_t buffer_size,
                            barnes_hut_tree_t *out)
{
    if (!buffer || !out || !scratch_buffer)
        return false;
    if (n_sources == 0 || !settings_valid(settings))
        return false;
    if (sources_coords == NULL || sources_values == NULL)
        return false;

    /* --- Size and partition the scratch buffer. --- */
    const barnes_hut_scratch_sizes_t scratch_sizes = barnes_hut_size_scratch(n_sources, settings);
    const size_t needed_scratch = barnes_hut_total_scratch_size(scratch_sizes, n_threads);
    if (scratch_size < needed_scratch)
        return false;

    const barnes_hut_scratch_t scratch = barnes_hut_scratch_partition(n_threads, scratch_buffer, scratch_sizes);

    const unsigned work_order = resolve_work_order(settings);

    /* Zero the topo scratch — count_pass reads `is_internal` and `children[]`
     * before writing them, so leaving garbage would corrupt the descent. */
    const size_t topo_bytes = (8u * (size_t)n_sources + 1u) * sizeof(topo_node_t);
    memset(scratch.topo, 0, topo_bytes);

    /* --- Run the count pass to learn the topology. --- */
    const barnes_hut_count_res_t count_res =
        barnes_hut_count_pass(n_sources, sources_coords, settings, scratch.topo, scratch.source_leaf_topo);

    const unsigned n_internal = count_res.n_internal;
    const unsigned n_multipole = count_res.n_multipole_leaves;
    const unsigned n_particle = count_res.n_particle_leaves;
    const unsigned max_depth = count_res.max_depth;
    const uint32_t n_topo_nodes = (uint32_t)(n_internal + n_multipole + n_particle);
    if (n_topo_nodes == 0)
        return false;

    /* --- Compute partition layout and validate buffer_size. --- */
    const barnes_hut_work_sizes_t work_sizes = barnes_hut_size_work_buffer(n_sources, settings, count_res);
    if (buffer_size < barnes_hut_total_work_size(work_sizes))
        return false;

    /* --- Set up views into the caller buffer. --- */
    const barnes_hut_work_t work_buffers = partition_work_buffer(work_sizes, buffer);

    bh_node_t *nodes = work_buffers.nodes;
    unsigned *particle_order = work_buffers.particle_order;
    real_t *multipole_coeffs = work_buffers.multipole_coeffs;

    /* --- Allocate the count-pass-output-sized residual scratch through the
     * allocator. `topo_to_real` maps each topo node to its bh_node_t index;
     * `mp_slices` carries each multipole-bearing node's coefficient slice
     * pointer. Both are sized from `n_topo_nodes` which is only known after
     * the count pass, hence the fallback to the allocator. --- */
    uint32_t *topo_to_real = work_buffers.topo_to_real;
    real_t **mp_slices = work_buffers.mp_slices;

    /* --- Materialize the bh_node_t tree from the topo array. --- */
    materialize_tree(scratch.topo, n_topo_nodes, settings, topo_to_real, nodes, multipole_coeffs, mp_slices);

    /* --- Descend each source through the bh_node_t tree to count per-leaf. --- */
    descend_for_each_source(n_sources, sources_coords, n_topo_nodes, nodes, scratch.source_leaf_real, n_threads);

    /* --- Assign particle ranges, reset counts, and count multipole leaves
     * in one combined pass (was three separate O(nodes) walks). --- */
    const unsigned n_mp_leaves = compute_node_metadata(n_topo_nodes, nodes, 0, NULL, NULL);

    /* --- Fill particle_order[]. --- */
    // NOTE: Can parallelise this using atomic capture. Schedule is static, with big chunks, because typically the
    // leaves close in index are close in space, so by giving them large chunks, we have less of a chance of triggering
    // atomics.
    // TODO: set the chunk size to something like N/num_threads, but make sure it's at least 1. This will help with load
    // balancing.
#pragma omp parallel for default(none) shared(n_sources, scratch, nodes, particle_order) schedule(static)              \
    num_threads(n_threads)
    for (unsigned i = 0; i < n_sources; ++i)
    {
        const unsigned leaf_idx = scratch.source_leaf_real[i];
        bh_node_t *leaf = nodes + leaf_idx;
        unsigned cnt;
#pragma omp atomic capture
        {
            cnt = leaf->particle_count;
            leaf->particle_count += 1;
        }
        const unsigned slot = leaf->particle_begin + cnt;
        particle_order[slot] = i;
    }

    /* Compute |-weighted centroids for leaves before building
     * multipoles — the multipole coefficients must use the weighted center
     * as expansion origin.  Internal-node centroids are computed during
     * the upward sweep. */
    barnes_hut_compute_leaf_centers(n_topo_nodes, nodes, particle_order, sources_coords, sources_values, n_threads);

    /* --- Build multipoles for each multipole-bearing leaf. Parallelised:
     * the scratch is pre-partitioned into n_threads independent leaf
     * scratch slices; thread `t` writes only into slice `t`. --- */

    bool leaf_ok = true;
    const size_t n_coeffs = multipole_num_coeffs(settings->order);
    if (n_mp_leaves > 0)
    {
        const size_t multipole_scratch = multipole_scratch_size(settings->order);

        const size_t leaf_buf_size = (3u * n_coeffs > multipole_scratch ? 3u * n_coeffs : multipole_scratch);
        uint32_t *mp_leaf_indices = (uint32_t *)bh_alloc(allocator, (size_t)n_mp_leaves * sizeof(uint32_t));
        if (mp_leaf_indices == NULL)
        {
            return false;
        }
        {
            unsigned k = 0;
            for (uint32_t i = 0; i < n_topo_nodes; ++i)
            {
                if (nodes[i].kind == BH_NODE_MULTIPOLE)
                {
                    mp_leaf_indices[k] = i;
                    k += 1;
                }
            }
        }

#pragma omp parallel default(none)                                                                                     \
    shared(n_mp_leaves, mp_leaf_indices, nodes, particle_order, sources_coords, sources_values, settings, scratch,     \
               leaf_buf_size, n_sources, leaf_ok, mp_slices) num_threads(n_threads)
        {
            const int tid = omp_get_thread_num();
            const barnes_hut_leaf_scratch_t leaf =
                barnes_hut_leaf_scratch_for(&scratch, (unsigned)tid, n_sources, settings->order);
#pragma omp for reduction(&& : leaf_ok) schedule(static)
            for (unsigned ml = 0; ml < n_mp_leaves; ++ml)
            {
                const uint32_t i = mp_leaf_indices[ml];
                const size_t n_particles = nodes[i].particle_count;

                /* Pack sources into the leaf buffer. */
#pragma omp simd
                for (unsigned k = 0; k < n_particles; ++k)
                {
                    const unsigned src = particle_order[nodes[i].particle_begin + k];
                    leaf.leaf_coords[3u * k + 0] = sources_coords[src].x;
                    leaf.leaf_coords[3u * k + 1] = sources_coords[src].y;
                    leaf.leaf_coords[3u * k + 2] = sources_coords[src].z;
                    leaf.leaf_values[3u * k + 0] = sources_values[src].x;
                    leaf.leaf_values[3u * k + 1] = sources_values[src].y;
                    leaf.leaf_values[3u * k + 2] = sources_values[src].z;
                }

                const bool mp_ok = multipole_create(
                    settings->order, (unsigned)leaf_buf_size, mp_slices[i], nodes[i].center, (unsigned)n_particles,
                    (const real3_t *)leaf.leaf_coords, (const real3_t *)leaf.leaf_values, leaf.leaf_cur, leaf.leaf_nxt,
                    &nodes[i].data.mp);
                /* Force gcc to keep nodes[i].kind live across the call —
                 * the strict-aliasing rules can otherwise let it spill a
                 * temporary into the kind slot via the inactive-union
                 * access patterns. */
                if (nodes[i].kind != BH_NODE_MULTIPOLE)
                    leaf_ok = false;

                if (!mp_ok)
                    leaf_ok = false;
            }
        }
        bh_free(allocator, mp_leaf_indices);
    }
    if (!leaf_ok)
    {
        return false;
    }

    /* Per-thread scratch stride for multipole_update. */
    const unsigned order = settings->order;
    const size_t leaf_stride =
        (3u * n_coeffs > multipole_scratch_size(order)) ? (3u * n_coeffs) : multipole_scratch_size(order);

    /* Build depth-range table for slice-based iteration. Tree depth is
     * stored in uint8_t, so 256 entries cover all possible depths. */
    unsigned depth_start[256], depth_end[256];
    compute_depth_ranges(n_topo_nodes, nodes, max_depth, depth_start, depth_end);

    /* Per-thread stride for shift_exp and pse (sized for n_threads in scratch). */
    const size_t shift_stride = 3u * (size_t)(work_order + 1) * (size_t)(work_order + 1);
    const size_t pse_stride = 2u * multipole_num_coeffs(work_order);

    /* --- Upward sweep: aggregate all children into internal nodes. --- */
    for (unsigned d = max_depth;; --d)
    {
        upward_sweep_level(depth_start[d], depth_end[d], nodes, order, n_coeffs, mp_slices, work_order,
                           scratch.shift_exp, scratch.pse, shift_stride, pse_stride, particle_order, sources_coords,
                           sources_values, &scratch, leaf_stride, n_threads);
        if (d == 0)
            break;
    }

    out->settings = *settings;
    out->root_center = nodes[0].center;
    out->root_half_size = nodes[0].half_size;
    out->n_sources = n_sources;
    out->n_nodes = n_topo_nodes;
    out->n_internal = n_internal;
    out->n_multipole_leaves = n_multipole;
    out->n_particle_leaves = n_particle;
    out->max_depth_reached = max_depth;
    out->buffer = (uint8_t *)buffer;
    out->buffer_size = buffer_size;
    out->nodes = nodes;
    out->particle_order = particle_order;
    out->multipole_coeffs = multipole_coeffs;
    out->mp_slices = work_buffers.mp_slices;

    return true;
}

/**
 * @brief Run the downward pass: materialise nodes, descend sources, count leaves.
 *
 * Converts the topo array into a bh_node_t array, descends every source
 * through the materialised tree to accumulate per-leaf counts, then
 * computes node metadata (particle ranges, multipole counts).
 *
 * @param n_sources       Number of source points.
 * @param sources_coords  Source coordinates.
 * @param settings        Build settings.
 * @param scratch         Scratch buffer views (topo array, source-leaf maps).
 * @param work_buffers    Work buffer views (nodes, particle_order, etc.).
 * @param count_res       Count-pass results.
 * @param n_threads       Number of OpenMP threads (>= 1).
 * @return Number of multipole leaves found.
 */
static size_t barnes_hut_downward_pass(const unsigned n_sources,
                                       const real3_t CVL_ARRAY_ARG(sources_coords, restrict n_sources),
                                       const barnes_hut_settings_t CVL_ARRAY_ARG(settings, restrict),
                                       barnes_hut_scratch_t scratch, barnes_hut_work_t work_buffers,
                                       barnes_hut_count_res_t count_res, unsigned n_threads)
{
    /* --- Materialize the bh_node_t tree from the topo array. --- */
    const size_t n_topo_nodes = count_res.n_internal + count_res.n_multipole_leaves + count_res.n_particle_leaves;
    materialize_tree(scratch.topo, n_topo_nodes, settings, work_buffers.topo_to_real, work_buffers.nodes,
                     work_buffers.multipole_coeffs, work_buffers.mp_slices);

    /* --- Descend each source through the bh_node_t tree to count per-leaf. --- */
    descend_for_each_source(n_sources, sources_coords, n_topo_nodes, work_buffers.nodes, scratch.source_leaf_real,
                            n_threads);

    /* --- Assign particle ranges, reset counts, and count multipole leaves
     * in one combined pass. --- */
    return compute_node_metadata((uint32_t)n_topo_nodes, work_buffers.nodes, count_res.max_depth, NULL, NULL);
}

bool barnes_hut_tree_build(unsigned n_sources, unsigned n_threads,
                           const real3_t CVL_ARRAY_ARG(sources_coords, restrict n_sources),
                           const real3_t CVL_ARRAY_ARG(sources_values, restrict n_sources),
                           const barnes_hut_settings_t CVL_ARRAY_ARG(settings, restrict), const allocator_t *allocator,
                           barnes_hut_tree_t *out)
{
    void *scratch_buffer = NULL, *work_buffer = NULL;
    uint32_t *multipole_leaf_indices = NULL;
    // Return value for later cleanup
    bool ret = false;

    // Validation
    if (!out || n_sources == 0 || !settings_valid(settings) || sources_coords == NULL || sources_values == NULL)
        return false;

    // Size the scratch buffer and partition it
    const barnes_hut_scratch_sizes_t scratch_sizes = barnes_hut_size_scratch(n_sources, settings);
    const size_t needed_scratch = barnes_hut_total_scratch_size(scratch_sizes, n_threads);
    if (needed_scratch == 0)
        return false;

    // Allocation 1
    scratch_buffer = bh_alloc(allocator, needed_scratch);
    if (!scratch_buffer)
        return false;

    const barnes_hut_scratch_t scratch = barnes_hut_scratch_partition(n_threads, scratch_buffer, scratch_sizes);

    /* Zero the topo scratch — count_pass reads `is_internal` and `children[]`
     * before writing them, so leaving garbage would corrupt the descent. */
    const size_t topo_bytes = (8u * (size_t)n_sources + 1u) * sizeof(topo_node_t);
    memset(scratch.topo, 0, topo_bytes);

    /* --- Run the count pass to learn the topology. --- */
    const barnes_hut_count_res_t count_res =
        barnes_hut_count_pass(n_sources, sources_coords, settings, scratch.topo, scratch.source_leaf_topo);

    // Size the work buffer and allocate it

    /* --- Compute partition layout and validate buffer_size. --- */
    const barnes_hut_work_sizes_t work_sizes = barnes_hut_size_work_buffer(n_sources, settings, count_res);
    const size_t total_work_size = barnes_hut_total_work_size(work_sizes);

    // Allocation 2
    work_buffer = bh_alloc(allocator, total_work_size);
    if (!work_buffer)
        goto cleanup;

    /* --- Set up views into the caller buffer. --- */
    const barnes_hut_work_t work_buffers = partition_work_buffer(work_sizes, work_buffer);

    /* --- Perform the downward pass and count multipole leaves. --- */
    const size_t n_multipole_leaves =
        barnes_hut_downward_pass(n_sources, sources_coords, settings, scratch, work_buffers, count_res, n_threads);

    /* --- Fill particle_order[]. --- */
    // NOTE: Can parallelise this using atomic capture. Schedule is static, with big chunks, because typically the
    // leaves close in index are close in space, so by giving them large chunks, we have less of a chance of triggering
    // atomics.
#pragma omp parallel for default(none) shared(n_sources, scratch, work_buffers) schedule(static) num_threads(n_threads)
    for (unsigned i = 0; i < n_sources; ++i)
    {
        const unsigned leaf_idx = scratch.source_leaf_real[i];
        bh_node_t *leaf = work_buffers.nodes + leaf_idx;
        unsigned cnt;
#pragma omp atomic capture
        {
            cnt = leaf->particle_count;
            leaf->particle_count += 1;
        }
        const unsigned slot = leaf->particle_begin + cnt;
        work_buffers.particle_order[slot] = i;
    }

    /* Compute |-weighted centroids for leaves before building
     * multipoles — the multipole coefficients must use the weighted center
     * as expansion origin.  Internal-node centroids are computed during
     * the upward sweep. */
    {
        const size_t n_nodes_total =
            (size_t)count_res.n_internal + (size_t)count_res.n_multipole_leaves + (size_t)count_res.n_particle_leaves;
        barnes_hut_compute_leaf_centers((unsigned)n_nodes_total, work_buffers.nodes, work_buffers.particle_order,
                                        sources_coords, sources_values, n_threads);
    }

    if (n_multipole_leaves > 0)
    {
        // We have to construct the multipoles
        multipole_leaf_indices = (uint32_t *)bh_alloc(allocator, (size_t)n_multipole_leaves * sizeof(uint32_t));
        if (!multipole_leaf_indices)
            goto cleanup;

        // Build the multipoles
        const size_t n_coeffs_mp = multipole_num_coeffs(settings->order);
        const size_t mp_scratch = multipole_scratch_size(settings->order);
        const size_t leaf_buf_size = (3u * n_coeffs_mp > mp_scratch ? 3u * n_coeffs_mp : mp_scratch);
        const bool leaf_ok = barnes_hut_tree_build_multipoles(
            (unsigned)count_res.n_internal + count_res.n_multipole_leaves + count_res.n_particle_leaves,
            work_buffers.nodes, work_buffers.particle_order, sources_coords, sources_values, settings, scratch,
            leaf_buf_size, n_sources, n_threads, work_buffers.mp_slices, (unsigned)n_multipole_leaves,
            multipole_leaf_indices);

        // We can cleanup early!
        bh_free(allocator, multipole_leaf_indices);

        if (!leaf_ok)
            goto cleanup;
    }

    *out = barnes_hut_tree_complete(count_res, n_sources, settings, resolve_work_order(settings), sources_coords,
                                    sources_values, work_buffers.nodes, work_buffers.particle_order,
                                    work_buffers.multipole_coeffs, work_buffers.mp_slices, &scratch);
    out->buffer = (uint8_t *)work_buffer;
    out->buffer_size = total_work_size;
    work_buffer = NULL; /* ownership transferred to out */

    ret = true;
    // End of the function/cleanup
cleanup:
    bh_free(allocator, work_buffer);
    bh_free(allocator, scratch_buffer);
    return ret;
}

unsigned barnes_hut_tree_n_nodes(const barnes_hut_tree_t *tree)
{
    return tree ? tree->n_nodes : 0;
}

void barnes_hut_tree_depth_stats(const barnes_hut_tree_t *tree, unsigned *min_depth, unsigned *max_depth)
{
    if (tree == NULL || tree->n_nodes == 0)
    {
        if (min_depth)
            *min_depth = 0;
        if (max_depth)
            *max_depth = 0;
        return;
    }

    /* Walk all materialized nodes and accumulate the depth range.
     * The root sits at depth 0, internal nodes may sit at any depth in
     * (1, max_depth_reached], and leaves share depths with internal nodes. */
    unsigned lo = tree->nodes[0].depth;
    unsigned hi = tree->nodes[0].depth;
    for (unsigned i = 1; i < tree->n_nodes; ++i)
    {
        const unsigned d = tree->nodes[i].depth;
        if (d < lo)
            lo = d;
        if (d > hi)
            hi = d;
    }

    if (min_depth)
        *min_depth = lo;
    if (max_depth)
        *max_depth = hi;
}

size_t barnes_hut_tree_memory_bytes(const barnes_hut_tree_t *tree)
{
    return tree ? tree->buffer_size : 0;
}

/* ------------------------------------------------------------------ */
/* Evaluation                                                         */
/* ------------------------------------------------------------------ */

/**
 * @brief Particle-leaf direct-evaluation kernel: @f$ \Gamma / |r|^2 @f$.
 *
 * Matches the asymptotic scaling of `multipole_eval` (no cross product, no
 * @f$ 1/4\pi @f$ factor). Returns zero for singularities (r < 1e-30).
 */
static inline real3_t particle_kernel(real3_t gamma, real3_t r_vec)
{
    const real_t r2 = real3_dot(r_vec, r_vec);
    if (r2 < 1e-30)
        return (real3_t){.x = 0, .y = 0, .z = 0};
    return real3_mul1(gamma, 1.0 / r2);
}

/**
 * @brief Multipole Acceptance Criterion (MAC).
 *
 * When `theta <= 0` (default): neighbour criterion — accept if the target
 * point is outside the cell's @f$ 3 \times 3 \times 3 @f$ neighbourhood
 * (@f$ |\Delta x| > 2 h @f$ or @f$ |\Delta y| > 2 h @f$ or
 * @f$ |\Delta z| > 2 h @f$).
 *
 * When `theta > 0`: opening-angle criterion — accept if
 * @f$ h / |r| < \theta @f$.
 */
static inline bool mac_accept(const bh_node_t *node, real3_t point, double theta)
{
    const real3_t diff = real3_sub(point, node->center);
    if (theta <= 0.0)
    {
        /* Neighbour criterion: outside 3x3x3 cell neighbourhood. */
        return fabs(diff.x) > 2.0 * node->half_size || fabs(diff.y) > 2.0 * node->half_size ||
               fabs(diff.z) > 2.0 * node->half_size;
    }
    /* Opening-angle criterion. */
    const real_t dist = real3_mag(diff);
    if (dist < 1e-30)
        return false;
    return node->half_size / dist < theta;
}

real3_t barnes_hut_tree_eval(const barnes_hut_tree_t *tree, const real3_t CVL_ARRAY_ARG(sources_coords, restrict),
                             const real3_t CVL_ARRAY_ARG(sources_values, restrict), real3_t point,
                             barnes_hut_eval_settings_t eval_settings)
{
    if (tree == NULL || tree->nodes == NULL || tree->n_nodes == 0)
        return (real3_t){.x = 0, .y = 0, .z = 0};

    real3_t result = {.x = 0, .y = 0, .z = 0};

    enum
    {
        EVAL_STACK_MAX = 1024
    };
    uint32_t stack[EVAL_STACK_MAX];
    int sp = 0;
    stack[sp++] = 0;

    const double theta = eval_settings.theta;
    const unsigned order = tree->settings.order;
    const size_t n_coeffs = multipole_num_coeffs(order);

    while (sp > 0)
    {
        sp -= 1;
        const uint32_t idx = stack[sp];
        const bh_node_t *node = &tree->nodes[idx];

        /* Hot path: BH_NODE_INTERNAL is most common, especially near root. */
        if (CVL_EXPECT_CONDITION(node->kind == BH_NODE_INTERNAL))
        {
            if (mac_accept(node, point, theta))
            {
                /* Evaluate the internal node's aggregated multipole.
                 * Prefetch coefficient arrays before constructing the
                 * multipole_t to hide memory latency. */
                real_t *slice = tree->mp_slices[idx];
                if (CVL_EXPECT_CONDITION(slice != NULL))
                {
                    CVL_PREFETCH(slice, 0, 3);
                    CVL_PREFETCH(slice + n_coeffs, 0, 3);
                    CVL_PREFETCH(slice + 2u * n_coeffs, 0, 3);
                    const multipole_t mp = {.order = order,
                                            .center = node->center,
                                            .coeffs_x = slice,
                                            .coeffs_y = slice + n_coeffs,
                                            .coeffs_z = slice + 2u * n_coeffs};
                    result = real3_add(result, multipole_eval(&mp, point));
                }
            }
            else
            {
                /* Descend into children. */
                for (int k = 0; k < 8; ++k)
                {
                    const bh_node_t *child = node->data.internal.children[k];
                    if (CVL_EXPECT_CONDITION(child == NULL))
                        continue;
                    if ((size_t)sp + 1 > EVAL_STACK_MAX)
                        break;
                    stack[sp++] = (uint32_t)(child - tree->nodes);
                }
            }
        }
        else
        {
            /* BH_NODE_MULTIPOLE or BH_NODE_PARTICLE — if the leaf has a
             * far-field multipole and the MAC accepts it, use the multipole.
             * Otherwise fall back to direct particle sum for accuracy. */
            real_t *slice = tree->mp_slices[idx];
            if (slice != NULL && mac_accept(node, point, theta))
            {
                CVL_PREFETCH(slice, 0, 3);
                CVL_PREFETCH(slice + n_coeffs, 0, 3);
                CVL_PREFETCH(slice + 2u * n_coeffs, 0, 3);
                const multipole_t mp = {.order = order,
                                        .center = node->data.mp.center,
                                        .coeffs_x = slice,
                                        .coeffs_y = slice + n_coeffs,
                                        .coeffs_z = slice + 2u * n_coeffs};
                result = real3_add(result, multipole_eval(&mp, point));
            }
            else
            {
                const unsigned begin = node->particle_begin;
                const unsigned end = begin + node->particle_count;
                for (unsigned k = begin; k < end; ++k)
                {
                    const unsigned src = tree->particle_order[k];
                    const real3_t dr = real3_sub(point, sources_coords[src]);
                    result = real3_add(result, particle_kernel(sources_values[src], dr));
                }
            }
        }
    }

    return result;
}

void barnes_hut_tree_eval_all(const barnes_hut_tree_t *tree, const real3_t CVL_ARRAY_ARG(sources_coords, restrict),
                              const real3_t CVL_ARRAY_ARG(sources_values, restrict), unsigned n_targets,
                              const real3_t CVL_ARRAY_ARG(targets, restrict n_targets),
                              real3_t CVL_ARRAY_ARG(results, restrict n_targets),
                              barnes_hut_eval_settings_t eval_settings, unsigned n_threads)
{
#pragma omp parallel for default(none) shared(tree, sources_coords, sources_values, n_targets, targets, results,       \
                                                  eval_settings) num_threads(n_threads) schedule(static)
    for (unsigned i = 0; i < n_targets; ++i)
    {
        results[i] = barnes_hut_tree_eval(tree, sources_coords, sources_values, targets[i], eval_settings);
    }
}
