#include "barnes_hut_tree.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
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
    /* Cell geometry, replicated from the parent's split so we can descend. */
    real3_t center;
    real_t half_size;
} topo_node_t;

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
 * The thresholds encode the cost model: approximating n > (P+1)^3 particles
 * with one multipole of order P is cheaper than n direct 1/r² evaluations,
 * and subdividing is worth it only when the multipole would represent a
 * large cluster (n >> (P+1)^3).
 *
 * @f$ \mathrm{mp\_threshold} = \max(1, \mathrm{critical\_particle\_count}) @f$
 * @f$ \mathrm{subdivide\_threshold} = \mathrm{critical\_particle\_count} \cdot (P + 1)^3 @f$
 */
static unsigned multipole_threshold(const barnes_hut_settings_t *settings)
{
    return settings->critical_particle_count < 1u ? 1u : settings->critical_particle_count;
}

static unsigned subdivide_threshold(const barnes_hut_settings_t *settings)
{
    const unsigned p1 = settings->order + 1;
    /* (P+1)^3 fits in 64-bit for any realistic P (P+1 <= 1024 fits easily). */
    const uint64_t num = (uint64_t)settings->critical_particle_count * (uint64_t)p1 * (uint64_t)p1 * (uint64_t)p1;
    const unsigned threshold = (unsigned)num;
    return threshold < 1u ? 1u : threshold;
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
    if (settings->n_threads < 1)
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
static void count_pass(unsigned n_sources, const real3_t CVL_ARRAY_ARG(sources_coords, restrict n_sources),
                       const barnes_hut_settings_t *settings, topo_node_t *topo, uint32_t *source_leaf,
                       unsigned *out_n_internal, unsigned *out_n_multipole_leaves, unsigned *out_n_particle_leaves,
                       unsigned *out_max_depth)
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
                    return;
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

        /* Subdivide if threshold exceeded. */
        if (should_subdivide(topo[node_idx].particle_count, depth, settings))
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
                        return;
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

    *out_n_internal = n_internal;
    *out_n_multipole_leaves = n_multipole;
    *out_n_particle_leaves = n_particle;
    *out_max_depth = max_depth;
}

/* ------------------------------------------------------------------ */
/* Public API — buffer sizing                                         */
/* ------------------------------------------------------------------ */

size_t barnes_hut_buffer_size(unsigned n_sources, const barnes_hut_settings_t *settings)
{
    if (n_sources == 0 || !settings_valid(settings))
        return 0;

    const unsigned work_order = resolve_work_order(settings);

    /* Pessimistic upper bound on node count: n_sources leaves + ~ n_sources/7
     * internal nodes. We add a small constant for alignment slack. */
    const size_t max_nodes = (size_t)n_sources + (size_t)((n_sources + 6u) / 7u) + 16u;
    const size_t nodes_bytes = max_nodes * sizeof(bh_node_t);
    const size_t particle_order_bytes = (size_t)n_sources * sizeof(unsigned);
    /* Worst case: every node carries a multipole. */
    const size_t multipole_coeffs_bytes = max_nodes * 3u * multipole_num_coeffs(settings->order) * sizeof(real_t);
    const size_t scratch_bytes = multipole_scratch_size(work_order) * sizeof(real_t);
    const size_t shift_exp_bytes = 3u * (size_t)(work_order + 1) * (size_t)(work_order + 1) * sizeof(real_t);
    const size_t pse_bytes = 2u * multipole_num_coeffs(work_order) * sizeof(real_t);

    return nodes_bytes + particle_order_bytes + multipole_coeffs_bytes + scratch_bytes + shift_exp_bytes + pse_bytes;
}

/* ------------------------------------------------------------------ */
/* Public API — scratch sizing                                        */
/* ------------------------------------------------------------------ */

/**
 * @brief Sum of the per-region sizes of the scratch layout returned by
 *        `barnes_hut_scratch_size`. Kept as a static helper so the layout
 *        computation is shared between the public sizer and the internal
 *        partition routine.
 *
 * `n_thread_partitions` is the number of identical per-thread multipole
 * scratch regions to allocate. Callers must set `settings.n_threads >= 1`
 * (validated by `settings_valid`); the scratch is sized for exactly that
 * many thread partitions, and the OpenMP parallel region is pinned to the
 * same thread count via `num_threads(n_threads)`.
 */
static size_t barnes_hut_scratch_layout(unsigned n_sources, const barnes_hut_settings_t *settings,
                                        unsigned n_thread_partitions, size_t *out_topo, size_t *out_source_leaf_topo,
                                        size_t *out_source_leaf_real, size_t *out_leaf_buf_total,
                                        size_t *out_leaf_coords_total, size_t *out_leaf_values_total)
{
    if (n_sources == 0 || !settings_valid(settings))
        return 0;

    /* Layout (one contiguous byte stream, all offsets are byte offsets):
     *   1. topo                — (8 * n_sources + 1) * sizeof(topo_node_t)
     *   2. source_leaf_topo    — n_sources * sizeof(uint32_t)
     *   3. source_leaf_real    — n_sources * sizeof(unsigned)
     *   4. leaf_cur, leaf_nxt  — 2 * leaf_buf_size * sizeof(real_t) per thread
     *   5. leaf_coords         — n_sources * 3 * sizeof(real_t) per thread
     *   6. leaf_values         — n_sources * 3 * sizeof(real_t) per thread
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

    const size_t leaf_buf_total = leaf_buf_per_thread * n_thread_partitions;
    const size_t leaf_coords_total = leaf_coords_per_thread * n_thread_partitions;
    const size_t leaf_values_total = leaf_values_per_thread * n_thread_partitions;

    if (out_topo)
        *out_topo = topo_bytes;
    if (out_source_leaf_topo)
        *out_source_leaf_topo = source_leaf_topo_bytes;
    if (out_source_leaf_real)
        *out_source_leaf_real = source_leaf_real_bytes;
    if (out_leaf_buf_total)
        *out_leaf_buf_total = leaf_buf_total;
    if (out_leaf_coords_total)
        *out_leaf_coords_total = leaf_coords_total;
    if (out_leaf_values_total)
        *out_leaf_values_total = leaf_values_total;

    return topo_bytes + source_leaf_topo_bytes + source_leaf_real_bytes + leaf_buf_total + leaf_coords_total +
           leaf_values_total;
}

size_t barnes_hut_scratch_size(unsigned n_sources, const barnes_hut_settings_t *settings)
{
    /* `settings.n_threads` is guaranteed >= 1 by `settings_valid`. */
    return barnes_hut_scratch_layout(n_sources, settings, settings->n_threads, NULL, NULL, NULL, NULL, NULL, NULL);
}

/**
 * @brief Internal scratch view. Each pointer aliases a partition of the
 *        caller-provided `scratch_buffer`. The buffer is single-use; callers
 *        are not required to zero or otherwise initialise it.
 */
typedef struct
{
    topo_node_t *topo;
    uint32_t *source_leaf_topo;
    unsigned *source_leaf_real;
    /* Per-thread multipole scratch. The arrays are contiguous blocks of
     * `n_thread_partitions` slices; thread `t` uses offset `t * per_thread`. */
    unsigned n_thread_partitions;
    real_t *leaf_cur;    /* [n_thread_partitions * leaf_buf_size] */
    real_t *leaf_nxt;    /* [n_thread_partitions * leaf_buf_size] */
    real_t *leaf_coords; /* [n_thread_partitions * n_sources * 3] */
    real_t *leaf_values; /* [n_thread_partitions * n_sources * 3] */
} barnes_hut_scratch_t;

/**
 * @brief Partition @p scratch_buffer into the regions described by
 *        `barnes_hut_scratch_layout`. Returns `false` (and leaves `out`
 *        untouched) if @p scratch_size is smaller than required.
 */
static bool barnes_hut_scratch_partition(unsigned n_sources, const barnes_hut_settings_t *settings,
                                         unsigned n_thread_partitions, void *scratch_buffer, size_t scratch_size,
                                         barnes_hut_scratch_t *out)
{
    const size_t n_coeffs = multipole_num_coeffs(settings->order);
    const size_t multipole_scratch = multipole_scratch_size(settings->order);
    const size_t leaf_buf_size = (3u * n_coeffs > multipole_scratch ? 3u * n_coeffs : multipole_scratch);

    size_t s_topo = 0, s_leaf_topo = 0, s_leaf_real = 0, s_cur = 0, s_nxt = 0, s_coords = 0, s_values = 0;
    const size_t total = barnes_hut_scratch_layout(n_sources, settings, n_thread_partitions, &s_topo, &s_leaf_topo,
                                                   &s_leaf_real, NULL, &s_coords, &s_values);
    if (scratch_size < total)
        return false;

    /* Recover the per-region sizes the layout intentionally collapsed
     * into a single `s_buf`. The layout was originally written assuming
     * `leaf_cur + leaf_nxt` were one combined region, but the per-thread
     * accessor in `barnes_hut_leaf_scratch_for` treats them as two
     * independent per-thread arrays. We split the s_buf bucket here so
     * both leaf_cur and leaf_nxt each get their own `n_thread_partitions *
     * leaf_buf_size * sizeof(real_t)` slab. */
    s_cur = (size_t)n_thread_partitions * leaf_buf_size * sizeof(real_t);
    s_nxt = s_cur;

    uint8_t *bp = (uint8_t *)scratch_buffer;
    out->topo = (topo_node_t *)bp;
    bp += s_topo;
    out->source_leaf_topo = (uint32_t *)bp;
    bp += s_leaf_topo;
    out->source_leaf_real = (unsigned *)bp;
    bp += s_leaf_real;
    out->leaf_cur = (real_t *)bp;
    bp += s_cur;
    out->leaf_nxt = (real_t *)bp;
    bp += s_nxt;
    out->leaf_coords = (real_t *)bp;
    bp += s_coords;
    out->leaf_values = (real_t *)bp;
    bp += s_values;
    out->n_thread_partitions = n_thread_partitions;
    return true;
}

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
/* Public API — count only                                            */
/* ------------------------------------------------------------------ */

bool barnes_hut_tree_count(unsigned n_sources, const real3_t CVL_ARRAY_ARG(sources_coords, restrict n_sources),
                           const barnes_hut_settings_t CVL_ARRAY_ARG(settings, restrict), void *scratch_buffer,
                           size_t scratch_size, const allocator_t *allocator, size_t *required_buffer_size)
{
    (void)allocator; /* count pass needs no residual allocations: the scratch buffer carries everything. */
    if (!required_buffer_size)
        return false;
    *required_buffer_size = 0;
    if (n_sources == 0 || !settings_valid(settings) || sources_coords == NULL)
        return false;
    if (scratch_buffer == NULL || scratch_size == 0)
        return false;

    barnes_hut_scratch_t scratch;
    if (!barnes_hut_scratch_partition(n_sources, settings, settings->n_threads, scratch_buffer, scratch_size, &scratch))
        return false;

    /* Zero the topo scratch — count_pass reads `is_internal` and `children[]`
     * before writing them, so leaving garbage would corrupt the descent. */
    const size_t topo_bytes = (8u * (size_t)n_sources + 1u) * sizeof(topo_node_t);
    memset(scratch.topo, 0, topo_bytes);

    unsigned n_internal = 0, n_multipole = 0, n_particle = 0, max_depth = 0;
    count_pass(n_sources, sources_coords, settings, scratch.topo, scratch.source_leaf_topo, &n_internal, &n_multipole,
               &n_particle, &max_depth);

    const unsigned n_total = n_internal + n_multipole + n_particle;
    (void)max_depth;

    const unsigned work_order = resolve_work_order(settings);
    const size_t nodes_bytes = (size_t)n_total * sizeof(bh_node_t);
    const size_t particle_order_bytes = (size_t)n_sources * sizeof(unsigned);
    const size_t multipole_coeffs_bytes =
        (size_t)(n_internal + n_multipole) * 3u * multipole_num_coeffs(settings->order) * sizeof(real_t);
    const size_t scratch_bytes = multipole_scratch_size(work_order) * sizeof(real_t);
    const size_t shift_exp_bytes = 3u * (size_t)(work_order + 1) * (size_t)(work_order + 1) * sizeof(real_t);
    const size_t pse_bytes = 2u * multipole_num_coeffs(work_order) * sizeof(real_t);

    *required_buffer_size =
        nodes_bytes + particle_order_bytes + multipole_coeffs_bytes + scratch_bytes + shift_exp_bytes + pse_bytes;

    return true;
}

/* ------------------------------------------------------------------ */
/* Public API stubs for now — fleshed out in step 5                   */
/* ------------------------------------------------------------------ */

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
 * Fills `nodes[]` for the indices it touches. Returns the total number of
 * real-tree nodes written (== `n_total`).
 */
static uint32_t materialize_tree(const topo_node_t CVL_ARRAY_ARG(topo, restrict), uint32_t n_topo_nodes,
                                 const barnes_hut_settings_t *settings,
                                 uint32_t CVL_ARRAY_ARG(topo_to_real, restrict n_topo_nodes), unsigned work_order,
                                 bh_node_t CVL_ARRAY_ARG(nodes, restrict n_topo_nodes), real_t *multipole_coeffs,
                                 real_t *CVL_ARRAY_ARG(mp_slices_out, restrict n_topo_nodes),
                                 unsigned *out_n_multipole_slices)
{
    real_t *cursor = multipole_coeffs;
    unsigned n_multipole_slices = 0;
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
                        return next_real;
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
            fprintf(stderr, "PROBE-MATMP ti=%u real=%u slice=%p cursor_off=%ld\n", (unsigned)ti,
                    (unsigned)(next_real - 1), (void *)node->data.mp.coeffs_x,
                    (long)((char *)node->data.mp.coeffs_x - (char *)multipole_coeffs));
            fflush(stderr);
            (void)work_order;
            n_multipole_slices += 1;
        }
        else
        {
            node->kind = BH_NODE_PARTICLE;
        }
    }

    /* Second pass: resolve child pointers that hold topo-index sentinels. */
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

    if (out_n_multipole_slices != NULL)
        *out_n_multipole_slices = n_multipole_slices;
    return next_real;
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
            bh_node_t *child = nodes[idx].data.internal.children[oct];
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
 * @brief Compute per-leaf `particle_begin` as a prefix sum across leaves.
 *        Internal nodes are skipped — they don't own particle slots.
 */
static void compute_particle_begins(uint32_t n_real_nodes, bh_node_t CVL_ARRAY_ARG(nodes, restrict n_real_nodes))
{
    unsigned cursor = 0;
    for (uint32_t i = 0; i < n_real_nodes; ++i)
    {
        if (nodes[i].kind != BH_NODE_INTERNAL)
        {
            nodes[i].particle_begin = cursor;
            cursor += nodes[i].particle_count;
            /* Reset to 0 — we'll use it as a write cursor below. */
            nodes[i].particle_count = 0;
        }
    }
}

bool barnes_hut_tree_insert(unsigned n_sources, const real3_t CVL_ARRAY_ARG(sources_coords, restrict n_sources),
                            const real3_t CVL_ARRAY_ARG(sources_values, restrict n_sources),
                            const barnes_hut_settings_t CVL_ARRAY_ARG(settings, restrict), void *scratch_buffer,
                            size_t scratch_size, const allocator_t *allocator, void *buffer, size_t buffer_size,
                            barnes_hut_tree_t *out)
{
    if (!buffer || !out)
        return false;
    if (!scratch_buffer || scratch_size == 0)
        return false;
    if (n_sources == 0 || !settings_valid(settings))
        return false;
    if (sources_coords == NULL || sources_values == NULL)
        return false;

    const unsigned work_order = resolve_work_order(settings);
    const unsigned n_threads = settings->n_threads;

    /* --- Partition the scratch buffer (topo + source-leaf maps + per-thread
     * leaf multipole scratch). All of these are input-sized, so the scratch
     * is reusable across calls and across trees. --- */
    barnes_hut_scratch_t scratch;
    if (!barnes_hut_scratch_partition(n_sources, settings, n_threads, scratch_buffer, scratch_size, &scratch))
        return false;

    /* Zero the topo scratch — count_pass reads `is_internal` and `children[]`
     * before writing them, so leaving garbage would corrupt the descent. */
    const size_t topo_bytes = (8u * (size_t)n_sources + 1u) * sizeof(topo_node_t);
    memset(scratch.topo, 0, topo_bytes);

    /* --- Run the count pass to learn the topology. --- */
    unsigned n_internal = 0, n_multipole = 0, n_particle = 0, max_depth = 0;
    count_pass(n_sources, sources_coords, settings, scratch.topo, scratch.source_leaf_topo, &n_internal, &n_multipole,
               &n_particle, &max_depth);

    const uint32_t n_topo_nodes = (uint32_t)(n_internal + n_multipole + n_particle);
    if (n_topo_nodes == 0)
        return false;

    /* --- Allocate the count-pass-output-sized residual scratch through the
     * allocator. `topo_to_real` maps each topo node to its bh_node_t index;
     * `mp_slices` carries each multipole-bearing node's coefficient slice
     * pointer. Both are sized from `n_topo_nodes` which is only known after
     * the count pass, hence the fallback to the allocator. --- */
    uint32_t *topo_to_real = (uint32_t *)bh_alloc(allocator, (size_t)n_topo_nodes * sizeof(*topo_to_real));
    real_t **mp_slices = (real_t **)bh_alloc(allocator, (size_t)n_topo_nodes * sizeof(*mp_slices));
    if (!topo_to_real || !mp_slices)
    {
        bh_free(allocator, topo_to_real);
        bh_free(allocator, mp_slices);
        return false;
    }

    /* --- Compute partition layout and validate buffer_size. --- */
    const size_t nodes_bytes = (size_t)n_topo_nodes * sizeof(bh_node_t);
    const size_t particle_order_bytes = (size_t)n_sources * sizeof(unsigned);
    const size_t multipole_coeffs_bytes =
        (size_t)(n_internal + n_multipole) * 3u * multipole_num_coeffs(settings->order) * sizeof(real_t);
    const size_t scratch_bytes = multipole_scratch_size(work_order) * sizeof(real_t);
    const size_t shift_exp_bytes = 3u * (size_t)(work_order + 1) * (size_t)(work_order + 1) * sizeof(real_t);
    const size_t pse_bytes = 2u * multipole_num_coeffs(work_order) * sizeof(real_t);
    const size_t total_bytes =
        nodes_bytes + particle_order_bytes + multipole_coeffs_bytes + scratch_bytes + shift_exp_bytes + pse_bytes;
    if (buffer_size < total_bytes)
    {
        bh_free(allocator, topo_to_real);
        bh_free(allocator, mp_slices);
        return false;
    }

    /* --- Set up views into the caller buffer. --- */
    uint8_t *bp = (uint8_t *)buffer;
    bh_node_t *nodes = (bh_node_t *)bp;
    bp += nodes_bytes;
    unsigned *particle_order = (unsigned *)bp;
    bp += particle_order_bytes;
    real_t *multipole_coeffs = (real_t *)bp;
    bp += multipole_coeffs_bytes;
    real_t *buf_scratch = (real_t *)bp;
    bp += scratch_bytes;
    real_t *shift_exp = (real_t *)bp;
    bp += shift_exp_bytes;
    real_t *pse = (real_t *)bp;

    /* --- Materialize the bh_node_t tree from the topo array. --- */
    const uint32_t n_real = materialize_tree(scratch.topo, n_topo_nodes, settings, topo_to_real, work_order, nodes,
                                             multipole_coeffs, mp_slices, NULL);
    (void)n_real;

    /* --- Descend each source through the bh_node_t tree to count per-leaf. --- */
    descend_for_each_source(n_sources, sources_coords, n_topo_nodes, nodes, scratch.source_leaf_real, n_threads);

    /* --- Assign particle ranges per leaf. --- */
    compute_particle_begins(n_topo_nodes, nodes);

    /* --- Fill particle_order[]. --- */
    for (unsigned i = 0; i < n_sources; ++i)
    {
        const unsigned leaf_idx = scratch.source_leaf_real[i];
        bh_node_t *leaf = &nodes[leaf_idx];
        const unsigned slot = leaf->particle_begin + leaf->particle_count;
        particle_order[slot] = i;
        leaf->particle_count += 1;
    }

    /* --- Build multipoles for each multipole-bearing leaf. Parallelised:
     * the scratch is pre-partitioned into n_threads independent leaf
     * scratch slices; thread `t` writes only into slice `t`. --- */
    const size_t n_coeffs = multipole_num_coeffs(settings->order);
    const size_t multipole_scratch = multipole_scratch_size(settings->order);
    const size_t leaf_buf_size = (3u * n_coeffs > multipole_scratch ? 3u * n_coeffs : multipole_scratch);

    /* Count multipole leaves up front so each thread gets a contiguous
     * subrange of indices in a packed array (omp doesn't iterate sparse
     * indices cleanly). */
    unsigned n_mp_leaves = 0;
    for (uint32_t i = 0; i < n_topo_nodes; ++i)
        if (nodes[i].kind == BH_NODE_MULTIPOLE)
            n_mp_leaves += 1;

    bool leaf_ok = true;
    if (n_mp_leaves > 0)
    {
        uint32_t *mp_leaf_indices = (uint32_t *)bh_alloc(allocator, (size_t)n_mp_leaves * sizeof(uint32_t));
        if (mp_leaf_indices == NULL)
        {
            bh_free(allocator, topo_to_real);
            bh_free(allocator, mp_slices);
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
        fprintf(stderr, "DEBUG-PRE-OMP n=%u counts:", (unsigned)n_topo_nodes);
        for (uint32_t i = 0; i < n_topo_nodes; ++i)
            fprintf(stderr, " %u:%u", (unsigned)i, (unsigned)nodes[i].particle_count);
        fprintf(stderr, "\n");
        fflush(stderr);
        /* Print which node kinds are right before OMP, in serial. */
        fprintf(stderr, "DEBUG-KIND-PRE2:");
        for (uint32_t i = 0; i < n_topo_nodes; ++i)
            fprintf(stderr, " %u:%d", (unsigned)i, (int)nodes[i].kind);
        fprintf(stderr, "\n");
        fflush(stderr);
        fprintf(stderr, "DEBUG-NODES-PTR nodes=%p allocator=%p\n", (void *)nodes, (void *)(allocator));
        fflush(stderr);

        /* OMP disabled for serial debug. */
        {
            /* DEBUG */
            fprintf(stderr, "DEBUG-INNER-KINDS:");
            for (uint32_t i = 0; i < n_topo_nodes; ++i)
                fprintf(stderr, " %u:%d", (unsigned)i, (int)nodes[i].kind);
            fprintf(stderr, "\n");
            fflush(stderr);
            /* SERIAL: skip OMP and run inline. */
            const int tid = 0;
            const barnes_hut_leaf_scratch_t leaf =
                barnes_hut_leaf_scratch_for(&scratch, (unsigned)tid, n_sources, settings->order);
            for (unsigned ml = 0; ml < n_mp_leaves; ++ml)
            {
                const uint32_t i = mp_leaf_indices[ml];
                fprintf(stderr, "DEBUG-MLB ml=%u i=%u kind=%d count=%u\n", ml, (unsigned)i, (int)nodes[i].kind,
                        (unsigned)nodes[i].particle_count);
                fflush(stderr);
                const size_t n_particles = nodes[i].particle_count;

                /* Compute |Γ|-weighted centroid. */
                real3_t center = {.x = 0, .y = 0, .z = 0};
                real_t total_weight = 0.0;
                for (unsigned k = 0; k < n_particles; ++k)
                {
                    const unsigned src = particle_order[nodes[i].particle_begin + k];
                    const real_t w = real3_mag(sources_values[src]);
                    center.x += sources_coords[src].x * w;
                    center.y += sources_coords[src].y * w;
                    center.z += sources_coords[src].z * w;
                    total_weight += w;
                }
                if (total_weight > 0.0)
                {
                    center.x /= total_weight;
                    center.y /= total_weight;
                    center.z /= total_weight;
                }
                else
                {
                    center = nodes[i].center;
                }
                fprintf(stderr, "PROBE-BEFORE-MPC i=%u kind=%d depth=%u px=%u\n", (unsigned)i, (int)nodes[i].kind,
                        (unsigned)nodes[i].depth, 0xabcdef);
                fflush(stderr);
                nodes[i].data.mp.center = center;
                fprintf(stderr, "PROBE-AFTER-MPC= i=%u kind=%d depth=%u\n", (unsigned)i, (int)nodes[i].kind,
                        (unsigned)nodes[i].depth);
                fflush(stderr);

                fprintf(stderr,
                        "PROBE-PTRS i=%u nodes=%p leaf.coords=%p leaf.values=%p leaf.cur=%p leaf.nxt=%p nodes_i=%p "
                        "ni_off=%ld\n",
                        (unsigned)i, (void *)nodes, (void *)leaf.leaf_coords, (void *)leaf.leaf_values,
                        (void *)leaf.leaf_cur, (void *)leaf.leaf_nxt, (void *)&nodes[i],
                        (long)((char *)&nodes[i] - (char *)nodes));
                fflush(stderr);
                /* Pack sources into the leaf buffer. */
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

                fprintf(stderr, "PROBE-BEFORE-MPC2 i=%u kind=%d depth=%u slice=%p\n", (unsigned)i, (int)nodes[i].kind,
                        (unsigned)nodes[i].depth, (void *)mp_slices[i]);
                fflush(stderr);
                const bool mp_ok = multipole_create(settings->order, (unsigned)leaf_buf_size, mp_slices[i], center,
                                                    (unsigned)n_particles, (const real3_t *)leaf.leaf_coords,
                                                    (const real3_t *)leaf.leaf_values, leaf.leaf_cur, leaf.leaf_nxt,
                                                    &nodes[i].data.mp);
                fprintf(stderr, "PROBE-AFTER-MPC2 i=%u kind=%d depth=%u\n", (unsigned)i, (int)nodes[i].kind,
                        (unsigned)nodes[i].depth);
                fflush(stderr);
                /* Force gcc to keep nodes[i].kind live across the call —
                 * the strict-aliasing rules can otherwise let it spill a
                 * temporary into the kind slot via the inactive-union
                 * access patterns. */
                if (nodes[i].kind != BH_NODE_MULTIPOLE)
                    leaf_ok = false;
                if (ml == 0)
                {
                    fprintf(stderr, "POSTCREATE i=%u kind=%d depth=%u half_size=%g\n", (unsigned)i, (int)nodes[i].kind,
                            (unsigned)nodes[i].depth, nodes[i].half_size);
                    fflush(stderr);
                    for (int j = 0; j < 16; ++j)
                        fprintf(stderr, " %02x", (unsigned)((const unsigned char *)&nodes[i])[j]);
                    fprintf(stderr, "\n");
                    fflush(stderr);
                }
                if (!mp_ok)
                    leaf_ok = false;
            }
        }
        bh_free(allocator, mp_leaf_indices);
    }
    if (!leaf_ok)
    {
        bh_free(allocator, topo_to_real);
        bh_free(allocator, mp_slices);
        return false;
    }

    /* --- Upward sweep: aggregate child multipoles into internal nodes. ---
     * Parallelised by depth level: all internal nodes at the same depth
     * can be processed in parallel because they touch disjoint children
     * (the tree partitions the source set). Within a level, the work
     * scales with the number of children, not just node count, so we use
     * a dynamic schedule. --- */
    for (unsigned d = max_depth; d > 0; --d)
    {
#pragma omp parallel for default(none) shared(d, n_topo_nodes, nodes, settings, mp_slices, work_order, shift_exp, pse, \
                                                  n_coeffs, n_threads) schedule(dynamic, 16) num_threads(n_threads)
        for (uint32_t i = 0; i < n_topo_nodes; ++i)
        {
            if (nodes[i].kind != BH_NODE_INTERNAL || nodes[i].depth != d)
                continue;

            /* Zero the internal node's slice and prepare a multipole_t for it. */
            real_t *slice = mp_slices[i];
            memset(slice, 0, 3u * n_coeffs * sizeof(real_t));
            multipole_t internal_mp = {.order = settings->order,
                                       .center = nodes[i].center,
                                       .coeffs_x = slice,
                                       .coeffs_y = slice + n_coeffs,
                                       .coeffs_z = slice + 2u * n_coeffs};

            for (int oct = 0; oct < 8; ++oct)
            {
                bh_node_t *child = nodes[i].data.internal.children[oct];
                if (child == NULL)
                    continue;
                if (child->kind == BH_NODE_PARTICLE)
                    continue;

                /* For both BH_NODE_INTERNAL and BH_NODE_MULTIPOLE children,
                 * the multipole slice is in mp_slices[] (we use a side
                 * table to avoid clobbering children[] in the union). */
                real_t *child_slice = mp_slices[child - nodes];
                if (child_slice == NULL)
                    continue; /* safety guard */
                multipole_t child_mp = {.order = settings->order,
                                        .center = child->data.mp.center,
                                        .coeffs_x = child_slice,
                                        .coeffs_y = child_slice + n_coeffs,
                                        .coeffs_z = child_slice + 2u * n_coeffs};
                multipole_add_shift(&child_mp, &internal_mp, work_order, shift_exp, pse);
            }
        }
    }

    /* --- Populate the output handle. --- */
    {
        fprintf(stderr, "DEBUG-COUNTS-POST n=%u", (unsigned)n_topo_nodes);
        for (uint32_t i = 0; i < n_topo_nodes; ++i)
            fprintf(stderr, " %u:%u", (unsigned)i, (unsigned)nodes[i].particle_count);
        fprintf(stderr, "\n");
        fflush(stderr);
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
    out->scratch = buf_scratch;
    out->shift_exp = shift_exp;
    out->pse = pse;

    bh_free(allocator, topo_to_real);
    bh_free(allocator, mp_slices);
    return true;
}

bool barnes_hut_tree_build(unsigned n_sources, const real3_t CVL_ARRAY_ARG(sources_coords, restrict n_sources),
                           const real3_t CVL_ARRAY_ARG(sources_values, restrict n_sources),
                           const barnes_hut_settings_t CVL_ARRAY_ARG(settings, restrict), void *scratch_buffer,
                           size_t scratch_size, const allocator_t *allocator, void *buffer, size_t buffer_size,
                           barnes_hut_tree_t *out)
{
    /* `barnes_hut_tree_build` is identical to `barnes_hut_tree_insert`:
     * the count pass + buffer sizing is already done inside insert (we just
     * re-validate that the user-supplied buffer is big enough). Callers who
     * want to allocate the buffer themselves use barnes_hut_tree_count() to
     * size it and then call barnes_hut_tree_insert().
     *
     * This is the one-shot entry point: it accepts a pre-allocated buffer
     * and refuses if too small (so the caller can either pre-size via
     * barnes_hut_buffer_size or barnes_hut_tree_count and re-allocate). */
    return barnes_hut_tree_insert(n_sources, sources_coords, sources_values, settings, scratch_buffer, scratch_size,
                                  allocator, buffer, buffer_size, out);
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
