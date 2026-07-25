#include "fmm_tree.h"
#include "fmm_operators.h"

#include <omp.h>

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* Default allocator (libc malloc/free fallback)                      */
/* ------------------------------------------------------------------ */

static void *fmm_default_allocate(void *state, size_t size)
{
    (void)state;
    return malloc(size);
}

static void fmm_default_deallocate(void *state, void *ptr)
{
    (void)state;
    free(ptr);
}

static void *fmm_default_reallocate(void *state, void *ptr, size_t new_size)
{
    (void)state;
    return realloc(ptr, new_size);
}

static const allocator_t FMM_DEFAULT_ALLOCATOR = {
    .allocate = fmm_default_allocate,
    .deallocate = fmm_default_deallocate,
    .reallocate = fmm_default_reallocate,
    .state = NULL,
};

static inline const allocator_t *fmm_allocator(const allocator_t *allocator)
{
    return allocator ? allocator : &FMM_DEFAULT_ALLOCATOR;
}

static inline void *fmm_alloc(const allocator_t *allocator, size_t size)
{
    const allocator_t *a = fmm_allocator(allocator);
    return a->allocate(a->state, size);
}

static inline void fmm_free(const allocator_t *allocator, void *ptr)
{
    if (ptr == NULL)
        return;
    const allocator_t *a = fmm_allocator(allocator);
    a->deallocate(a->state, ptr);
}

/* ------------------------------------------------------------------ */
/* Internal helpers                                                   */
/* ------------------------------------------------------------------ */

/** Return the octant index (0..7) of @p p relative to @p center. */
static inline unsigned fmm_octant_of(const real3_t p, const real3_t center)
{
    return (unsigned)(p.x >= center.x) * 1u + (unsigned)(p.y >= center.y) * 2u + (unsigned)(p.z >= center.z) * 4u;
}

/**
 * @brief Resolve the effective work order. `settings.work_order == 0` falls
 *        back to `settings.order`.
 */
static inline unsigned fmm_resolve_work_order(const fmm_settings_t *settings)
{
    return settings->work_order ? settings->work_order : settings->order;
}

/** @brief Minimum source count to form a multipole leaf. */
static unsigned fmm_multipole_threshold(const fmm_settings_t *settings)
{
    return settings->critical_particle_count < 1u ? 1u : settings->critical_particle_count;
}

/** @brief Source count threshold above which a leaf subdivides (8× critical). */
static unsigned fmm_subdivide_threshold(const fmm_settings_t *settings)
{
    const uint64_t num = (uint64_t)settings->critical_particle_count * 8u;
    return (unsigned)(num < 1u ? 1u : num);
}

/** @brief Decide whether a leaf with @p n particles at @p depth should subdivide. */
static inline bool fmm_should_subdivide(uint32_t n, unsigned depth, const fmm_settings_t *settings)
{
    return n > fmm_subdivide_threshold(settings) && depth < settings->max_depth;
}

/** @brief Decide whether a non-subdividing leaf should become a multipole. */
static inline bool fmm_should_be_multipole(uint32_t n, const fmm_settings_t *settings)
{
    return n > fmm_multipole_threshold(settings);
}

/** @brief Centroid-based subdivision: subdivide if |pos - center| > alpha * half_size. */
static inline bool fmm_should_subdivide_centroid(const real3_t pos, const real3_t center, real_t half_size,
                                                 const fmm_settings_t *settings)
{
    if (settings->alpha_centroid <= 0.0)
        return false;
    const real3_t diff = real3_sub(pos, center);
    const real_t dist = real3_mag(diff);
    return dist > settings->alpha_centroid * half_size;
}

/** @brief Validate build settings. Returns false if any parameter is out of range. */
static bool fmm_settings_valid(const fmm_settings_t *settings)
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
/* Reuse barnes-hut topo_node_t and count pass                        */
/* ------------------------------------------------------------------ */

/* The count pass uses topo_node_t from barnes_hut_tree.h.  We include
 * the header for the type declaration; the count-pass function itself
 * is shared via barnes_hut_count_pass (same logic, same types). */
#include "barnes_hut_tree.h"

/* ------------------------------------------------------------------ */
/* Local scratch view (per-thread)                                    */
/* ------------------------------------------------------------------ */

typedef struct
{
    real_t *leaf_cur;
    real_t *leaf_nxt;
    real_t *leaf_coords;
    real_t *leaf_values;
} fmm_leaf_scratch_t;

/* ------------------------------------------------------------------ */
/* Sizing                                                             */
/* ------------------------------------------------------------------ */

fmm_scratch_sizes_t fmm_size_scratch(unsigned n_sources, const fmm_settings_t *settings)
{
    const size_t topo_bytes = (8u * (size_t)n_sources + 1u) * sizeof(topo_node_t);
    const size_t source_leaf_topo_bytes = (size_t)n_sources * sizeof(uint32_t);
    const size_t source_leaf_real_bytes = (size_t)n_sources * sizeof(unsigned);

    const size_t n_coeffs = multipole_num_coeffs(settings->order);
    const size_t multipole_scratch_sz = multipole_scratch_size(settings->order);
    const size_t leaf_buf_size = (3u * n_coeffs > multipole_scratch_sz ? 3u * n_coeffs : multipole_scratch_sz);
    const size_t leaf_buf_per_thread = 2u * leaf_buf_size * sizeof(real_t);
    const size_t leaf_coords_per_thread = (size_t)n_sources * 3u * sizeof(real_t);
    const size_t leaf_values_per_thread = (size_t)n_sources * 3u * sizeof(real_t);

    const unsigned work_order = fmm_resolve_work_order(settings);
    const size_t shift_exp_per_thread = 3u * (size_t)(work_order + 1) * (size_t)(work_order + 1) * sizeof(real_t);
    const size_t pse_per_thread = 2u * multipole_num_coeffs(work_order) * sizeof(real_t);

    return (fmm_scratch_sizes_t){
        .size_topo = topo_bytes,
        .size_source_leaf_topo = source_leaf_topo_bytes,
        .size_source_leaf_real = source_leaf_real_bytes,
        .size_leaf_buf_per_thread = leaf_buf_per_thread,
        .size_leaf_coords_per_thread = leaf_coords_per_thread,
        .size_leaf_values_per_thread = leaf_values_per_thread,
        .size_shift_exp_per_thread = shift_exp_per_thread,
        .size_pse_per_thread = pse_per_thread,
    };
}

size_t fmm_total_scratch_size(fmm_scratch_sizes_t sizes, unsigned n_threads)
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

size_t fmm_scratch_size(unsigned n_sources, unsigned n_threads, const fmm_settings_t *settings)
{
    if (n_sources == 0 || !fmm_settings_valid(settings) || n_threads < 1)
        return 0;
    const fmm_scratch_sizes_t sizes = fmm_size_scratch(n_sources, settings);
    return fmm_total_scratch_size(sizes, n_threads);
}

fmm_work_sizes_t fmm_size_work_buffer(unsigned n_sources, const fmm_settings_t CVL_ARRAY_ARG(settings, restrict),
                                      fmm_count_res_t count_pass_res)
{
    const unsigned n_internal = count_pass_res.n_internal;
    const unsigned n_multipole = count_pass_res.n_multipole_leaves;
    const unsigned n_particle = count_pass_res.n_particle_leaves;
    const unsigned n_total = n_internal + n_multipole + n_particle;

    const size_t nodes_bytes = (size_t)n_total * sizeof(fmm_node_t);
    const size_t particle_order_bytes = (size_t)n_sources * sizeof(unsigned);
    const size_t multipole_coeffs_bytes =
        (size_t)(n_internal + n_multipole) * 3u * multipole_num_coeffs(settings->order) * sizeof(real_t);
    const size_t topo_to_real_bytes = (size_t)n_total * sizeof(uint32_t);
    const size_t mp_slices_bytes = (size_t)n_total * sizeof(real_t *);
    const size_t leaf_indices_bytes = (size_t)(n_multipole + n_particle) * sizeof(unsigned);
    const size_t local_coeffs_bytes =
        (size_t)(n_multipole + n_particle) * 3u * multipole_num_coeffs(settings->order) * sizeof(real_t);
    const size_t local_slices_bytes = (size_t)n_total * sizeof(real_t *);

    /* Interaction lists: V-list and NF-list each get their own CSR.
     * Pessimistic bound: each list may hold all leaf pairs = n_leaves*(n_leaves-1)/2. */
    const unsigned n_leaves = n_multipole + n_particle;
    const size_t max_per_list = n_leaves > 0 ? (size_t)n_leaves * (size_t)(n_leaves - 1) : 1u;
    const size_t vlist_offsets_bytes = (size_t)(n_leaves + 1) * sizeof(unsigned);
    const size_t vlist_indices_bytes = max_per_list * sizeof(unsigned);
    const size_t nflist_offsets_bytes = (size_t)(n_leaves + 1) * sizeof(unsigned);
    const size_t nflist_indices_bytes = max_per_list * sizeof(unsigned);

    return (fmm_work_sizes_t){
        .nodes_bytes = nodes_bytes,
        .particle_order_bytes = particle_order_bytes,
        .multipole_coeffs_bytes = multipole_coeffs_bytes,
        .topo_to_real_bytes = topo_to_real_bytes,
        .mp_slices_bytes = mp_slices_bytes,
        .leaf_indices_bytes = leaf_indices_bytes,
        .local_coeffs_bytes = local_coeffs_bytes,
        .local_slices_bytes = local_slices_bytes,
        .interaction_lists_bytes =
            vlist_offsets_bytes + vlist_indices_bytes + nflist_offsets_bytes + nflist_indices_bytes,
    };
}

size_t fmm_total_work_size(fmm_work_sizes_t sizes)
{
    return sizes.nodes_bytes + sizes.particle_order_bytes + sizes.multipole_coeffs_bytes + sizes.topo_to_real_bytes +
           sizes.mp_slices_bytes + sizes.leaf_indices_bytes + sizes.local_coeffs_bytes + sizes.local_slices_bytes +
           sizes.interaction_lists_bytes;
}

size_t fmm_buffer_size(unsigned n_sources, const fmm_settings_t *settings)
{
    if (n_sources == 0 || !fmm_settings_valid(settings))
        return 0;
    const size_t max_nodes = (size_t)n_sources + (size_t)((n_sources + 6u) / 7u) + 16u;
    const size_t nodes_bytes = max_nodes * sizeof(fmm_node_t);
    const size_t particle_order_bytes = (size_t)n_sources * sizeof(unsigned);
    const size_t multipole_coeffs_bytes = max_nodes * 3u * multipole_num_coeffs(settings->order) * sizeof(real_t);
    const size_t il_bytes = (size_t)(n_sources + 1) * sizeof(unsigned) + (size_t)n_sources * 2u * sizeof(unsigned);
    return nodes_bytes + particle_order_bytes + multipole_coeffs_bytes + il_bytes;
}

/* ------------------------------------------------------------------ */
/* Scratch partition                                                  */
/* ------------------------------------------------------------------ */

typedef struct
{
    topo_node_t *topo;
    uint32_t *source_leaf_topo;
    unsigned *source_leaf_real;
    real_t *leaf_cur;
    real_t *leaf_nxt;
    real_t *leaf_coords;
    real_t *leaf_values;
    real_t *shift_exp;
    real_t *pse;
    unsigned n_thread_partitions;
} fmm_scratch_t;

static fmm_scratch_t fmm_scratch_partition(unsigned n_threads, void *scratch_buffer, fmm_scratch_sizes_t sizes)
{
    const size_t s_topo = sizes.size_topo;
    const size_t s_leaf_topo = sizes.size_source_leaf_topo;
    const size_t s_leaf_real = sizes.size_source_leaf_real;
    const size_t s_coords = sizes.size_leaf_coords_per_thread * (size_t)n_threads;
    const size_t s_values = sizes.size_leaf_values_per_thread * (size_t)n_threads;
    const size_t s_shift_exp = sizes.size_shift_exp_per_thread * (size_t)n_threads;
    const size_t s_pse = sizes.size_pse_per_thread * (size_t)n_threads;
    const size_t leaf_buf_each = sizes.size_leaf_buf_per_thread / 2;
    const size_t s_cur = (size_t)n_threads * leaf_buf_each;
    const size_t s_nxt = s_cur;

    uint8_t *bp = (uint8_t *)scratch_buffer;
    fmm_scratch_t out = {0};
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
    out.n_thread_partitions = n_threads;
    return out;
}

static fmm_leaf_scratch_t fmm_leaf_scratch_for(const fmm_scratch_t *scratch, unsigned thread_id, unsigned n_sources,
                                               unsigned order)
{
    const size_t n_coeffs = multipole_num_coeffs(order);
    const size_t ms = multipole_scratch_size(order);
    const size_t leaf_buf_size = (3u * n_coeffs > ms ? 3u * n_coeffs : ms);
    const size_t coords_per_thread = (size_t)n_sources * 3u;
    fmm_leaf_scratch_t out = {0};
    out.leaf_cur = scratch->leaf_cur + (size_t)thread_id * leaf_buf_size;
    out.leaf_nxt = scratch->leaf_nxt + (size_t)thread_id * leaf_buf_size;
    out.leaf_coords = scratch->leaf_coords + (size_t)thread_id * coords_per_thread;
    out.leaf_values = scratch->leaf_values + (size_t)thread_id * coords_per_thread;
    return out;
}

/* ------------------------------------------------------------------ */
/* Work buffer partition                                              */
/* ------------------------------------------------------------------ */

typedef struct
{
    fmm_node_t *nodes;
    unsigned *particle_order;
    real_t *multipole_coeffs;
    uint32_t *topo_to_real;
    real_t **mp_slices;
    /* Interaction list storage (CSR).  The interaction_lists_bytes region is
     * subdivided as:
     *   [vlist_offsets: (n_leaves+1) * sizeof(unsigned)]
     *   [vlist_indices:  capacity * sizeof(unsigned)]
     *   [nflist_offsets: (n_leaves+1) * sizeof(unsigned)]
     *   [nflist_indices:  capacity * sizeof(unsigned)]
     *
     * We split the total il bytes evenly: half for V-list, half for NF list.
     * Within each half, the first (n_leaves+1)*4 bytes are offsets, the rest
     * are flat indices.  n_leaves is known only at partition time; we
     * compute it from the sizes struct offset info.
     */
    unsigned *vlist_offsets;
    unsigned *vlist_indices;
    unsigned *nflist_offsets;
    unsigned *nflist_indices;
    size_t vlist_capacity;  /**< Max entries in vlist_indices[]. */
    size_t nflist_capacity; /**< Max entries in nflist_indices[]. */
    unsigned *leaf_indices; /**< Leaf index → node index map [n_leaves]. */
    real_t *local_coeffs;   /**< Contiguous local expansion coefficients. */
    real_t **local_slices;  /**< Per-node local coefficient slice pointers. */
} fmm_work_t;

static fmm_work_t fmm_partition_work(fmm_work_sizes_t sizes, void *buffer, unsigned n_leaves)
{
    uint8_t *bp = (uint8_t *)buffer;
    fmm_work_t out = {0};
    out.nodes = (fmm_node_t *)bp;
    bp += sizes.nodes_bytes;
    out.particle_order = (unsigned *)bp;
    bp += sizes.particle_order_bytes;
    out.multipole_coeffs = (real_t *)bp;
    bp += sizes.multipole_coeffs_bytes;
    out.topo_to_real = (uint32_t *)bp;
    bp += sizes.topo_to_real_bytes;
    out.mp_slices = (real_t **)bp;
    bp += sizes.mp_slices_bytes;

    /* Leaf index reverse map: leaf_id → node index. */
    out.leaf_indices = (unsigned *)bp;
    bp += sizes.leaf_indices_bytes;

    /* Local expansion coefficient storage. */
    out.local_coeffs = (real_t *)bp;
    bp += sizes.local_coeffs_bytes;
    out.local_slices = (real_t **)bp;
    bp += sizes.local_slices_bytes;

    /* Interaction list region: V-list CSR then NF-list CSR.
     * Sizing guarantees each has exactly max_pairs index entries. */
    {
        const size_t offsets_bytes = (size_t)(n_leaves + 1) * sizeof(unsigned);
        const size_t max_per = n_leaves > 0 ? (size_t)n_leaves * (size_t)(n_leaves - 1) : 1u;

        /* V-list: offsets + indices. */
        out.vlist_offsets = (unsigned *)bp;
        bp += offsets_bytes;
        out.vlist_indices = (unsigned *)bp;
        out.vlist_capacity = max_per;
        bp += max_per * sizeof(unsigned);

        /* NF-list: offsets + indices. */
        out.nflist_offsets = (unsigned *)bp;
        bp += offsets_bytes;
        out.nflist_indices = (unsigned *)bp;
        out.nflist_capacity = max_per;
        /* No bp advance needed after NF-list indices — this is the end. */
    }

    return out;
}

/* ------------------------------------------------------------------ */
/* Materialize tree                                                   */
/* ------------------------------------------------------------------ */

static void fmm_materialize_tree(const topo_node_t CVL_ARRAY_ARG(topo, restrict), uint32_t n_topo_nodes,
                                 const fmm_settings_t *settings,
                                 uint32_t CVL_ARRAY_ARG(topo_to_real, restrict n_topo_nodes),
                                 fmm_node_t CVL_ARRAY_ARG(nodes, restrict n_topo_nodes), real_t *multipole_coeffs,
                                 real_t *CVL_ARRAY_ARG(mp_slices_out, restrict n_topo_nodes))
{
    real_t *cursor = multipole_coeffs;
    uint32_t next_real = 0;

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
        fmm_node_t *node = &nodes[next_real];
        next_real += 1;

        node->depth = depth;
        node->center = tn->center;
        node->half_size = tn->half_size;
        node->particle_begin = 0;
        node->particle_count = 0;
        mp_slices_out[next_real - 1] = NULL;

        if (tn->is_internal)
        {
            node->kind = FMM_NODE_INTERNAL;
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
                    node->data.internal.children[k] = (fmm_node_t *)(uintptr_t)((uint32_t)tn->children[k] + 1u);
                }
            }
        }
        else if (fmm_should_be_multipole(tn->particle_count, settings))
        {
            node->kind = FMM_NODE_MULTIPOLE;
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
            node->kind = FMM_NODE_PARTICLE;
        }
    }

#pragma omp parallel for default(none) shared(n_topo_nodes, topo, topo_to_real, nodes) schedule(static, 256)
    for (uint32_t ti = 0; ti < n_topo_nodes; ++ti)
    {
        const topo_node_t *tn = &topo[ti];
        if (!tn->is_internal)
            continue;
        fmm_node_t *node = &nodes[topo_to_real[ti]];
        for (int k = 0; k < 8; ++k)
        {
            if (tn->children[k] < 0)
                continue;
            const uint32_t child_real = topo_to_real[(uint32_t)tn->children[k]];
            node->data.internal.children[k] = &nodes[child_real];
        }
    }
}

/* ------------------------------------------------------------------ */
/* Descend sources                                                    */
/* ------------------------------------------------------------------ */

static void fmm_descend_for_each_source(unsigned n_sources,
                                        const real3_t CVL_ARRAY_ARG(sources_coords, restrict n_sources),
                                        uint32_t n_real_nodes, fmm_node_t CVL_ARRAY_ARG(nodes, restrict n_real_nodes),
                                        unsigned *source_leaf_real, unsigned n_threads)
{
#pragma omp parallel for default(none) shared(n_sources, sources_coords, n_real_nodes, nodes, source_leaf_real)        \
    schedule(static) num_threads(n_threads)
    for (unsigned i = 0; i < n_sources; ++i)
    {
        uint32_t idx = 0;
        while (nodes[idx].kind == FMM_NODE_INTERNAL)
        {
            const real3_t p = sources_coords[i];
            const real3_t c = nodes[idx].center;
            const unsigned oct =
                (unsigned)(p.x >= c.x) * 1u + (unsigned)(p.y >= c.y) * 2u + (unsigned)(p.z >= c.z) * 4u;
            const fmm_node_t *child = nodes[idx].data.internal.children[oct];
            if (child == NULL)
                break;
            idx = (uint32_t)(child - nodes);
        }
        source_leaf_real[i] = idx;
        if (nodes[idx].kind != FMM_NODE_INTERNAL)
        {
#pragma omp atomic
            nodes[idx].particle_count += 1;
        }
    }
}

/* ------------------------------------------------------------------ */
/* Node metadata (particle_begin prefix sums, leaf count)             */
/* ------------------------------------------------------------------ */

static unsigned fmm_compute_node_metadata(uint32_t n_nodes, fmm_node_t CVL_ARRAY_ARG(nodes, restrict n_nodes),
                                          unsigned max_depth, unsigned CVL_ARRAY_ARG(depth_start, restrict),
                                          unsigned CVL_ARRAY_ARG(depth_end, restrict))
{
    unsigned n_mp_leaves = 0;
    unsigned cursor = 0;
    unsigned leaf_cnt = 0;

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

        if (nodes[i].kind != FMM_NODE_INTERNAL)
        {
            nodes[i].particle_begin = cursor;
            cursor += nodes[i].particle_count;
            nodes[i].particle_count = 0;
            nodes[i].leaf_id = (int32_t)(leaf_cnt);
            leaf_cnt += 1;

            if (nodes[i].kind == FMM_NODE_MULTIPOLE)
                n_mp_leaves += 1;
        }
        else
        {
            nodes[i].leaf_id = -1;
        }
    }
    return n_mp_leaves;
}

/* ------------------------------------------------------------------ */
/* Downward pass (materialize + descend + metadata)                    */
/* ------------------------------------------------------------------ */

static unsigned fmm_downward_pass(unsigned n_sources, const real3_t CVL_ARRAY_ARG(sources_coords, restrict n_sources),
                                  const fmm_settings_t CVL_ARRAY_ARG(settings, restrict), fmm_scratch_t scratch,
                                  fmm_work_t work_buffers, fmm_count_res_t count_res, unsigned n_threads)
{
    const size_t n_topo_nodes =
        (size_t)count_res.n_internal + count_res.n_multipole_leaves + count_res.n_particle_leaves;
    fmm_materialize_tree(scratch.topo, (uint32_t)n_topo_nodes, settings, work_buffers.topo_to_real, work_buffers.nodes,
                         work_buffers.multipole_coeffs, work_buffers.mp_slices);
    fmm_descend_for_each_source(n_sources, sources_coords, (uint32_t)n_topo_nodes, work_buffers.nodes,
                                scratch.source_leaf_real, n_threads);
    return fmm_compute_node_metadata((uint32_t)n_topo_nodes, work_buffers.nodes, count_res.max_depth, NULL, NULL);
}

/* ------------------------------------------------------------------ */
/* Leaf centroid computation                                          */
/* ------------------------------------------------------------------ */

static void fmm_compute_leaf_centers(unsigned n_nodes, fmm_node_t CVL_ARRAY_ARG(nodes, restrict n_nodes),
                                     const unsigned CVL_ARRAY_ARG(particle_order, restrict),
                                     const real3_t CVL_ARRAY_ARG(sources_coords, restrict),
                                     const real3_t CVL_ARRAY_ARG(sources_values, restrict), unsigned n_threads)
{
#pragma omp parallel for default(none) shared(n_nodes, nodes, particle_order, sources_coords, sources_values)          \
    schedule(static) num_threads(n_threads)
    for (uint32_t i = 0; i < n_nodes; ++i)
    {
        if (nodes[i].kind == FMM_NODE_INTERNAL)
            continue;
        const unsigned begin = nodes[i].particle_begin;
        const unsigned n_p = nodes[i].particle_count;
        if (n_p == 0)
            continue;
        const unsigned end = begin + n_p;

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

        if (nodes[i].kind == FMM_NODE_MULTIPOLE)
            nodes[i].data.mp.center = nodes[i].center;
    }
}

/* ------------------------------------------------------------------ */
/* Build leaf multipoles (P2M)                                        */
/* ------------------------------------------------------------------ */

static bool fmm_build_leaf_multipoles(unsigned n_topo_nodes, fmm_node_t CVL_ARRAY_ARG(nodes, restrict n_topo_nodes),
                                      const unsigned *restrict particle_order,
                                      const real3_t CVL_ARRAY_ARG(sources_coords, restrict),
                                      const real3_t CVL_ARRAY_ARG(sources_values, restrict),
                                      const fmm_settings_t *settings, fmm_scratch_t scratch, unsigned n_sources,
                                      unsigned n_threads, real_t *CVL_ARRAY_ARG(mp_slices, restrict n_topo_nodes),
                                      unsigned n_mp_leaves,
                                      uint32_t CVL_ARRAY_ARG(mp_leaf_indices, restrict n_mp_leaves))
{
    bool leaf_ok = true;
    unsigned k = 0;
    for (uint32_t i = 0; i < n_topo_nodes; ++i)
    {
        if (nodes[i].kind == FMM_NODE_MULTIPOLE)
            mp_leaf_indices[k++] = i;
    }

    const size_t n_coeffs = multipole_num_coeffs(settings->order);
    const size_t mp_scratch_sz = multipole_scratch_size(settings->order);
    const size_t leaf_buf_size = (3u * n_coeffs > mp_scratch_sz ? 3u * n_coeffs : mp_scratch_sz);

#pragma omp parallel default(none)                                                                                     \
    shared(n_mp_leaves, mp_leaf_indices, nodes, particle_order, sources_coords, sources_values, settings, scratch,     \
               leaf_buf_size, n_sources, leaf_ok, mp_slices) num_threads(n_threads)
    {
        const int tid = omp_get_thread_num();
        const fmm_leaf_scratch_t leaf = fmm_leaf_scratch_for(&scratch, (unsigned)tid, n_sources, settings->order);
#pragma omp for reduction(&& : leaf_ok) schedule(static)
        for (unsigned ml = 0; ml < n_mp_leaves; ++ml)
        {
            const uint32_t i = mp_leaf_indices[ml];
            const size_t n_p = nodes[i].particle_count;

#pragma omp simd
            for (unsigned kk = 0; kk < n_p; ++kk)
            {
                const unsigned src = particle_order[nodes[i].particle_begin + kk];
                leaf.leaf_coords[3u * kk + 0] = sources_coords[src].x;
                leaf.leaf_coords[3u * kk + 1] = sources_coords[src].y;
                leaf.leaf_coords[3u * kk + 2] = sources_coords[src].z;
                leaf.leaf_values[3u * kk + 0] = sources_values[src].x;
                leaf.leaf_values[3u * kk + 1] = sources_values[src].y;
                leaf.leaf_values[3u * kk + 2] = sources_values[src].z;
            }

            const bool mp_ok =
                multipole_create(settings->order, (unsigned)leaf_buf_size, mp_slices[i], nodes[i].center, (unsigned)n_p,
                                 (const real3_t *)leaf.leaf_coords, (const real3_t *)leaf.leaf_values, leaf.leaf_cur,
                                 leaf.leaf_nxt, &nodes[i].data.mp);

            if (nodes[i].kind != FMM_NODE_MULTIPOLE)
                leaf_ok = false;
            if (!mp_ok)
                leaf_ok = false;
        }
    }
    return leaf_ok;
}

/* ------------------------------------------------------------------ */
/* Depth range helper                                                 */
/* ------------------------------------------------------------------ */

static void fmm_compute_depth_ranges(unsigned n_nodes, const fmm_node_t CVL_ARRAY_ARG(nodes, restrict n_nodes),
                                     unsigned max_depth, unsigned CVL_ARRAY_ARG(depth_start, restrict max_depth + 2),
                                     unsigned CVL_ARRAY_ARG(depth_end, restrict max_depth + 2))
{
    for (unsigned d = 0; d <= max_depth + 1; ++d)
    {
        depth_start[d] = n_nodes;
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

/* ------------------------------------------------------------------ */
/* Upward sweep (M2M)                                                 */
/* ------------------------------------------------------------------ */

static void fmm_upward_sweep_level(unsigned depth_start, unsigned depth_end, fmm_node_t CVL_ARRAY_ARG(nodes, restrict),
                                   unsigned order, size_t n_coeffs, real_t *CVL_ARRAY_ARG(mp_slices, restrict),
                                   unsigned work_order, real_t CVL_ARRAY_ARG(shift_exp, restrict),
                                   real_t CVL_ARRAY_ARG(pse, restrict), size_t shift_stride, size_t pse_stride,
                                   const unsigned CVL_ARRAY_ARG(particle_order, restrict),
                                   const real3_t CVL_ARRAY_ARG(sources_coords, restrict),
                                   const real3_t CVL_ARRAY_ARG(sources_values, restrict), const fmm_scratch_t *scratch,
                                   size_t leaf_stride, unsigned n_threads)
{
#pragma omp parallel for default(none)                                                                                 \
    shared(depth_start, depth_end, nodes, order, n_coeffs, mp_slices, work_order, shift_exp, pse, shift_stride,        \
               pse_stride, particle_order, sources_coords, sources_values, scratch, leaf_stride) schedule(dynamic, 16) \
    num_threads(n_threads)
    for (uint32_t i = depth_start; i < depth_end; ++i)
    {
        if (nodes[i].kind != FMM_NODE_INTERNAL)
            continue;

        /* Step 1: Compute |Γ|-weighted centroid from children. */
        {
            real_t cx = 0, cy = 0, cz = 0;
            real_t total_weight = 0.0;
            unsigned n_children = 0;

            for (int oct = 0; oct < 8; ++oct)
            {
                const fmm_node_t *child = nodes[i].data.internal.children[oct];
                if (child == NULL)
                    continue;
                n_children += 1;
                real_t w;
                if (child->kind == FMM_NODE_PARTICLE)
                {
                    w = 0.0;
                    for (unsigned kk = child->particle_begin; kk < child->particle_begin + child->particle_count; ++kk)
                    {
                        const unsigned src = particle_order[kk];
                        w += real3_mag(sources_values[src]);
                    }
                }
                else
                {
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
                real_t ax = 0, ay = 0, az = 0;
                for (int oct = 0; oct < 8; ++oct)
                {
                    const fmm_node_t *child = nodes[i].data.internal.children[oct];
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
        }

        /* Zero the internal node's slice */
        real_t *slice = mp_slices[i];
        memset(slice, 0, 3u * n_coeffs * sizeof(real_t));
        const multipole_t internal_mp = {.order = order,
                                         .center = nodes[i].center,
                                         .coeffs_x = slice,
                                         .coeffs_y = slice + n_coeffs,
                                         .coeffs_z = slice + 2u * n_coeffs};

        const int tid = omp_get_thread_num();
        real_t *particle_cur = scratch->leaf_cur + (size_t)tid * leaf_stride;
        real_t *particle_nxt = scratch->leaf_nxt + (size_t)tid * leaf_stride;
        real_t *my_shift_exp = shift_exp + (size_t)tid * shift_stride;
        real_t *my_pse = pse + (size_t)tid * pse_stride;

        for (int oct = 0; oct < 8; ++oct)
        {
            fmm_node_t *child = nodes[i].data.internal.children[oct];
            if (child == NULL)
                continue;

            if (child->kind == FMM_NODE_PARTICLE)
            {
                for (unsigned kk = child->particle_begin; kk < child->particle_begin + child->particle_count; ++kk)
                {
                    const unsigned src = particle_order[kk];
                    multipole_update(&internal_mp, nodes[i].center, sources_coords[src], sources_values[src],
                                     particle_cur, particle_nxt);
                }
            }
            else
            {
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

/* ------------------------------------------------------------------ */
/* Interaction list computation                                       */
/* ------------------------------------------------------------------ */

/**
 * @brief Build CSR interaction lists for all leaves.
 *
 * For each leaf, two lists are built:
 *  - V-list: well-separated leaves (|center_diff| >= 3 * max(half_a, half_b))
 *  - Near-field list: all other leaves (including self placeholder — self
 *    is handled separately in eval so we skip it here).
 *
 * @param n_nodes         Total number of nodes.
 * @param nodes           Node array.
 * @param n_leaves        Number of leaves.
 * @param leaf_indices    Mapping from leaf index (0..n_leaves-1) to node index.
 * @param vlist_offsets   Output CSR offsets [n_leaves+1].
 * @param vlist_indices   Output flat V-list indices (leaf indices).
 * @param vlist_capacity  Capacity of vlist_indices.
 * @param nflist_offsets  Output CSR offsets [n_leaves+1].
 * @param nflist_indices  Output flat near-field leaf indices.
 * @param nflist_capacity Capacity of nflist_indices.
 * @param out_vlist_count  Output total V-list entries written.
 * @param out_nflist_count Output total near-field entries written.
 */
static void fmm_compute_interaction_lists(unsigned n_nodes, const fmm_node_t CVL_ARRAY_ARG(nodes, restrict),
                                          unsigned n_leaves,
                                          const unsigned CVL_ARRAY_ARG(leaf_indices, restrict n_leaves),
                                          unsigned CVL_ARRAY_ARG(vlist_offsets, restrict n_leaves + 1),
                                          unsigned CVL_ARRAY_ARG(vlist_indices, restrict), size_t vlist_capacity,
                                          unsigned CVL_ARRAY_ARG(nflist_offsets, restrict n_leaves + 1),
                                          unsigned CVL_ARRAY_ARG(nflist_indices, restrict), size_t nflist_capacity,
                                          size_t *out_vlist_count, size_t *out_nflist_count)
{
    (void)n_nodes;
    size_t vlist_total = 0;
    size_t nflist_total = 0;

    for (unsigned li = 0; li < n_leaves; ++li)
    {
        const unsigned idx_a = leaf_indices[li];
        const fmm_node_t *node_a = &nodes[idx_a];
        vlist_offsets[li] = (unsigned)vlist_total;
        nflist_offsets[li] = (unsigned)nflist_total;

        for (unsigned lj = 0; lj < n_leaves; ++lj)
        {
            if (li == lj)
                continue;
            const unsigned idx_b = leaf_indices[lj];
            const fmm_node_t *node_b = &nodes[idx_b];

            const real3_t diff = real3_sub(node_a->center, node_b->center);
            const real_t dist = real3_mag(diff);
            const real_t max_h = (node_a->half_size > node_b->half_size) ? node_a->half_size : node_b->half_size;

            /* V-list criterion: |center_diff| >= 3 * max(half_a, half_b). */
            if (dist >= 3.0 * max_h - 1e-12)
            {
                if (vlist_total < vlist_capacity)
                    vlist_indices[vlist_total] = lj;
                vlist_total++;
            }
            else
            {
                if (nflist_total < nflist_capacity)
                    nflist_indices[nflist_total] = lj;
                nflist_total++;
            }
        }
    }
    vlist_offsets[n_leaves] = (unsigned)vlist_total;
    nflist_offsets[n_leaves] = (unsigned)nflist_total;

    *out_vlist_count = vlist_total;
    *out_nflist_count = nflist_total;
}

/**
 * @brief Rebuild leaf-index map from node leaf_id fields.
 *
 * Iterates nodes and places each non-internal node's index at
 * leaf_indices[node->leaf_id] = node_idx.
 *
 * @param n_nodes      Number of nodes.
 * @param nodes        Node array (leaf_id must already be populated).
 * @param leaf_indices Output array [n_nodes]; leaf_id → node index map.
 * @return Number of leaves found.
 */
static unsigned fmm_build_leaf_index_map(unsigned n_nodes, const fmm_node_t CVL_ARRAY_ARG(nodes, restrict n_nodes),
                                         unsigned CVL_ARRAY_ARG(leaf_indices, restrict))
{
    unsigned cnt = 0;
    for (uint32_t i = 0; i < n_nodes; ++i)
    {
        if (nodes[i].kind != FMM_NODE_INTERNAL)
        {
            const unsigned li = (unsigned)nodes[i].leaf_id;
            leaf_indices[li] = i;
            cnt++;
        }
    }
    return cnt;
}

/* ------------------------------------------------------------------ */
/* Local expansion slice assignment                                   */
/* ------------------------------------------------------------------ */

/**
 * @brief Assign local expansion coefficient slices to all non-internal nodes.
 *
 * Each leaf gets a contiguous slice of 3 * n_coeffs doubles in the
 * local_coeffs arena.  The slice pointer is stored in local_slices[node_idx].
 * The coefficients are zeroed for fresh M2L accumulation.
 *
 * @param n_nodes       Number of nodes.
 * @param nodes         Node array (leaf_id must be populated).
 * @param order         Multipole/local expansion order.
 * @param local_coeffs  Flat arena of local expansion coefficients.
 * @param local_slices  Output per-node local slice pointers [n_nodes].
 */
static void fmm_assign_local_slices(unsigned n_nodes, const fmm_node_t CVL_ARRAY_ARG(nodes, restrict n_nodes),
                                    unsigned order, real_t *local_coeffs,
                                    real_t *CVL_ARRAY_ARG(local_slices, restrict n_nodes))
{
    const size_t n_coeffs = multipole_num_coeffs(order);
    real_t *cursor = local_coeffs;

    for (uint32_t i = 0; i < n_nodes; ++i)
    {
        if (nodes[i].kind != FMM_NODE_INTERNAL)
        {
            local_slices[i] = cursor;
            cursor += 3u * n_coeffs;
        }
        else
        {
            local_slices[i] = NULL;
        }
    }
    /* Zero all local coefficients. */
    const size_t total = (size_t)(cursor - local_coeffs);
    memset(local_coeffs, 0, total * sizeof(real_t));
}

/* ------------------------------------------------------------------ */
/* M2L sweep                                                          */
/* ------------------------------------------------------------------ */

/**
 * @brief Run the M2L (multipole-to-local) conversion sweep.
 *
 * For each leaf, converts all V-list multipoles into a local expansion
 * centred at the leaf's centre and accumulates into local_slices.
 *
 * @param n_nodes         Number of nodes.
 * @param nodes           Node array.
 * @param n_leaves        Number of leaves.
 * @param leaf_indices    Leaf index to node index map [n_leaves].
 * @param order           Multipole/local expansion order.
 * @param work_order      Internal work order for M2L series.
 * @param mp_slices       Per-node multipole coefficient slices.
 * @param local_slices    Per-node local coefficient slices (accumulated).
 * @param vlist_offsets   V-list CSR offsets [n_leaves+1].
 * @param vlist_indices   V-list flat leaf indices.
 * @param scratch         Scratch buffer (per-thread shift_exp/pse).
 * @param n_threads       Number of OpenMP threads.
 */
static void fmm_m2l_sweep(unsigned n_nodes, const fmm_node_t CVL_ARRAY_ARG(nodes, restrict n_nodes), unsigned n_leaves,
                          const unsigned CVL_ARRAY_ARG(leaf_indices, restrict n_leaves), unsigned order,
                          unsigned work_order, real_t *const *restrict mp_slices, real_t *const *restrict local_slices,
                          unsigned CVL_ARRAY_ARG(vlist_offsets, restrict n_leaves + 1),
                          unsigned CVL_ARRAY_ARG(vlist_indices, restrict), const fmm_scratch_t *scratch,
                          unsigned n_threads)
{
    const size_t n_coeffs = multipole_num_coeffs(order);

#pragma omp parallel for default(none) shared(n_leaves, leaf_indices, nodes, order, work_order, mp_slices,             \
                                                  local_slices, vlist_offsets, vlist_indices, scratch, n_coeffs)       \
    schedule(dynamic, 16) num_threads(n_threads)
    for (unsigned li = 0; li < n_leaves; ++li)
    {
        const unsigned ni = leaf_indices[li];
        real_t *local_slice = local_slices[ni];
        if (local_slice == NULL)
            continue;

        const local_expansion_t local = {.order = order,
                                         .center = nodes[ni].center,
                                         .coeffs_x = local_slice,
                                         .coeffs_y = local_slice + n_coeffs,
                                         .coeffs_z = local_slice + 2u * n_coeffs};

        const int tid = omp_get_thread_num();
        real_t *my_shift_exp =
            scratch->shift_exp + (size_t)tid * 3u * (size_t)(work_order + 1) * (size_t)(work_order + 1);
        real_t *my_pse = scratch->pse + (size_t)tid * 2u * multipole_num_coeffs(work_order);

        const unsigned v_start = vlist_offsets[li];
        const unsigned v_end = vlist_offsets[li + 1];
        for (unsigned vi = v_start; vi < v_end; ++vi)
        {
            const unsigned li_src = vlist_indices[vi];
            const unsigned ni_src = leaf_indices[li_src];
            real_t *mp_slice = mp_slices[ni_src];
            if (mp_slice == NULL)
                continue;

            const multipole_t mp = {.order = order,
                                    .center = nodes[ni_src].data.mp.center,
                                    .coeffs_x = mp_slice,
                                    .coeffs_y = mp_slice + n_coeffs,
                                    .coeffs_z = mp_slice + 2u * n_coeffs};
            multipole_to_local(&mp, (local_expansion_t *)&local, work_order, my_shift_exp, my_pse);
        }
    }
}

/* ------------------------------------------------------------------ */
/* L2L downward sweep                                                 */
/* ------------------------------------------------------------------ */

/**
 * @brief Run the L2L (local-to-local) downward sweep.
 *
 * Iterates over all internal nodes (depth-first pre-order, so parents
 * before children), and for each internal node that has a local expansion,
 * shifts it to each non-NULL child.
 *
 * @param n_nodes        Number of nodes.
 * @param nodes          Node array.
 * @param order          Local expansion order.
 * @param work_order     Internal work order for L2L shift.
 * @param local_slices   Per-node local coefficient slices.
 * @param scratch        Scratch buffer (per-thread shift_exp/pse).
 * @param n_threads      Number of OpenMP threads.
 */
static void fmm_downward_l2l_sweep(unsigned n_nodes, fmm_node_t CVL_ARRAY_ARG(nodes, restrict n_nodes), unsigned order,
                                   unsigned work_order, real_t *const *restrict local_slices,
                                   const fmm_scratch_t *scratch, unsigned n_threads)
{
    const size_t n_coeffs = multipole_num_coeffs(order);

    /* Nodes are in DFS pre-order, so parents always have lower indices
     * than their children.  A single parallel-for over all internal nodes
     * is safe because each node writes only to its own children (distinct
     * cache lines).  Parallelise at depth level for locality. */
#pragma omp parallel for default(none) shared(n_nodes, nodes, order, work_order, n_coeffs, local_slices, scratch)      \
    schedule(static, 64) num_threads(n_threads)
    for (uint32_t i = 0; i < n_nodes; ++i)
    {
        if (nodes[i].kind != FMM_NODE_INTERNAL)
            continue;

        real_t *parent_slice = local_slices[i];
        if (parent_slice == NULL)
            continue;

        const local_expansion_t parent_local = {.order = order,
                                                .center = nodes[i].center,
                                                .coeffs_x = parent_slice,
                                                .coeffs_y = parent_slice + n_coeffs,
                                                .coeffs_z = parent_slice + 2u * n_coeffs};

        const int tid = omp_get_thread_num();
        real_t *my_shift_exp =
            scratch->shift_exp + (size_t)tid * 3u * (size_t)(work_order + 1) * (size_t)(work_order + 1);
        real_t *my_pse = scratch->pse + (size_t)tid * 2u * multipole_num_coeffs(work_order);

        for (int k = 0; k < 8; ++k)
        {
            fmm_node_t *child = nodes[i].data.internal.children[k];
            if (child == NULL)
                continue;
            const uint32_t ci = (uint32_t)(child - nodes);
            real_t *child_slice = local_slices[ci];
            if (child_slice == NULL)
                continue;

            local_expansion_t child_local = {.order = order,
                                             .center = child->center,
                                             .coeffs_x = child_slice,
                                             .coeffs_y = child_slice + n_coeffs,
                                             .coeffs_z = child_slice + 2u * n_coeffs};
            local_expansion_shift(&parent_local, &child_local, work_order, my_shift_exp, my_pse);
        }
    }
}

/* ------------------------------------------------------------------ */
/* Particle kernel                                                    */
/* ------------------------------------------------------------------ */

static inline real3_t fmm_particle_kernel(real3_t gamma, real3_t r_vec)
{
    const real_t r2 = real3_dot(r_vec, r_vec);
    if (r2 < 1e-30)
        return (real3_t){.x = 0, .y = 0, .z = 0};
    return real3_mul1(gamma, 1.0 / r2);
}

/* ------------------------------------------------------------------ */
/* Tree-code evaluation (single point)                                */
/* ------------------------------------------------------------------ */

real3_t fmm_tree_eval(const fmm_tree_t *tree, const real3_t CVL_ARRAY_ARG(sources_coords, restrict),
                      const real3_t CVL_ARRAY_ARG(sources_values, restrict), real3_t point,
                      fmm_eval_settings_t eval_settings)
{
    if (tree == NULL || tree->nodes == NULL || tree->n_nodes == 0)
        return (real3_t){.x = 0, .y = 0, .z = 0};

    real3_t result = {.x = 0, .y = 0, .z = 0};

    /* Step 1: Descend to the target's leaf. */
    uint32_t leaf_node_idx = 0;
    {
        uint32_t idx = 0;
        while (tree->nodes[idx].kind == FMM_NODE_INTERNAL)
        {
            const real3_t p = point;
            const real3_t c = tree->nodes[idx].center;
            const unsigned oct =
                (unsigned)(p.x >= c.x) * 1u + (unsigned)(p.y >= c.y) * 2u + (unsigned)(p.z >= c.z) * 4u;
            const fmm_node_t *child = tree->nodes[idx].data.internal.children[oct];
            if (child == NULL)
                break;
            idx = (uint32_t)(child - tree->nodes);
        }
        leaf_node_idx = idx;
    }

    const fmm_node_t *target_leaf = &tree->nodes[leaf_node_idx];
    const int32_t li = target_leaf->leaf_id;
    if (li < 0)
        return result;

    const unsigned order = tree->settings.order;
    const size_t n_coeffs = multipole_num_coeffs(order);
    const unsigned *lindices = tree->leaf_indices;

    /* --- Far-field --- */
    if (eval_settings.mode == FMM_EVAL_FMM && tree->local_coeffs != NULL && tree->local_slices != NULL)
    {
        /* FMM mode: evaluate the leaf's precomputed local expansion. */
        real_t *local_slice = tree->local_slices[leaf_node_idx];
        if (local_slice != NULL)
        {
            const local_expansion_t local = {.order = order,
                                             .center = target_leaf->center,
                                             .coeffs_x = local_slice,
                                             .coeffs_y = local_slice + n_coeffs,
                                             .coeffs_z = local_slice + 2u * n_coeffs};
            result = local_expansion_eval(&local, point);
        }
    }
    else if (tree->vlist_offsets && tree->vlist_indices && tree->leaf_indices)
    {
        /* Tree-code mode: per-V-list multipole_eval. */
        if (li >= 0)
        {
            const unsigned v_start = tree->vlist_offsets[(unsigned)li];
            const unsigned v_end = tree->vlist_offsets[(unsigned)li + 1];
            for (unsigned vi = v_start; vi < v_end; ++vi)
            {
                const unsigned li_other = tree->vlist_indices[vi];
                const unsigned other_node_idx = lindices[li_other];
                real_t *slice = tree->mp_slices[other_node_idx];
                if (slice == NULL)
                    continue;
                const fmm_node_t *other = &tree->nodes[other_node_idx];
                const multipole_t mp = {.order = order,
                                        .center = other->data.mp.center,
                                        .coeffs_x = slice,
                                        .coeffs_y = slice + n_coeffs,
                                        .coeffs_z = slice + 2u * n_coeffs};
                result = real3_add(result, multipole_eval(&mp, point));
            }
        }
    }
    else
    {
        /* Fallback: direct sum over all sources. */
        for (unsigned kk = 0; kk < tree->n_sources; ++kk)
        {
            const real3_t dr = real3_sub(point, sources_coords[kk]);
            result = real3_add(result, fmm_particle_kernel(sources_values[kk], dr));
        }
    }

    /* --- Near-field: direct sum over neighbour leaves + own leaf --- */
    if (tree->nflist_offsets && tree->nflist_indices && li >= 0)
    {
        const unsigned nf_start = tree->nflist_offsets[(unsigned)li];
        const unsigned nf_end = tree->nflist_offsets[(unsigned)li + 1];

        /* Own leaf direct sum. */
        if (target_leaf->kind != FMM_NODE_INTERNAL)
        {
            for (unsigned kk = target_leaf->particle_begin;
                 kk < target_leaf->particle_begin + target_leaf->particle_count; ++kk)
            {
                const unsigned src = tree->particle_order[kk];
                const real3_t dr = real3_sub(point, sources_coords[src]);
                result = real3_add(result, fmm_particle_kernel(sources_values[src], dr));
            }
        }

        /* Neighbour leaves. */
        for (unsigned ni = nf_start; ni < nf_end; ++ni)
        {
            const unsigned li_other = tree->nflist_indices[ni];
            const unsigned other_node_idx = lindices[li_other];
            const fmm_node_t *other = &tree->nodes[other_node_idx];
            for (unsigned kk = other->particle_begin; kk < other->particle_begin + other->particle_count; ++kk)
            {
                const unsigned src = tree->particle_order[kk];
                const real3_t dr = real3_sub(point, sources_coords[src]);
                result = real3_add(result, fmm_particle_kernel(sources_values[src], dr));
            }
        }
    }

    return result;
}

/* ------------------------------------------------------------------ */
/* Batched evaluation                                                 */
/* ------------------------------------------------------------------ */

void fmm_tree_eval_all(const fmm_tree_t *tree, const real3_t CVL_ARRAY_ARG(sources_coords, restrict),
                       const real3_t CVL_ARRAY_ARG(sources_values, restrict), unsigned n_targets,
                       const real3_t CVL_ARRAY_ARG(targets, restrict n_targets),
                       real3_t CVL_ARRAY_ARG(results, restrict n_targets), fmm_eval_settings_t eval_settings,
                       unsigned n_threads)
{
#pragma omp parallel for default(none) shared(tree, sources_coords, sources_values, n_targets, targets, results,       \
                                                  eval_settings) num_threads(n_threads) schedule(static)
    for (unsigned i = 0; i < n_targets; ++i)
    {
        results[i] = fmm_tree_eval(tree, sources_coords, sources_values, targets[i], eval_settings);
    }
}

/* ------------------------------------------------------------------ */
/* fmm_tree_build                                                     */
/* ------------------------------------------------------------------ */

bool fmm_tree_build(unsigned n_sources, unsigned n_threads,
                    const real3_t CVL_ARRAY_ARG(sources_coords, restrict n_sources),
                    const real3_t CVL_ARRAY_ARG(sources_values, restrict n_sources),
                    const fmm_settings_t CVL_ARRAY_ARG(settings, restrict), const allocator_t *allocator,
                    fmm_tree_t *out)
{
    void *scratch_buffer = NULL, *work_buffer = NULL;
    uint32_t *mp_leaf_indices = NULL;
    bool ret = false;

    if (!out || n_sources == 0 || !fmm_settings_valid(settings) || sources_coords == NULL || sources_values == NULL)
        return false;

    const fmm_scratch_sizes_t scratch_sizes = fmm_size_scratch(n_sources, settings);
    const size_t needed_scratch = fmm_total_scratch_size(scratch_sizes, n_threads);
    if (needed_scratch == 0)
        return false;

    scratch_buffer = fmm_alloc(allocator, needed_scratch);
    if (!scratch_buffer)
        return false;

    const fmm_scratch_t scratch = fmm_scratch_partition(n_threads, scratch_buffer, scratch_sizes);
    const size_t topo_bytes = (8u * (size_t)n_sources + 1u) * sizeof(topo_node_t);
    memset(scratch.topo, 0, topo_bytes);

    /* --- Count pass (reuse BH count pass — same topology logic). --- */
    /* We use barnes_hut_count_pass directly since it works with topo_node_t
     * and barnes_hut_settings_t.  ABI compatibility: barnes_hut_settings_t
     * and fmm_settings_t have identical field order (order, critical_particle_count,
     * max_depth, work_order, alpha_centroid). */
    const barnes_hut_settings_t *bh_settings = (const barnes_hut_settings_t *)settings;
    const barnes_hut_count_res_t bh_count =
        barnes_hut_count_pass(n_sources, sources_coords, bh_settings, scratch.topo, scratch.source_leaf_topo);

    const fmm_count_res_t count_res = {
        .n_internal = bh_count.n_internal,
        .n_multipole_leaves = bh_count.n_multipole_leaves,
        .n_particle_leaves = bh_count.n_particle_leaves,
        .max_depth = bh_count.max_depth,
    };

    /* --- Size and allocate work buffer. --- */
    const fmm_work_sizes_t work_sizes = fmm_size_work_buffer(n_sources, settings, count_res);
    const size_t total_work_size = fmm_total_work_size(work_sizes);
    work_buffer = fmm_alloc(allocator, total_work_size);
    if (!work_buffer)
        goto cleanup;

    const fmm_work_t work =
        fmm_partition_work(work_sizes, work_buffer, count_res.n_multipole_leaves + count_res.n_particle_leaves);

    /* --- Downward pass. --- */
    const unsigned n_mp_leaves =
        fmm_downward_pass(n_sources, sources_coords, settings, scratch, work, count_res, n_threads);

    /* --- Fill particle_order. --- */
#pragma omp parallel for default(none) shared(n_sources, scratch, work) schedule(static) num_threads(n_threads)
    for (unsigned i = 0; i < n_sources; ++i)
    {
        const unsigned leaf_idx = scratch.source_leaf_real[i];
        fmm_node_t *leaf = work.nodes + leaf_idx;
        unsigned cnt;
#pragma omp atomic capture
        {
            cnt = leaf->particle_count;
            leaf->particle_count += 1;
        }
        const unsigned slot = leaf->particle_begin + cnt;
        work.particle_order[slot] = i;
    }

    /* --- Leaf centroids. --- */
    {
        const size_t n_total =
            (size_t)count_res.n_internal + count_res.n_multipole_leaves + count_res.n_particle_leaves;
        fmm_compute_leaf_centers((unsigned)n_total, work.nodes, work.particle_order, sources_coords, sources_values,
                                 n_threads);
    }

    /* --- Build leaf multipoles. --- */
    if (n_mp_leaves > 0)
    {
        mp_leaf_indices = (uint32_t *)fmm_alloc(allocator, (size_t)n_mp_leaves * sizeof(uint32_t));
        if (!mp_leaf_indices)
            goto cleanup;

        if (!fmm_build_leaf_multipoles(
                (unsigned)(count_res.n_internal + count_res.n_multipole_leaves + count_res.n_particle_leaves),
                work.nodes, work.particle_order, sources_coords, sources_values, settings, scratch, n_sources,
                n_threads, work.mp_slices, n_mp_leaves, mp_leaf_indices))
            goto cleanup;

        fmm_free(allocator, mp_leaf_indices);
        mp_leaf_indices = NULL;
    }

    /* --- Upward sweep. --- */
    {
        const unsigned order = settings->order;
        const size_t n_coeffs = multipole_num_coeffs(order);
        const unsigned work_order = fmm_resolve_work_order(settings);
        const size_t leaf_stride =
            (3u * n_coeffs > multipole_scratch_size(order)) ? (3u * n_coeffs) : multipole_scratch_size(order);
        const size_t shift_stride = 3u * (size_t)(work_order + 1) * (size_t)(work_order + 1);
        const size_t pse_stride = 2u * multipole_num_coeffs(work_order);

        unsigned depth_start[256], depth_end[256];
        fmm_compute_depth_ranges(
            (unsigned)(count_res.n_internal + count_res.n_multipole_leaves + count_res.n_particle_leaves), work.nodes,
            count_res.max_depth, depth_start, depth_end);

        for (unsigned d = count_res.max_depth;; --d)
        {
            fmm_upward_sweep_level(depth_start[d], depth_end[d], work.nodes, order, n_coeffs, work.mp_slices,
                                   work_order, scratch.shift_exp, scratch.pse, shift_stride, pse_stride,
                                   work.particle_order, sources_coords, sources_values, &scratch, leaf_stride,
                                   n_threads);
            if (d == 0)
                break;
        }
    }

    /* --- Build interaction lists. --- */
    {
        const unsigned n_leaves = count_res.n_multipole_leaves + count_res.n_particle_leaves;

        /* Build leaf-index map into the persistent work buffer. */
        const unsigned n_total =
            (unsigned)(count_res.n_internal + count_res.n_multipole_leaves + count_res.n_particle_leaves);
        fmm_build_leaf_index_map(n_total, work.nodes, work.leaf_indices);

        size_t vlist_count = 0, nflist_count = 0;
        fmm_compute_interaction_lists(n_total, work.nodes, n_leaves, work.leaf_indices, work.vlist_offsets,
                                      work.vlist_indices, work.vlist_capacity, work.nflist_offsets, work.nflist_indices,
                                      work.nflist_capacity, &vlist_count, &nflist_count);

        out->leaf_indices = work.leaf_indices;
        out->vlist_offsets = work.vlist_offsets;
        out->vlist_indices = work.vlist_indices;
        out->nflist_offsets = work.nflist_offsets;
        out->nflist_indices = work.nflist_indices;
        out->vlist_count = vlist_count;
        out->nflist_count = nflist_count;
        out->n_leaves = n_leaves;
    }

    /* --- M2L + L2L sweeps (FMM mode). --- */
    {
        const unsigned order = settings->order;
        const unsigned work_order = fmm_resolve_work_order(settings);
        const unsigned n_leaves = count_res.n_multipole_leaves + count_res.n_particle_leaves;

        /* Assign local expansion slices. */
        const unsigned n_total =
            (unsigned)(count_res.n_internal + count_res.n_multipole_leaves + count_res.n_particle_leaves);
        fmm_assign_local_slices(n_total, work.nodes, order, work.local_coeffs, work.local_slices);

        /* M2L: convert V-list multipoles to local expansions at each leaf. */
        fmm_m2l_sweep(n_total, work.nodes, n_leaves, work.leaf_indices, order, work_order, work.mp_slices,
                      work.local_slices, work.vlist_offsets, work.vlist_indices, &scratch, n_threads);

        /* L2L: propagate local expansions from parents to children. */
        fmm_downward_l2l_sweep(n_total, work.nodes, order, work_order, work.local_slices, &scratch, n_threads);

        out->local_coeffs = work.local_coeffs;
        out->local_slices = work.local_slices;
    }

    /* --- Populate tree handle. --- */
    out->settings = *settings;
    out->root_center = work.nodes[0].center;
    out->root_half_size = work.nodes[0].half_size;
    out->n_sources = n_sources;
    out->n_nodes = (unsigned)(count_res.n_internal + count_res.n_multipole_leaves + count_res.n_particle_leaves);
    out->n_internal = count_res.n_internal;
    out->n_multipole_leaves = count_res.n_multipole_leaves;
    out->n_particle_leaves = count_res.n_particle_leaves;
    out->max_depth_reached = count_res.max_depth;
    out->buffer = (uint8_t *)work_buffer;
    out->buffer_size = total_work_size;
    out->nodes = work.nodes;
    out->particle_order = work.particle_order;
    out->multipole_coeffs = work.multipole_coeffs;
    out->mp_slices = work.mp_slices;

    work_buffer = NULL;
    ret = true;

cleanup:
    if (mp_leaf_indices)
        fmm_free(allocator, mp_leaf_indices);
    if (work_buffer)
        fmm_free(allocator, work_buffer);
    if (scratch_buffer)
        fmm_free(allocator, scratch_buffer);
    return ret;
}

bool fmm_tree_insert(unsigned n_sources, unsigned n_threads,
                     const real3_t CVL_ARRAY_ARG(sources_coords, restrict n_sources),
                     const real3_t CVL_ARRAY_ARG(sources_values, restrict n_sources),
                     const fmm_settings_t CVL_ARRAY_ARG(settings, restrict), void *scratch_buffer, size_t scratch_size,
                     const allocator_t *allocator, void *buffer, size_t buffer_size, fmm_tree_t *out)
{
    if (!buffer || !out || !scratch_buffer)
        return false;
    if (n_sources == 0 || !fmm_settings_valid(settings))
        return false;
    if (sources_coords == NULL || sources_values == NULL)
        return false;

    const fmm_scratch_sizes_t scratch_sizes = fmm_size_scratch(n_sources, settings);
    const size_t needed_scratch = fmm_total_scratch_size(scratch_sizes, n_threads);
    if (scratch_size < needed_scratch)
        return false;

    const fmm_scratch_t scratch = fmm_scratch_partition(n_threads, scratch_buffer, scratch_sizes);
    const size_t topo_bytes = (8u * (size_t)n_sources + 1u) * sizeof(topo_node_t);
    memset(scratch.topo, 0, topo_bytes);

    const barnes_hut_settings_t *bh_settings = (const barnes_hut_settings_t *)settings;
    const barnes_hut_count_res_t bh_count =
        barnes_hut_count_pass(n_sources, sources_coords, bh_settings, scratch.topo, scratch.source_leaf_topo);

    const fmm_count_res_t count_res = {
        .n_internal = bh_count.n_internal,
        .n_multipole_leaves = bh_count.n_multipole_leaves,
        .n_particle_leaves = bh_count.n_particle_leaves,
        .max_depth = bh_count.max_depth,
    };

    const fmm_work_sizes_t work_sizes = fmm_size_work_buffer(n_sources, settings, count_res);
    if (buffer_size < fmm_total_work_size(work_sizes))
        return false;

    const unsigned n_leaves = count_res.n_multipole_leaves + count_res.n_particle_leaves;
    const fmm_work_t work = fmm_partition_work(work_sizes, buffer, n_leaves);

    /* Same pipeline as build above (minus scratch/work allocation). */
    const unsigned n_mp_leaves =
        fmm_downward_pass(n_sources, sources_coords, settings, scratch, work, count_res, n_threads);

#pragma omp parallel for default(none) shared(n_sources, scratch, work) schedule(static) num_threads(n_threads)
    for (unsigned i = 0; i < n_sources; ++i)
    {
        const unsigned leaf_idx = scratch.source_leaf_real[i];
        fmm_node_t *leaf = work.nodes + leaf_idx;
        unsigned cnt;
#pragma omp atomic capture
        {
            cnt = leaf->particle_count;
            leaf->particle_count += 1;
        }
        const unsigned slot = leaf->particle_begin + cnt;
        work.particle_order[slot] = i;
    }

    {
        const size_t n_total =
            (size_t)count_res.n_internal + count_res.n_multipole_leaves + count_res.n_particle_leaves;
        fmm_compute_leaf_centers((unsigned)n_total, work.nodes, work.particle_order, sources_coords, sources_values,
                                 n_threads);
    }

    if (n_mp_leaves > 0)
    {
        uint32_t *mp_idx = (uint32_t *)fmm_alloc(allocator, (size_t)n_mp_leaves * sizeof(uint32_t));
        if (!mp_idx)
            return false;

        const bool ok = fmm_build_leaf_multipoles(
            (unsigned)(count_res.n_internal + count_res.n_multipole_leaves + count_res.n_particle_leaves), work.nodes,
            work.particle_order, sources_coords, sources_values, settings, scratch, n_sources, n_threads,
            work.mp_slices, n_mp_leaves, mp_idx);
        fmm_free(allocator, mp_idx);
        if (!ok)
            return false;
    }

    {
        const unsigned order = settings->order;
        const size_t n_coeffs = multipole_num_coeffs(order);
        const unsigned work_order = fmm_resolve_work_order(settings);
        const size_t leaf_stride =
            (3u * n_coeffs > multipole_scratch_size(order)) ? (3u * n_coeffs) : multipole_scratch_size(order);
        const size_t shift_stride = 3u * (size_t)(work_order + 1) * (size_t)(work_order + 1);
        const size_t pse_stride = 2u * multipole_num_coeffs(work_order);

        unsigned depth_start[256], depth_end[256];
        fmm_compute_depth_ranges(
            (unsigned)(count_res.n_internal + count_res.n_multipole_leaves + count_res.n_particle_leaves), work.nodes,
            count_res.max_depth, depth_start, depth_end);

        for (unsigned d = count_res.max_depth;; --d)
        {
            fmm_upward_sweep_level(depth_start[d], depth_end[d], work.nodes, order, n_coeffs, work.mp_slices,
                                   work_order, scratch.shift_exp, scratch.pse, shift_stride, pse_stride,
                                   work.particle_order, sources_coords, sources_values, &scratch, leaf_stride,
                                   n_threads);
            if (d == 0)
                break;
        }
    }

    /* Build interaction lists. */
    {
        const unsigned n_leaves = count_res.n_multipole_leaves + count_res.n_particle_leaves;
        const unsigned n_total =
            (unsigned)(count_res.n_internal + count_res.n_multipole_leaves + count_res.n_particle_leaves);
        fmm_build_leaf_index_map(n_total, work.nodes, work.leaf_indices);

        size_t vlist_count = 0, nflist_count = 0;
        fmm_compute_interaction_lists(n_total, work.nodes, n_leaves, work.leaf_indices, work.vlist_offsets,
                                      work.vlist_indices, work.vlist_capacity, work.nflist_offsets, work.nflist_indices,
                                      work.nflist_capacity, &vlist_count, &nflist_count);

        out->leaf_indices = work.leaf_indices;
        out->vlist_offsets = work.vlist_offsets;
        out->vlist_indices = work.vlist_indices;
        out->nflist_offsets = work.nflist_offsets;
        out->nflist_indices = work.nflist_indices;
        out->vlist_count = vlist_count;
        out->nflist_count = nflist_count;
        out->n_leaves = n_leaves;
    }

    /* --- M2L + L2L sweeps (FMM mode). --- */
    {
        const unsigned order = settings->order;
        const unsigned work_order = fmm_resolve_work_order(settings);
        const unsigned n_leaves = count_res.n_multipole_leaves + count_res.n_particle_leaves;
        const unsigned n_total =
            (unsigned)(count_res.n_internal + count_res.n_multipole_leaves + count_res.n_particle_leaves);

        fmm_assign_local_slices(n_total, work.nodes, order, work.local_coeffs, work.local_slices);
        fmm_m2l_sweep(n_total, work.nodes, n_leaves, work.leaf_indices, order, work_order, work.mp_slices,
                      work.local_slices, work.vlist_offsets, work.vlist_indices, &scratch, n_threads);
        fmm_downward_l2l_sweep(n_total, work.nodes, order, work_order, work.local_slices, &scratch, n_threads);

        out->local_coeffs = work.local_coeffs;
        out->local_slices = work.local_slices;
    }

    out->settings = *settings;
    out->root_center = work.nodes[0].center;
    out->root_half_size = work.nodes[0].half_size;
    out->n_sources = n_sources;
    out->n_nodes = (unsigned)(count_res.n_internal + count_res.n_multipole_leaves + count_res.n_particle_leaves);
    out->n_internal = count_res.n_internal;
    out->n_multipole_leaves = count_res.n_multipole_leaves;
    out->n_particle_leaves = count_res.n_particle_leaves;
    out->max_depth_reached = count_res.max_depth;
    out->buffer = (uint8_t *)buffer;
    out->buffer_size = buffer_size;
    out->nodes = work.nodes;
    out->particle_order = work.particle_order;
    out->multipole_coeffs = work.multipole_coeffs;
    out->mp_slices = work.mp_slices;

    return true;
}

/* ------------------------------------------------------------------ */
/* Inspection                                                         */
/* ------------------------------------------------------------------ */

unsigned fmm_tree_n_nodes(const fmm_tree_t *tree)
{
    return tree ? tree->n_nodes : 0;
}

size_t fmm_tree_memory_bytes(const fmm_tree_t *tree)
{
    return tree ? tree->buffer_size : 0;
}
