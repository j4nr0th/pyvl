#include "octree.h"

#include <omp.h>

#include <assert.h>

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* ================================================================ */
/* Internal helpers                                                 */
/* ================================================================ */

static inline unsigned octant_of(real3_t p, real3_t center)
{
    return (unsigned)(p.x >= center.x) * 1u + (unsigned)(p.y >= center.y) * 2u + (unsigned)(p.z >= center.z) * 4u;
}

static inline unsigned multipole_threshold(const octree_settings_t *s)
{
    return s->critical_particle_count < 1u ? 1u : s->critical_particle_count;
}

static inline unsigned subdivide_threshold(const octree_settings_t *s)
{
    const uint64_t num = (uint64_t)s->critical_particle_count * 8u;
    return (unsigned)(num < 1u ? 1u : num);
}

static inline bool should_subdivide(uint32_t n, unsigned depth, const octree_settings_t *s)
{
    return n > subdivide_threshold(s) && depth < s->max_depth;
}

static inline bool should_be_multipole(uint32_t n, const octree_settings_t *s)
{
    return n > multipole_threshold(s);
}

static inline bool should_subdivide_centroid(real3_t pos, real3_t center, real_t half_size, const octree_settings_t *s)
{
    if (s->alpha_centroid <= 0.0)
        return false;
    const real3_t diff = real3_sub(pos, center);
    return real3_mag(diff) > s->alpha_centroid * half_size;
}

/* ================================================================ */
/* Default allocator implementation                                 */
/* ================================================================ */

static void *cvl_default_allocate(void *state, size_t size)
{
    (void)state;
    return malloc(size);
}
static void cvl_default_deallocate(void *state, void *ptr)
{
    (void)state;
    free(ptr);
}
static void *cvl_default_reallocate(void *state, void *ptr, size_t new_size)
{
    (void)state;
    return realloc(ptr, new_size);
}
const allocator_t CVL_DEFAULT_ALLOCATOR = {
    .allocate = cvl_default_allocate,
    .deallocate = cvl_default_deallocate,
    .reallocate = cvl_default_reallocate,
    .state = NULL,
};

static inline bool settings_valid(const octree_settings_t *s)
{
    return s && s->order >= 1 && s->critical_particle_count >= 1 && s->max_depth >= 1 && s->alpha_centroid >= 0.0;
}

/* ================================================================ */
/* Count pass                                                       */
/* ================================================================ */

octree_count_t octree_count_pass(unsigned n_sources, const real3_t sources_coords[restrict n_sources],
                                 const octree_settings_t *settings, topo_node_t *topo, uint32_t *source_leaf)
{
    topo[0] = (topo_node_t){
        .children = {-1, -1, -1, -1, -1, -1, -1, -1},
        .particle_count = 0,
        .is_internal = 1,
        .depth = 0,
        .center = {.x = 0, .y = 0, .z = 0},
        .half_size = 0,
    };

    real3_t bbox_min = sources_coords[0];
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

    uint32_t next_node = 1;
    uint32_t max_depth = 0;

    for (unsigned i = 0; i < n_sources; ++i)
    {
        const real3_t p = sources_coords[i];
        uint32_t node_idx = 0;
        unsigned depth = 0;

        while (topo[node_idx].is_internal)
        {
            const unsigned oct = octant_of(p, topo[node_idx].center);
            int32_t child = topo[node_idx].children[oct];
            if (child < 0)
            {
                child = (int32_t)next_node;
                assert(next_node < 8u * n_sources + 1u);
                next_node++;
                topo[child] = (topo_node_t){
                    .children = {-1, -1, -1, -1, -1, -1, -1, -1},
                    .particle_count = 0,
                    .is_internal = 0,
                    .depth = (uint8_t)(topo[node_idx].depth + 1),
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
            depth = topo[node_idx].depth;
        }

        topo[node_idx].particle_count += 1;

        if (should_subdivide(topo[node_idx].particle_count, depth, settings) ||
            should_subdivide_centroid(p, topo[node_idx].center, topo[node_idx].half_size, settings))
        {
            topo[node_idx].is_internal = 1;
            for (int k = 0; k < 8; ++k)
                topo[node_idx].children[k] = -1;
            topo[node_idx].particle_count = 0;

            while (topo[node_idx].is_internal)
            {
                const unsigned oct = octant_of(p, topo[node_idx].center);
                int32_t child = topo[node_idx].children[oct];
                if (child < 0)
                {
                    child = (int32_t)next_node;
                    assert(next_node < 8u * n_sources + 1u);
                    next_node++;
                    topo[child] = (topo_node_t){
                        .children = {-1, -1, -1, -1, -1, -1, -1, -1},
                        .particle_count = 0,
                        .is_internal = 0,
                        .depth = (uint8_t)(topo[node_idx].depth + 1),
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
                depth = topo[node_idx].depth;
            }
            topo[node_idx].particle_count += 1;
        }

        source_leaf[i] = node_idx;
        if (depth > max_depth)
            max_depth = depth;
    }

    unsigned n_internal = 0, n_multipole = 0, n_particle = 0;
    for (uint32_t i = 0; i < next_node; ++i)
    {
        if (topo[i].is_internal)
            n_internal += 1;
        else if (should_be_multipole(topo[i].particle_count, settings))
            n_multipole += 1;
        else
            n_particle += 1;
    }

    return (octree_count_t){
        .n_internal = n_internal,
        .n_multipole_leaves = n_multipole,
        .n_particle_leaves = n_particle,
        .max_depth = max_depth,
    };
}

/* ================================================================ */
/* Scratch sizing                                                   */
/* ================================================================ */

octree_scratch_sizes_t octree_size_scratch(unsigned n_sources, const octree_settings_t *settings)
{
    const size_t topo_bytes = (8u * (size_t)n_sources + 1u) * sizeof(topo_node_t);
    const size_t source_leaf_topo_bytes = (size_t)n_sources * sizeof(uint32_t);
    const size_t source_leaf_real_bytes = (size_t)n_sources * sizeof(unsigned);
    const size_t n_coeffs = multipole_num_coeffs(settings->order);
    const size_t ms = multipole_scratch_size(settings->order);
    const size_t leaf_buf_size = (3u * n_coeffs > ms ? 3u * n_coeffs : ms);
    const size_t leaf_buf_per_thread = 2u * leaf_buf_size * sizeof(real_t);
    const size_t leaf_coords_per_thread = (size_t)n_sources * 3u * sizeof(real_t);
    const size_t leaf_values_per_thread = (size_t)n_sources * 3u * sizeof(real_t);
    const unsigned work_order = octree_resolve_work_order(settings);
    const size_t shift_exp_per_thread = 3u * (size_t)(work_order + 1) * (size_t)(work_order + 1) * sizeof(real_t);
    const size_t pse_per_thread = 2u * multipole_num_coeffs(work_order) * sizeof(real_t);

    return (octree_scratch_sizes_t){
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

size_t octree_total_scratch_size(octree_scratch_sizes_t sizes, unsigned n_threads)
{
    if (n_threads < 1)
        return 0;
    return sizes.size_topo + sizes.size_source_leaf_topo + sizes.size_source_leaf_real +
           sizes.size_leaf_buf_per_thread * (size_t)n_threads + sizes.size_leaf_coords_per_thread * (size_t)n_threads +
           sizes.size_leaf_values_per_thread * (size_t)n_threads + sizes.size_shift_exp_per_thread * (size_t)n_threads +
           sizes.size_pse_per_thread * (size_t)n_threads;
}

size_t octree_scratch_size(unsigned n_sources, unsigned n_threads, const octree_settings_t *settings)
{
    if (n_sources == 0 || !settings_valid(settings) || n_threads < 1)
        return 0;
    return octree_total_scratch_size(octree_size_scratch(n_sources, settings), n_threads);
}

octree_scratch_t octree_scratch_partition(unsigned n_threads, void *buffer, octree_scratch_sizes_t sizes)
{
    uint8_t *bp = (uint8_t *)buffer;
    octree_scratch_t out = {0};
    out.topo = (topo_node_t *)bp;
    bp += sizes.size_topo;
    out.source_leaf_topo = (uint32_t *)bp;
    bp += sizes.size_source_leaf_topo;
    out.source_leaf_real = (unsigned *)bp;
    bp += sizes.size_source_leaf_real;
    const size_t buf_each = sizes.size_leaf_buf_per_thread / 2;
    out.leaf_cur = (real_t *)bp;
    bp += (size_t)n_threads * buf_each;
    out.leaf_nxt = (real_t *)bp;
    bp += (size_t)n_threads * buf_each;
    out.leaf_coords = (real_t *)bp;
    bp += (size_t)n_threads * sizes.size_leaf_coords_per_thread;
    out.leaf_values = (real_t *)bp;
    bp += (size_t)n_threads * sizes.size_leaf_values_per_thread;
    out.shift_exp = (real_t *)bp;
    bp += (size_t)n_threads * sizes.size_shift_exp_per_thread;
    out.pse = (real_t *)bp;
    bp += (size_t)n_threads * sizes.size_pse_per_thread;
    out.n_thread_partitions = n_threads;
    return out;
}

/* ================================================================ */
/* Work buffer sizing                                               */
/* ================================================================ */

octree_base_work_sizes_t octree_size_work_buffer(unsigned n_sources, const octree_settings_t settings[restrict],
                                                 octree_count_t count)
{
    const unsigned n_total = count.n_internal + count.n_multipole_leaves + count.n_particle_leaves;
    return (octree_base_work_sizes_t){
        .nodes_bytes = (size_t)n_total * sizeof(octree_node_t),
        .particle_order_bytes = (size_t)n_sources * sizeof(unsigned),
        .multipole_coeffs_bytes = (size_t)(count.n_internal + count.n_multipole_leaves) * 3u *
                                  multipole_num_coeffs(settings->order) * sizeof(real_t),
        .topo_to_real_bytes = (size_t)n_total * sizeof(uint32_t),
        .mp_slices_bytes = (size_t)n_total * sizeof(real_t *),
    };
}

size_t octree_total_work_size(octree_base_work_sizes_t sizes)
{
    return sizes.nodes_bytes + sizes.particle_order_bytes + sizes.multipole_coeffs_bytes + sizes.topo_to_real_bytes +
           sizes.mp_slices_bytes;
}

size_t octree_buffer_size(unsigned n_sources, const octree_settings_t *settings)
{
    if (n_sources == 0 || !settings_valid(settings))
        return 0;
    const size_t max_nodes = (size_t)n_sources + (size_t)((n_sources + 6u) / 7u) + 16u;
    return max_nodes * sizeof(octree_node_t) + (size_t)n_sources * sizeof(unsigned) +
           max_nodes * 3u * multipole_num_coeffs(settings->order) * sizeof(real_t);
}

/* ================================================================ */
/* Materialise                                                      */
/* ================================================================ */

void octree_materialize(const topo_node_t topo[restrict], uint32_t n_topo_nodes, const octree_settings_t *settings,
                        uint32_t topo_to_real[restrict n_topo_nodes], octree_node_t *nodes, real_t *multipole_coeffs,
                        real_t *mp_slices[restrict n_topo_nodes], unsigned n_threads)
{
    const unsigned order = settings->order;
    const size_t n_coeffs = multipole_num_coeffs(order);
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
        octree_node_t *node = &nodes[next_real];
        topo_to_real[ti] = next_real;
        next_real += 1;

        node->depth = depth;
        node->center = tn->center;
        node->geom_center = tn->center;
        node->half_size = tn->half_size;
        node->particle_begin = 0;
        node->particle_count = 0;
        node->leaf_id = -1;
        mp_slices[next_real - 1] = NULL;

        if (tn->is_internal)
        {
            node->kind = OCTREE_NODE_INTERNAL;
            mp_slices[next_real - 1] = cursor;
            cursor += 3u * n_coeffs;
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
                    node->data.internal.children[k] = (octree_node_t *)(uintptr_t)((uint32_t)tn->children[k] + 1u);
                }
            }
        }
        else if (should_be_multipole(tn->particle_count, settings))
        {
            node->kind = OCTREE_NODE_MULTIPOLE;
            node->data.mp.order = order;
            node->data.mp.center = tn->center;
            node->data.mp.coeffs_x = cursor;
            cursor += n_coeffs;
            node->data.mp.coeffs_y = cursor;
            cursor += n_coeffs;
            node->data.mp.coeffs_z = cursor;
            cursor += n_coeffs;
            mp_slices[next_real - 1] = node->data.mp.coeffs_x;
        }
        else
        {
            node->kind = OCTREE_NODE_PARTICLE;
        }
    }

    /* Second pass: resolve child pointers. */
#pragma omp parallel for default(none) shared(n_topo_nodes, topo, topo_to_real, nodes) schedule(static, 256)           \
    num_threads(n_threads)
    for (uint32_t ti = 0; ti < n_topo_nodes; ++ti)
    {
        if (!topo[ti].is_internal)
            continue;
        octree_node_t *node = &nodes[topo_to_real[ti]];
        for (int k = 0; k < 8; ++k)
        {
            if (topo[ti].children[k] < 0)
                continue;
            node->data.internal.children[k] = &nodes[topo_to_real[(uint32_t)topo[ti].children[k]]];
        }
    }
}

/* ================================================================ */
/* Descend                                                          */
/* ================================================================ */

void octree_descend(unsigned n_sources, const real3_t sources_coords[restrict n_sources], octree_node_t *nodes,
                    unsigned *source_leaf_real, unsigned n_threads)
{
#pragma omp parallel for default(none) shared(n_sources, sources_coords, nodes, source_leaf_real) schedule(static)     \
    num_threads(n_threads)
    for (unsigned i = 0; i < n_sources; ++i)
    {
        uint32_t idx = 0;
        while (nodes[idx].kind == OCTREE_NODE_INTERNAL)
        {
            const real3_t p = sources_coords[i];
            const real3_t c = nodes[idx].center;
            const unsigned oct = octant_of(p, c);
            octree_node_t *child = nodes[idx].data.internal.children[oct];
            if (child == NULL)
                break;
            idx = (uint32_t)(child - nodes);
        }
        source_leaf_real[i] = idx;
        if (nodes[idx].kind != OCTREE_NODE_INTERNAL)
        {
#pragma omp atomic
            nodes[idx].particle_count += 1;
        }
    }
}

/* ================================================================ */
/* Metadata                                                         */
/* ================================================================ */

unsigned octree_compute_metadata(uint32_t n_nodes, octree_node_t *nodes, unsigned max_depth,
                                 unsigned depth_start[restrict], unsigned depth_end[restrict])
{
    unsigned n_mp_leaves = 0;
    unsigned cursor = 0;

    if (depth_start && depth_end)
        for (unsigned d = 0; d <= max_depth + 1; ++d)
            depth_start[d] = depth_end[d] = n_nodes;

    unsigned leaf_counter = 0;
    for (uint32_t i = 0; i < n_nodes; ++i)
    {
        const unsigned d = nodes[i].depth;
        if (depth_start && i < depth_start[d])
            depth_start[d] = i;
        if (depth_end && i + 1 > depth_end[d])
            depth_end[d] = i + 1;

        if (nodes[i].kind != OCTREE_NODE_INTERNAL)
        {
            nodes[i].particle_begin = cursor;
            cursor += nodes[i].particle_count;
            nodes[i].particle_count = 0;
            nodes[i].leaf_id = (int32_t)leaf_counter;
            leaf_counter += 1;
            if (nodes[i].kind == OCTREE_NODE_MULTIPOLE)
                n_mp_leaves += 1;
        }
    }
    return n_mp_leaves;
}

/* ================================================================ */
/* Fill particle order                                              */
/* ================================================================ */

void octree_fill_particle_order(unsigned n_sources, const unsigned *source_leaf_real, octree_node_t *nodes,
                                unsigned *particle_order, unsigned n_threads)
{
#pragma omp parallel for default(none) shared(n_sources, source_leaf_real, nodes, particle_order) schedule(static)     \
    num_threads(n_threads)
    for (unsigned i = 0; i < n_sources; ++i)
    {
        octree_node_t *leaf = &nodes[source_leaf_real[i]];
        unsigned cnt;
#pragma omp atomic capture
        {
            cnt = leaf->particle_count;
            leaf->particle_count += 1;
        }
        particle_order[leaf->particle_begin + cnt] = i;
    }
}

/* ================================================================ */
/* Leaf centroids                                                   */
/* ================================================================ */

void octree_compute_leaf_centers(unsigned n_nodes, octree_node_t *nodes, const unsigned particle_order[restrict],
                                 const real3_t sources_coords[restrict], const real3_t sources_values[restrict],
                                 unsigned n_threads)
{
#pragma omp parallel for default(none) shared(n_nodes, nodes, particle_order, sources_coords, sources_values)          \
    schedule(static) num_threads(n_threads)
    for (uint32_t i = 0; i < n_nodes; ++i)
    {
        if (nodes[i].kind == OCTREE_NODE_INTERNAL)
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
            nodes[i].center = (real3_t){.x = cx / total_weight, .y = cy / total_weight, .z = cz / total_weight};
        if (nodes[i].kind == OCTREE_NODE_MULTIPOLE)
            nodes[i].data.mp.center = nodes[i].center;
    }
}

/* ================================================================ */
/* Build leaf multipoles                                            */
/* ================================================================ */

bool octree_build_leaf_multipoles(unsigned n_nodes, octree_node_t *nodes, const unsigned *restrict particle_order,
                                  const real3_t sources_coords[restrict], const real3_t sources_values[restrict],
                                  const octree_settings_t *settings, octree_scratch_t scratch, unsigned n_sources,
                                  unsigned n_threads, real_t *mp_slices[restrict n_nodes], unsigned n_mp_leaves,
                                  const allocator_t *allocator)
{
    bool leaf_ok = true;

    const allocator_t *a = allocator ? allocator : NULL;
    uint32_t *mp_leaf_indices = (uint32_t *)(a ? a->allocate(a->state, (size_t)n_mp_leaves * sizeof(uint32_t))
                                               : malloc((size_t)n_mp_leaves * sizeof(uint32_t)));
    if (!mp_leaf_indices)
        return false;

    {
        unsigned k = 0;
        for (uint32_t i = 0; i < n_nodes; ++i)
            if (nodes[i].kind == OCTREE_NODE_MULTIPOLE)
                mp_leaf_indices[k++] = i;
    }

    const size_t leaf_buf_size = octree_leaf_stride(settings->order);

#pragma omp parallel default(none)                                                                                     \
    shared(n_mp_leaves, mp_leaf_indices, nodes, particle_order, sources_coords, sources_values, settings, scratch,     \
               leaf_buf_size, n_sources, leaf_ok, mp_slices) num_threads(n_threads)
    {
        const int tid = omp_get_thread_num();
        const size_t tid_s = (size_t)tid;
        const size_t coords_pt = (size_t)n_sources * 3u;
        real_t *leaf_cur = scratch.leaf_cur + tid_s * leaf_buf_size;
        real_t *leaf_nxt = scratch.leaf_nxt + tid_s * leaf_buf_size;
        real_t *leaf_coords = scratch.leaf_coords + tid_s * coords_pt;
        real_t *leaf_values = scratch.leaf_values + tid_s * coords_pt;

#pragma omp for reduction(&& : leaf_ok) schedule(static)
        for (unsigned ml = 0; ml < n_mp_leaves; ++ml)
        {
            const uint32_t i = mp_leaf_indices[ml];
            const size_t n_p = (size_t)nodes[i].particle_count;
            const unsigned begin = nodes[i].particle_begin;

#pragma omp simd
            for (unsigned kk = 0; kk < n_p; ++kk)
            {
                const unsigned src = particle_order[begin + kk];
                leaf_coords[3u * kk + 0] = sources_coords[src].x;
                leaf_coords[3u * kk + 1] = sources_coords[src].y;
                leaf_coords[3u * kk + 2] = sources_coords[src].z;
                leaf_values[3u * kk + 0] = sources_values[src].x;
                leaf_values[3u * kk + 1] = sources_values[src].y;
                leaf_values[3u * kk + 2] = sources_values[src].z;
            }

            const bool mp_ok = multipole_create(settings->order, (unsigned)leaf_buf_size, mp_slices[i], nodes[i].center,
                                                (unsigned)n_p, (const real3_t *)leaf_coords,
                                                (const real3_t *)leaf_values, leaf_cur, leaf_nxt, &nodes[i].data.mp);

            if (nodes[i].kind != OCTREE_NODE_MULTIPOLE)
                leaf_ok = false;
            if (!mp_ok)
                leaf_ok = false;
        }
    }

    if (a)
        a->deallocate(a->state, mp_leaf_indices);
    else
        free(mp_leaf_indices);
    return leaf_ok;
}

/* ================================================================ */
/* Depth ranges                                                     */
/* ================================================================ */

void octree_compute_depth_ranges(unsigned n_nodes, const octree_node_t nodes[restrict n_nodes], unsigned max_depth,
                                 unsigned depth_start[restrict max_depth + 2],
                                 unsigned depth_end[restrict max_depth + 2])
{
    for (unsigned d = 0; d <= max_depth + 1; ++d)
        depth_start[d] = depth_end[d] = n_nodes;
    for (unsigned i = 0; i < n_nodes; ++i)
    {
        const unsigned d = nodes[i].depth;
        if (i < depth_start[d])
            depth_start[d] = i;
        depth_end[d] = i + 1;
    }
}

/* ================================================================ */
/* Upward sweep level                                               */
/* ================================================================ */

void octree_upward_sweep_level(unsigned depth_start, unsigned depth_end, octree_node_t nodes[restrict], unsigned order,
                               size_t n_coeffs, real_t *mp_slices[restrict], unsigned work_order,
                               real_t shift_exp[restrict], real_t pse[restrict], size_t shift_stride, size_t pse_stride,
                               const unsigned particle_order[restrict], const real3_t sources_coords[restrict],
                               const real3_t sources_values[restrict], const octree_scratch_t *scratch,
                               size_t leaf_stride, unsigned n_threads)
{
#pragma omp parallel for default(none)                                                                                 \
    shared(depth_start, depth_end, nodes, order, n_coeffs, mp_slices, work_order, shift_exp, pse, shift_stride,        \
               pse_stride, particle_order, sources_coords, sources_values, scratch, leaf_stride) schedule(dynamic, 16) \
    num_threads(n_threads)
    for (uint32_t i = depth_start; i < depth_end; ++i)
    {
        if (nodes[i].kind != OCTREE_NODE_INTERNAL)
            continue;

        /* ---- |Γ|-weighted centroid from children ---- */
        {
            real_t cx = 0, cy = 0, cz = 0;
            real_t total_weight = 0.0;
            unsigned n_children = 0;

            for (int oct = 0; oct < 8; ++oct)
            {
                octree_node_t *child = nodes[i].data.internal.children[oct];
                if (child == NULL)
                    continue;
                n_children += 1;
                real_t w;
                if (child->kind == OCTREE_NODE_PARTICLE)
                {
                    w = 0.0;
                    for (unsigned kk = child->particle_begin; kk < child->particle_begin + child->particle_count; ++kk)
                        w += real3_mag(sources_values[particle_order[kk]]);
                }
                else
                {
                    real_t *cs = mp_slices[(uint32_t)(child - nodes)];
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
                nodes[i].center = (real3_t){.x = cx / total_weight, .y = cy / total_weight, .z = cz / total_weight};
            }
            else if (n_children > 0)
            {
                real_t ax = 0, ay = 0, az = 0;
                for (int oct = 0; oct < 8; ++oct)
                {
                    octree_node_t *child = nodes[i].data.internal.children[oct];
                    if (child == NULL)
                        continue;
                    ax += child->center.x;
                    ay += child->center.y;
                    az += child->center.z;
                }
                const real_t inv_n = 1.0 / (real_t)n_children;
                nodes[i].center = (real3_t){.x = ax * inv_n, .y = ay * inv_n, .z = az * inv_n};
            }
        }

        /* Zero the internal node's coefficient slice. */
        real_t *slice = mp_slices[i];
        memset(slice, 0, 3u * n_coeffs * sizeof(real_t));
        const multipole_t internal_mp = {
            .order = order,
            .center = nodes[i].center,
            .coeffs_x = slice,
            .coeffs_y = slice + n_coeffs,
            .coeffs_z = slice + 2u * n_coeffs,
        };

        const int tid = omp_get_thread_num();
        const size_t tid_s = (size_t)tid;
        real_t *particle_cur = scratch->leaf_cur + tid_s * leaf_stride;
        real_t *particle_nxt = scratch->leaf_nxt + tid_s * leaf_stride;
        real_t *my_shift_exp = shift_exp + tid_s * shift_stride;
        real_t *my_pse = pse + tid_s * pse_stride;

        for (int oct = 0; oct < 8; ++oct)
        {
            octree_node_t *child = nodes[i].data.internal.children[oct];
            if (child == NULL)
                continue;

            if (child->kind == OCTREE_NODE_PARTICLE)
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
                const uint32_t ci = (uint32_t)(child - nodes);
                real_t *child_slice = mp_slices[ci];
                if (child_slice == NULL)
                    continue;
                const multipole_t child_mp = {
                    .order = order,
                    .center = child->center,
                    .coeffs_x = child_slice,
                    .coeffs_y = child_slice + n_coeffs,
                    .coeffs_z = child_slice + 2u * n_coeffs,
                };
                multipole_add_shift(&child_mp, &internal_mp, work_order, my_shift_exp, my_pse);
            }
        }
    }
}

void octree_run_upward_sweep(unsigned n_nodes, octree_node_t nodes[restrict], unsigned max_depth,
                             const octree_settings_t settings[restrict], real_t **restrict mp_slices,
                             const octree_scratch_t *scratch, const unsigned particle_order[restrict],
                             const real3_t sources_coords[restrict], const real3_t sources_values[restrict],
                             unsigned n_threads)
{
    const unsigned order = settings->order;
    const size_t n_coeffs = multipole_num_coeffs(order);
    const unsigned work_order = octree_resolve_work_order(settings);
    const size_t leaf_stride = octree_leaf_stride(order);
    const size_t shift_stride = octree_shift_stride(work_order);
    const size_t pse_stride = octree_pse_stride(work_order);

    assert(max_depth <= 255 && "octree_run_upward_sweep: max_depth exceeds hardcoded depth_start/depth_end[256]");
    unsigned depth_start[256], depth_end[256];
    octree_compute_depth_ranges(n_nodes, nodes, max_depth, depth_start, depth_end);

    for (unsigned d = max_depth;; --d)
    {
        octree_upward_sweep_level(depth_start[d], depth_end[d], nodes, order, n_coeffs, mp_slices, work_order,
                                  scratch->shift_exp, scratch->pse, shift_stride, pse_stride, particle_order,
                                  sources_coords, sources_values, scratch, leaf_stride, n_threads);
        if (d == 0)
            break;
    }
}
