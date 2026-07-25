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

/* ------------------------------------------------------------------ */
/* FMM-specific sizing (extra regions beyond octree_base_work_sizes_t) */
/* ------------------------------------------------------------------ */

fmm_work_sizes_t fmm_size_work_buffer(unsigned n_sources, const fmm_settings_t settings[restrict],
                                      fmm_count_res_t count_pass_res)
{
    const unsigned n_total =
        count_pass_res.n_internal + count_pass_res.n_multipole_leaves + count_pass_res.n_particle_leaves;
    const octree_base_work_sizes_t base =
        octree_size_work_buffer(n_sources, (const octree_settings_t *)settings, count_pass_res);
    const unsigned n_leaves = count_pass_res.n_multipole_leaves + count_pass_res.n_particle_leaves;
    const size_t max_per_list = n_leaves > 0 ? (size_t)n_leaves * (size_t)(n_leaves - 1) : 1u;
    return (fmm_work_sizes_t){
        .nodes_bytes = base.nodes_bytes,
        .particle_order_bytes = base.particle_order_bytes,
        .multipole_coeffs_bytes = base.multipole_coeffs_bytes,
        .topo_to_real_bytes = base.topo_to_real_bytes,
        .mp_slices_bytes = base.mp_slices_bytes,
        .leaf_indices_bytes = (size_t)n_leaves * sizeof(unsigned),
        .local_coeffs_bytes = (size_t)(count_pass_res.n_multipole_leaves + count_pass_res.n_particle_leaves) * 3u *
                              multipole_num_coeffs(settings->order) * sizeof(real_t),
        .local_slices_bytes = (size_t)n_total * sizeof(real_t *),
        .interaction_lists_bytes = ((size_t)n_leaves + 1) * sizeof(unsigned) * 4 + max_per_list * sizeof(unsigned) * 2,
    };
}

size_t fmm_total_work_size(fmm_work_sizes_t sizes)
{
    return sizes.nodes_bytes + sizes.particle_order_bytes + sizes.multipole_coeffs_bytes + sizes.topo_to_real_bytes +
           sizes.mp_slices_bytes + sizes.leaf_indices_bytes + sizes.local_coeffs_bytes + sizes.local_slices_bytes +
           sizes.interaction_lists_bytes;
}

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
static void fmm_compute_interaction_lists(unsigned n_nodes, const octree_node_t CVL_ARRAY_ARG(nodes, restrict),
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
        const octree_node_t *node_a = &nodes[idx_a];
        vlist_offsets[li] = (unsigned)vlist_total;
        nflist_offsets[li] = (unsigned)nflist_total;

        for (unsigned lj = 0; lj < n_leaves; ++lj)
        {
            if (li == lj)
                continue;
            const unsigned idx_b = leaf_indices[lj];
            const octree_node_t *node_b = &nodes[idx_b];

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
static unsigned fmm_build_leaf_index_map(unsigned n_nodes, const octree_node_t CVL_ARRAY_ARG(nodes, restrict n_nodes),
                                         unsigned CVL_ARRAY_ARG(leaf_indices, restrict))
{
    unsigned cnt = 0;
    for (uint32_t i = 0; i < n_nodes; ++i)
    {
        if (nodes[i].kind != OCTREE_NODE_INTERNAL)
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
static void fmm_assign_local_slices(unsigned n_nodes, const octree_node_t CVL_ARRAY_ARG(nodes, restrict n_nodes),
                                    unsigned order, real_t *local_coeffs,
                                    real_t *CVL_ARRAY_ARG(local_slices, restrict n_nodes))
{
    const size_t n_coeffs = multipole_num_coeffs(order);
    real_t *cursor = local_coeffs;

    for (uint32_t i = 0; i < n_nodes; ++i)
    {
        if (nodes[i].kind != OCTREE_NODE_INTERNAL)
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
static void fmm_m2l_sweep(unsigned n_nodes, const octree_node_t CVL_ARRAY_ARG(nodes, restrict n_nodes),
                          unsigned n_leaves, const unsigned CVL_ARRAY_ARG(leaf_indices, restrict n_leaves),
                          unsigned order, unsigned work_order, real_t *const *restrict mp_slices,
                          real_t *const *restrict local_slices,
                          unsigned CVL_ARRAY_ARG(vlist_offsets, restrict n_leaves + 1),
                          unsigned CVL_ARRAY_ARG(vlist_indices, restrict), const octree_scratch_t *scratch,
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
static void fmm_downward_l2l_sweep(unsigned n_nodes, octree_node_t CVL_ARRAY_ARG(nodes, restrict n_nodes),
                                   unsigned order, unsigned work_order, real_t *const *restrict local_slices,
                                   const octree_scratch_t *scratch, unsigned n_threads)
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
        if (nodes[i].kind != OCTREE_NODE_INTERNAL)
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
            octree_node_t *child = nodes[i].data.internal.children[k];
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
        while (tree->nodes[idx].kind == OCTREE_NODE_INTERNAL)
        {
            const real3_t p = point;
            const real3_t c = tree->nodes[idx].center;
            const unsigned oct =
                (unsigned)(p.x >= c.x) * 1u + (unsigned)(p.y >= c.y) * 2u + (unsigned)(p.z >= c.z) * 4u;
            const octree_node_t *child = tree->nodes[idx].data.internal.children[oct];
            if (child == NULL)
                break;
            idx = (uint32_t)(child - tree->nodes);
        }
        leaf_node_idx = idx;
    }

    const octree_node_t *target_leaf = &tree->nodes[leaf_node_idx];
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
                const octree_node_t *other = &tree->nodes[other_node_idx];
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
        if (target_leaf->kind != OCTREE_NODE_INTERNAL)
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
            const octree_node_t *other = &tree->nodes[other_node_idx];
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
/* FMM work-buffer view (local type, not exported)                    */
/* ------------------------------------------------------------------ */

typedef struct
{
    octree_node_t *nodes;
    unsigned *particle_order;
    real_t *multipole_coeffs;
    uint32_t *topo_to_real;
    real_t **mp_slices;
    unsigned *leaf_indices;
    real_t *local_coeffs;
    real_t **local_slices;
    unsigned *vlist_offsets;
    unsigned *vlist_indices;
    size_t vlist_capacity;
    unsigned *nflist_offsets;
    unsigned *nflist_indices;
    size_t nflist_capacity;
} fmm_work_t;

/* ------------------------------------------------------------------ */
/* Work buffer partition helper (FMM-specific)                        */
/* ------------------------------------------------------------------ */

static fmm_work_t fmm_partition_work(fmm_work_sizes_t sizes, void *buffer, unsigned n_leaves)
{
    uint8_t *bp = (uint8_t *)buffer;
    fmm_work_t out = {0};
    out.nodes = (octree_node_t *)bp;
    bp += sizes.nodes_bytes;
    out.particle_order = (unsigned *)bp;
    bp += sizes.particle_order_bytes;
    out.multipole_coeffs = (real_t *)bp;
    bp += sizes.multipole_coeffs_bytes;
    out.topo_to_real = (uint32_t *)bp;
    bp += sizes.topo_to_real_bytes;
    out.mp_slices = (real_t **)bp;
    bp += sizes.mp_slices_bytes;
    out.leaf_indices = (unsigned *)bp;
    bp += sizes.leaf_indices_bytes;
    out.local_coeffs = (real_t *)bp;
    bp += sizes.local_coeffs_bytes;
    out.local_slices = (real_t **)bp;
    bp += sizes.local_slices_bytes;

    const size_t offsets_bytes = (size_t)(n_leaves + 1) * sizeof(unsigned);
    const size_t max_per = n_leaves > 0 ? (size_t)n_leaves * (size_t)(n_leaves - 1) : 1u;
    out.vlist_offsets = (unsigned *)bp;
    bp += offsets_bytes;
    out.vlist_indices = (unsigned *)bp;
    out.vlist_capacity = max_per;
    bp += max_per * sizeof(unsigned);
    out.nflist_offsets = (unsigned *)bp;
    bp += offsets_bytes;
    out.nflist_indices = (unsigned *)bp;
    out.nflist_capacity = max_per;
    return out;
}

/* ------------------------------------------------------------------ */
/* fmm_tree_build (convenience — allocates internally)                */
/* ------------------------------------------------------------------ */

bool fmm_tree_build(unsigned n_sources, unsigned n_threads, const real3_t sources_coords[restrict n_sources],
                    const real3_t sources_values[restrict n_sources], const fmm_settings_t settings[restrict],
                    const allocator_t *allocator, fmm_tree_t *out)
{
    void *scratch_buffer = NULL, *work_buffer = NULL;
    bool ret = false;

    if (!out || n_sources == 0 || settings == NULL || sources_coords == NULL || sources_values == NULL)
        return false;

    /* Scratch. */
    const size_t needed_scratch = octree_scratch_size(n_sources, n_threads, (const octree_settings_t *)settings);
    if (needed_scratch == 0)
        return false;
    scratch_buffer = fmm_alloc(allocator, needed_scratch);
    if (!scratch_buffer)
        return false;

    const octree_scratch_sizes_t scratch_sz = octree_size_scratch(n_sources, (const octree_settings_t *)settings);
    const octree_scratch_t scratch = octree_scratch_partition(n_threads, scratch_buffer, scratch_sz);
    const size_t topo_bytes = (8u * (size_t)n_sources + 1u) * sizeof(topo_node_t);
    memset(scratch.topo, 0, topo_bytes);

    /* Count pass. */
    const octree_count_t count = octree_count_pass(n_sources, sources_coords, (const octree_settings_t *)settings,
                                                   scratch.topo, scratch.source_leaf_topo);

    /* Work buffer. */
    const fmm_work_sizes_t ws = fmm_size_work_buffer(n_sources, settings, count);
    const size_t total_work_size = fmm_total_work_size(ws);
    work_buffer = fmm_alloc(allocator, total_work_size);
    if (!work_buffer)
        goto cleanup;

    /* Build pipeline. */
    if (!fmm_tree_insert(n_sources, n_threads, sources_coords, sources_values, settings, scratch_buffer, needed_scratch,
                         allocator, work_buffer, total_work_size, out))
        goto cleanup;

    out->buffer = (uint8_t *)work_buffer;
    out->buffer_size = total_work_size;
    work_buffer = NULL; /* ownership transferred */
    ret = true;

cleanup:
    fmm_free(allocator, work_buffer);
    fmm_free(allocator, scratch_buffer);
    return ret;
}

/* ------------------------------------------------------------------ */
/* fmm_tree_insert (pre-allocated buffers)                            */
/* ------------------------------------------------------------------ */

bool fmm_tree_insert(unsigned n_sources, unsigned n_threads, const real3_t sources_coords[restrict n_sources],
                     const real3_t sources_values[restrict n_sources], const fmm_settings_t settings[restrict],
                     void *scratch_buffer, size_t scratch_size, const allocator_t *allocator, void *buffer,
                     size_t buffer_size, fmm_tree_t *out)
{
    uint32_t *mp_leaf_indices = NULL;
    bool ret = false;

    if (!buffer || !out || !scratch_buffer)
        return false;
    if (n_sources == 0 || settings == NULL)
        return false;

    /* Partition scratch. */
    const octree_scratch_sizes_t scratch_sz = octree_size_scratch(n_sources, (const octree_settings_t *)settings);
    if (scratch_size < octree_total_scratch_size(scratch_sz, n_threads))
        return false;
    const octree_scratch_t scratch = octree_scratch_partition(n_threads, scratch_buffer, scratch_sz);
    const size_t topo_bytes = (8u * (size_t)n_sources + 1u) * sizeof(topo_node_t);
    memset(scratch.topo, 0, topo_bytes);

    /* Count pass. */
    const octree_count_t count = octree_count_pass(n_sources, sources_coords, (const octree_settings_t *)settings,
                                                   scratch.topo, scratch.source_leaf_topo);

    const unsigned n_topo = count.n_internal + count.n_multipole_leaves + count.n_particle_leaves;
    if (n_topo == 0)
        return false;

    /* Size work buffer and validate. */
    const fmm_work_sizes_t ws = fmm_size_work_buffer(n_sources, settings, count);
    if (buffer_size < fmm_total_work_size(ws))
        return false;

    const unsigned n_leaves = count.n_multipole_leaves + count.n_particle_leaves;
    const fmm_work_t work = fmm_partition_work(ws, buffer, n_leaves);

    /* === Shared pipeline stages === */
    octree_materialize(scratch.topo, n_topo, (const octree_settings_t *)settings, work.topo_to_real, work.nodes,
                       work.multipole_coeffs, work.mp_slices, n_threads);
    octree_descend(n_sources, sources_coords, n_topo, work.nodes, scratch.source_leaf_real, n_threads);

    const unsigned n_mp = octree_compute_metadata(n_topo, work.nodes, count.max_depth, NULL, NULL);

    octree_fill_particle_order(n_sources, scratch.source_leaf_real, work.nodes, work.particle_order, n_threads);
    octree_compute_leaf_centers(n_topo, work.nodes, work.particle_order, sources_coords, sources_values, n_threads);

    if (n_mp > 0)
    {
        if (!octree_build_leaf_multipoles(n_topo, work.nodes, work.particle_order, sources_coords, sources_values,
                                          (const octree_settings_t *)settings, scratch, n_sources, n_threads,
                                          work.mp_slices, n_mp, allocator))
            goto cleanup;
    }

    /* Upward sweep (M2M). */
    {
        const unsigned order = settings->order;
        const size_t n_coeffs = multipole_num_coeffs(order);
        const unsigned work_order = settings->work_order ? settings->work_order : settings->order;
        const size_t leaf_stride = octree_leaf_stride(order);
        const size_t shift_stride = octree_shift_stride(work_order);
        const size_t pse_stride = octree_pse_stride(work_order);

        unsigned depth_start[256], depth_end[256];
        octree_compute_depth_ranges(n_topo, work.nodes, count.max_depth, depth_start, depth_end);

        for (unsigned d = count.max_depth;; --d)
        {
            octree_upward_sweep_level(depth_start[d], depth_end[d], work.nodes, order, n_coeffs, work.mp_slices,
                                      work_order, scratch.shift_exp, scratch.pse, shift_stride, pse_stride,
                                      work.particle_order, sources_coords, sources_values, &scratch, leaf_stride,
                                      n_threads);
            if (d == 0)
                break;
        }
    }

    /* === FMM-specific stages === */

    /* Build interaction lists. */
    {
        fmm_build_leaf_index_map(n_topo, work.nodes, work.leaf_indices);

        size_t vlist_count = 0, nflist_count = 0;
        fmm_compute_interaction_lists(n_topo, work.nodes, n_leaves, work.leaf_indices, work.vlist_offsets,
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

    /* M2L + L2L. */
    {
        const unsigned order = settings->order;
        const unsigned work_order = settings->work_order ? settings->work_order : settings->order;

        fmm_assign_local_slices(n_topo, work.nodes, order, work.local_coeffs, work.local_slices);
        fmm_m2l_sweep(n_topo, work.nodes, n_leaves, work.leaf_indices, order, work_order, work.mp_slices,
                      work.local_slices, work.vlist_offsets, work.vlist_indices, &scratch, n_threads);
        fmm_downward_l2l_sweep(n_topo, work.nodes, order, work_order, work.local_slices, &scratch, n_threads);

        out->local_coeffs = work.local_coeffs;
        out->local_slices = work.local_slices;
    }

    /* Populate tree handle. */
    out->settings = *settings;
    out->root_center = work.nodes[0].center;
    out->root_half_size = work.nodes[0].half_size;
    out->n_sources = n_sources;
    out->n_nodes = n_topo;
    out->n_internal = count.n_internal;
    out->n_multipole_leaves = n_mp;
    out->n_particle_leaves = count.n_particle_leaves;
    out->max_depth_reached = count.max_depth;
    out->buffer = (uint8_t *)buffer;
    out->buffer_size = buffer_size;
    out->nodes = work.nodes;
    out->particle_order = work.particle_order;
    out->multipole_coeffs = work.multipole_coeffs;
    out->mp_slices = work.mp_slices;

    ret = true;

cleanup:
    fmm_free(allocator, mp_leaf_indices);
    return ret;
}

unsigned fmm_tree_n_nodes(const fmm_tree_t *tree)
{
    return tree ? tree->n_nodes : 0;
}

size_t fmm_tree_memory_bytes(const fmm_tree_t *tree)
{
    return tree ? tree->buffer_size : 0;
}
