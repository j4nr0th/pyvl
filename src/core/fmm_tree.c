#include "fmm_tree.h"
#include "fmm_operators.h"

#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* Internal constants                                                 */
/* ------------------------------------------------------------------ */

/** @brief Depth array sizing.  topo_node_t.depth is uint8_t \xE2\x86\x92 max depth 255. */
#define FMM_DEPTH_SLOTS 256u

/** @brief Stack for dual-tree walk in V-list building. */
#define FMM_VLIST_STACK_SIZE 256u

/** @brief Small epsilon for geometry comparisons. */
#define FMM_EPS ((real_t)1e-12)

/* ------------------------------------------------------------------ */
/* Internal helpers                                                   */
/* ------------------------------------------------------------------ */

/* Morton 3D helpers moved to octree.h (morton_split_21, morton_3d, morton_range_bounds)
 * and octree.c (octree_build_morton_sorted). */

/* ------------------------------------------------------------------ */
/* FMM-specific sizing (extra regions beyond octree_base_work_sizes_t) */
/* ------------------------------------------------------------------ */

fmm_work_sizes_t fmm_size_work_buffer(unsigned n_sources, const fmm_settings_t settings[restrict],
                                      fmm_count_res_t count_pass_res, unsigned n_threads)
{
    const unsigned n_total =
        count_pass_res.n_internal + count_pass_res.n_multipole_leaves + count_pass_res.n_particle_leaves;
    const octree_base_work_sizes_t base =
        octree_size_work_buffer(n_sources, (const octree_settings_t *)settings, count_pass_res);
    const unsigned n_leaves = count_pass_res.n_multipole_leaves + count_pass_res.n_particle_leaves;
    const size_t max_per_list = n_leaves > 0 ? (size_t)n_leaves * (size_t)(n_leaves - 1) : 1u;
    const size_t max_per_node = (size_t)n_total * (size_t)n_total;
    const unsigned work_order = settings->work_order ? settings->work_order : settings->order;
    const size_t shift_exp_per_thread = 3u * (size_t)(work_order + 1) * (size_t)(work_order + 1) * sizeof(real_t);
    const size_t pse_per_thread = 2u * multipole_num_coeffs(work_order) * sizeof(real_t);
    const size_t morton_bytes_per_node = sizeof(uint64_t) + sizeof(unsigned);
    const size_t pair_size = sizeof(uint64_t) + sizeof(unsigned);
    const size_t depth_slots = (size_t)count_pass_res.max_depth + 2u;

    return (fmm_work_sizes_t){
        .nodes_bytes = base.nodes_bytes,
        .particle_order_bytes = base.particle_order_bytes,
        .multipole_coeffs_bytes = base.multipole_coeffs_bytes,
        .topo_to_real_bytes = base.topo_to_real_bytes,
        .mp_slices_bytes = base.mp_slices_bytes,
        .leaf_indices_bytes = (size_t)n_leaves * sizeof(unsigned),
        /* Local expansions for ALL nodes (internal + leaves) for multi-level FMM. */
        .local_coeffs_bytes = (size_t)n_total * 3u * multipole_num_coeffs(settings->order) * sizeof(real_t),
        .local_slices_bytes = (size_t)n_total * sizeof(real_t *),
        /* Per-leaf V-list (tree-code mode) + per-node interaction lists (multi-level M2L). */
        .interaction_lists_bytes = ((size_t)n_leaves + 1) * sizeof(unsigned) * 4 + max_per_list * sizeof(unsigned) * 2 +
                                   ((size_t)n_total + 1) * sizeof(unsigned) + max_per_node * sizeof(unsigned),
        .m2l_shift_exp_bytes = (size_t)n_threads * shift_exp_per_thread,
        .m2l_pse_bytes = (size_t)n_threads * pse_per_thread,
        .morton_sort_bytes =
            (size_t)n_total * morton_bytes_per_node + (size_t)n_total * pair_size * 2u + depth_slots * sizeof(unsigned),
    };
}

size_t fmm_total_work_size(fmm_work_sizes_t sizes)
{
    return sizes.nodes_bytes + sizes.particle_order_bytes + sizes.multipole_coeffs_bytes + sizes.topo_to_real_bytes +
           sizes.mp_slices_bytes + sizes.leaf_indices_bytes + sizes.local_coeffs_bytes + sizes.local_slices_bytes +
           sizes.interaction_lists_bytes + sizes.m2l_shift_exp_bytes + sizes.m2l_pse_bytes + sizes.morton_sort_bytes;
}

/**
 * @brief Build per-leaf interaction lists for tree-code mode.
 *
 * For each leaf, walk the tree from the root and for each candidate node:
 *  - Well-separated: add the node's INDEX to the V-list.
 *  - Not well-separated AND internal: descend.
 *  - Not well-separated AND leaf: near-field.
 *
 * @param leaf_indices    Mapping from leaf index (0..n_leaves-1) to node index.
 * @param vlist_offsets   Output CSR offsets [n_leaves+1].
 * @param vlist_indices   Output flat V-list (NODE indices).
 * @param vlist_capacity  Capacity of vlist_indices.
 * @param nflist_offsets  Output CSR offsets [n_leaves+1].
 * @param nflist_indices  Output flat near-field leaf indices.
 * @param nflist_capacity Capacity of nflist_indices.
 * @param out_vlist_count  Output total V-list entries.
 * @param out_nflist_count Output total near-field entries.
 * @param theta            MAC opening-angle (0 = neighbour criterion).
 */
static bool fmm_build_leaf_vlists(const octree_node_t CVL_ARRAY_ARG(nodes, restrict), unsigned n_leaves,
                                  const unsigned CVL_ARRAY_ARG(leaf_indices, restrict n_leaves),
                                  unsigned CVL_ARRAY_ARG(vlist_offsets, restrict n_leaves + 1),
                                  unsigned CVL_ARRAY_ARG(vlist_indices, restrict), size_t vlist_capacity,
                                  unsigned CVL_ARRAY_ARG(nflist_offsets, restrict n_leaves + 1),
                                  unsigned CVL_ARRAY_ARG(nflist_indices, restrict), size_t nflist_capacity,
                                  size_t *out_vlist_count, size_t *out_nflist_count, unsigned n_threads, double theta)
{
    /* Depth never exceeds FMM_DEPTH_SLOTS; use a named constant for the local stack. */

    if (n_leaves == 0)
    {
        vlist_offsets[0] = nflist_offsets[0] = 0;
        *out_vlist_count = *out_nflist_count = 0;
        return true;
    }

    /*
     * Two-pass CSR construction:
     *   Pass 1 (parallel) — dual-tree walk to count per-leaf entries.
     *   Pass 2 (serial)   — prefix-sum to build CSR offsets.
     *   Pass 3 (parallel) — repeat dual-tree walk to fill indices.
     */

    /* Pass 1: count (parallel). */
#pragma omp parallel for default(none) shared(n_leaves, leaf_indices, nodes, vlist_offsets, nflist_offsets, theta)     \
    schedule(dynamic) num_threads(n_threads)
    for (unsigned li = 0; li < n_leaves; ++li)
    {
        const unsigned target_ni = leaf_indices[li];
        const octree_node_t *target = &nodes[target_ni];
        const real_t ht = target->half_size;
        unsigned vcnt = 0, nfcnt = 0;

        /* Dual-tree walk: depth-first, explicit stack. */
        unsigned stack[FMM_VLIST_STACK_SIZE];
        unsigned sp = 0;
        stack[sp++] = 0; /* root */

        while (sp > 0)
        {
            const unsigned ni = stack[--sp];
            if (ni == target_ni)
                continue;

            const octree_node_t *cand = &nodes[ni];
            const real3_t diff = real3_sub(target->center, cand->center);
            const real_t dist = real3_mag(diff);
            bool well_separated;
            if (theta > 0.0)
                well_separated = dist > (real_t)1e-30 && cand->half_size / dist < theta;
            else
                well_separated = dist >= 3.0 * (ht > cand->half_size ? ht : cand->half_size) - (real_t)1e-12;

            if (well_separated)
            {
                /* Well-separated: add to V-list (any node type). */
                vcnt++;
            }
            else if (cand->kind == OCTREE_NODE_INTERNAL)
            {
                /* Too close for the coarser cell — descend. */
                for (int oct = 7; oct >= 0; --oct)
                {
                    const octree_node_t *child = cand->data.internal.children[oct];
                    if (child != NULL)
                        stack[sp++] = (unsigned)(child - nodes);
                }
            }
            else
            {
                /* Leaf and not well-separated: near-field. */
                nfcnt++;
            }
        }
        vlist_offsets[li] = vcnt;
        nflist_offsets[li] = nfcnt;
    }

    /* Pass 2: prefix-sum (serial). */
    size_t vlist_total = 0, nflist_total = 0;
    for (unsigned li = 0; li < n_leaves; ++li)
    {
        const unsigned vcnt = vlist_offsets[li];
        const unsigned nfcnt = nflist_offsets[li];
        vlist_offsets[li] = (unsigned)vlist_total;
        nflist_offsets[li] = (unsigned)nflist_total;
        vlist_total += vcnt;
        nflist_total += nfcnt;
    }
    vlist_offsets[n_leaves] = (unsigned)vlist_total;
    nflist_offsets[n_leaves] = (unsigned)nflist_total;

    if (vlist_total > vlist_capacity || nflist_total > nflist_capacity)
        return false;

    /* Pass 3: fill index arrays (parallel). */
#pragma omp parallel for default(none)                                                                                 \
    shared(n_leaves, leaf_indices, nodes, vlist_offsets, vlist_indices, nflist_offsets, nflist_indices, theta)         \
    schedule(dynamic) num_threads(n_threads)
    for (unsigned li = 0; li < n_leaves; ++li)
    {
        unsigned vi = vlist_offsets[li];
        unsigned nfi = nflist_offsets[li];
        const unsigned target_ni = leaf_indices[li];
        const octree_node_t *target = &nodes[target_ni];
        const real_t ht = target->half_size;

        unsigned stack[FMM_VLIST_STACK_SIZE];
        unsigned sp = 0;
        stack[sp++] = 0;

        while (sp > 0)
        {
            const unsigned ni = stack[--sp];
            if (ni == target_ni)
                continue;

            const octree_node_t *cand = &nodes[ni];
            const real3_t diff = real3_sub(target->center, cand->center);
            const real_t dist = real3_mag(diff);
            bool well_separated;
            if (theta > 0.0)
                well_separated = dist > (real_t)1e-30 && cand->half_size / dist < theta;
            else
                well_separated = dist >= 3.0 * (ht > cand->half_size ? ht : cand->half_size) - (real_t)1e-12;

            if (well_separated)
            {
                vlist_indices[vi++] = ni;
            }
            else if (cand->kind == OCTREE_NODE_INTERNAL)
            {
                for (int oct = 7; oct >= 0; --oct)
                {
                    const octree_node_t *child = cand->data.internal.children[oct];
                    if (child != NULL)
                        stack[sp++] = (unsigned)(child - nodes);
                }
            }
            else
            {
                nflist_indices[nfi++] = (unsigned)nodes[ni].leaf_id;
            }
        }
    }

    *out_vlist_count = vlist_total;
    *out_nflist_count = nflist_total;
    return true;
}

/**
 * @brief Build per-node interaction lists for multi-level M2L using
 *        Morton-code-accelerated range queries.
 *
 * For each node, finds same-depth nodes in the interaction zone [3h, 6h).
 * Uses pre-computed Morton codes and per-depth sorted indices for O(N log N)
 * range queries instead of the O(N^2) dual-tree walk.
 */
static bool fmm_build_per_node_interaction_lists(const octree_node_t *nodes, unsigned n_nodes, unsigned *mlvl_offsets,
                                                 unsigned *mlvl_indices, size_t mlvl_capacity, size_t *out_count,
                                                 unsigned n_threads, const uint64_t *morton_codes,
                                                 const unsigned *morton_sorted, const unsigned *depth_offsets,
                                                 unsigned max_depth_found)
{
    if (n_nodes == 0)
    {
        mlvl_offsets[0] = 0;
        *out_count = 0;
        return true;
    }

    /* Reference frame for Morton: invariant geometric root centre. */
    const real3_t root_gc = nodes[0].geom_center;
    const real_t root_hs = nodes[0].half_size;

    /* Pass 1: count (parallel).  Morton bounding-box range query per node. */
#pragma omp parallel for default(none) shared(n_nodes, nodes, mlvl_offsets, morton_codes, morton_sorted,               \
                                                  depth_offsets, max_depth_found, root_gc, root_hs) schedule(dynamic)  \
    num_threads(n_threads)
    for (unsigned ni = 0; ni < n_nodes; ++ni)
    {
        const octree_node_t *target = &nodes[ni];
        const real_t h = target->half_size;
        const unsigned td = target->depth;
        unsigned cnt = 0;

        /* Skip if this node's depth exceeds our depth_offsets bounds. */
        if (td > max_depth_found)
        {
            mlvl_offsets[ni] = 0;
            continue;
        }

        /* Bounding box for the [3h, 6h) interaction zone. */
        real3_t bb_min, bb_max;
        bb_min.x = target->center.x - 6.0 * h;
        bb_min.y = target->center.y - 6.0 * h;
        bb_min.z = target->center.z - 6.0 * h;
        bb_max.x = target->center.x + 6.0 * h;
        bb_max.y = target->center.y + 6.0 * h;
        bb_max.z = target->center.z + 6.0 * h;

        const unsigned doff = depth_offsets[td];
        unsigned dend = n_nodes;
        /* Find end offset: the start of the next occupied depth level. */
        for (unsigned dd = td + 1; dd <= max_depth_found; ++dd)
        {
            const unsigned nxt = depth_offsets[dd];
            if (nxt < dend)
            {
                dend = nxt;
                break;
            }
        }
        if (doff >= dend)
        {
            mlvl_offsets[ni] = 0;
            continue;
        }
        const size_t ndepth = (size_t)(dend - doff);
        assert(ndepth <= n_nodes && "depth range count exceeds n_nodes");

        /* Morton range over all 8 bounding-box corners. */
        uint64_t mc_min = UINT64_MAX, mc_max = 0;
        for (int mc_ix = 0; mc_ix < 8; ++mc_ix)
        {
            real3_t corner;
            corner.x = (mc_ix & 1) ? bb_max.x : bb_min.x;
            corner.y = (mc_ix & 2) ? bb_max.y : bb_min.y;
            corner.z = (mc_ix & 4) ? bb_max.z : bb_min.z;
            const uint64_t mc = morton_3d(corner, root_gc, root_hs);
            if (mc < mc_min)
                mc_min = mc;
            if (mc > mc_max)
                mc_max = mc;
        }

        size_t b_begin, b_end;
        morton_range_bounds(morton_sorted, morton_codes, doff, ndepth, mc_min, mc_max, &b_begin, &b_end);

        for (size_t j = b_begin; j < b_end; ++j)
        {
            const unsigned ci = morton_sorted[j];
            assert(ci < n_nodes && "Morton-sorted ci out of range");
            if (ci == ni)
                continue;
            const real_t dist = real3_mag(real3_sub(target->center, nodes[ci].center));
            if (dist >= 3.0 * h - FMM_EPS && dist < 6.0 * h - FMM_EPS)
                cnt++;
        }
        mlvl_offsets[ni] = cnt;
    }

    /* Pass 2: prefix-sum (serial). */
    size_t total = 0;
    for (unsigned ni = 0; ni < n_nodes; ++ni)
    {
        const unsigned cnt = mlvl_offsets[ni];
        mlvl_offsets[ni] = (unsigned)total;
        total += cnt;
    }
    mlvl_offsets[n_nodes] = (unsigned)total;

    if (total > mlvl_capacity)
        return false;

    /* Pass 3: fill (parallel). */
#pragma omp parallel for default(none) shared(n_nodes, nodes, mlvl_offsets, mlvl_indices, morton_codes, morton_sorted, \
                                                  depth_offsets, max_depth_found, root_gc, root_hs) schedule(dynamic)  \
    num_threads(n_threads)
    for (unsigned ni = 0; ni < n_nodes; ++ni)
    {
        unsigned vi = mlvl_offsets[ni];
        const octree_node_t *target = &nodes[ni];
        const real_t h = target->half_size;
        const unsigned td = target->depth;

        if (td > max_depth_found)
            continue;

        real3_t bb_min, bb_max;
        bb_min.x = target->center.x - 6.0 * h;
        bb_min.y = target->center.y - 6.0 * h;
        bb_min.z = target->center.z - 6.0 * h;
        bb_max.x = target->center.x + 6.0 * h;
        bb_max.y = target->center.y + 6.0 * h;
        bb_max.z = target->center.z + 6.0 * h;

        const unsigned doff = depth_offsets[td];
        unsigned dend = n_nodes;
        for (unsigned dd = td + 1; dd <= max_depth_found; ++dd)
        {
            const unsigned nxt = depth_offsets[dd];
            if (nxt < dend)
            {
                dend = nxt;
                break;
            }
        }
        if (doff >= dend)
            continue;
        const size_t ndepth = (size_t)(dend - doff);

        uint64_t mc_min = UINT64_MAX, mc_max = 0;
        for (int mc_ix = 0; mc_ix < 8; ++mc_ix)
        {
            real3_t corner;
            corner.x = (mc_ix & 1) ? bb_max.x : bb_min.x;
            corner.y = (mc_ix & 2) ? bb_max.y : bb_min.y;
            corner.z = (mc_ix & 4) ? bb_max.z : bb_min.z;
            const uint64_t mc = morton_3d(corner, root_gc, root_hs);
            if (mc < mc_min)
                mc_min = mc;
            if (mc > mc_max)
                mc_max = mc;
        }

        size_t b_begin, b_end;
        morton_range_bounds(morton_sorted, morton_codes, doff, ndepth, mc_min, mc_max, &b_begin, &b_end);

        for (size_t j = b_begin; j < b_end; ++j)
        {
            const unsigned ci = morton_sorted[j];
            if (ci == ni)
                continue;
            const real_t dist = real3_mag(real3_sub(target->center, nodes[ci].center));
            if (dist >= 3.0 * h - FMM_EPS && dist < 6.0 * h - FMM_EPS)
                mlvl_indices[vi++] = ci;
        }
    }

    *out_count = total;
    return true;
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
#pragma omp parallel for reduction(+ : cnt) default(none) shared(n_nodes, nodes, leaf_indices)
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
static void fmm_assign_local_slices(unsigned n_nodes, unsigned order, real_t *local_coeffs,
                                    real_t *CVL_ARRAY_ARG(local_slices, restrict n_nodes))
{
    const size_t n_coeffs = multipole_num_coeffs(order);
    const size_t slice_stride = 3u * n_coeffs;

#pragma omp parallel for default(none) shared(n_nodes, local_slices, local_coeffs, slice_stride)
    for (uint32_t i = 0; i < n_nodes; ++i)
        local_slices[i] = local_coeffs + i * slice_stride;

    /* Zero all local coefficients. */
    memset(local_coeffs, 0, (size_t)n_nodes * slice_stride * sizeof(real_t));
}

/* ------------------------------------------------------------------ */
/* M2L sweep                                                          */
/* ------------------------------------------------------------------ */

/**
 * @brief Multi-level M2L sweep over ALL nodes using per-node interaction lists.
 *
 * For each node, converts all interaction-list sources (same-depth nodes in
 * the [3h, 6h) zone) into a local expansion at the node's centre.
 */
static void fmm_m2l_sweep_mlvl(unsigned n_nodes, const octree_node_t CVL_ARRAY_ARG(nodes, restrict n_nodes),
                               unsigned order, unsigned work_order, real_t *const *restrict mp_slices,
                               real_t *const *restrict local_slices,
                               unsigned CVL_ARRAY_ARG(mlvl_offsets, restrict n_nodes + 1),
                               unsigned CVL_ARRAY_ARG(mlvl_indices, restrict), real_t *work_shift_exp, real_t *work_pse,
                               unsigned n_threads)
{
    const size_t n_coeffs = multipole_num_coeffs(order);
    const size_t shift_per_thread = 3u * (size_t)(work_order + 1) * (size_t)(work_order + 1);
    const size_t pse_per_thread = 2u * multipole_num_coeffs(work_order);

    unsigned thread_counter_m2l = 0;
#pragma omp parallel default(none)                                                                                     \
    shared(n_nodes, nodes, order, work_order, mp_slices, local_slices, mlvl_offsets, mlvl_indices, work_shift_exp,     \
               work_pse, n_coeffs, shift_per_thread, pse_per_thread, thread_counter_m2l) num_threads(n_threads)
    {
        unsigned tid;
#pragma omp atomic capture
        tid = thread_counter_m2l++;
        real_t *my_shift_exp = work_shift_exp + (size_t)tid * shift_per_thread;
        real_t *my_pse = work_pse + (size_t)tid * pse_per_thread;
#pragma omp for schedule(dynamic, 16)
        for (unsigned ni = 0; ni < n_nodes; ++ni)
        {
            real_t *local_slice = local_slices[ni];
            if (local_slice == NULL)
                continue;

            const local_expansion_t local = {.order = order,
                                             .center = nodes[ni].center,
                                             .coeffs_x = local_slice,
                                             .coeffs_y = local_slice + n_coeffs,
                                             .coeffs_z = local_slice + 2u * n_coeffs};

            const unsigned v_start = mlvl_offsets[ni];
            const unsigned v_end = mlvl_offsets[ni + 1];
            for (unsigned vi = v_start; vi < v_end; ++vi)
            {
                const unsigned ni_src = mlvl_indices[vi];
                real_t *mp_slice = mp_slices[ni_src];
                if (mp_slice == NULL)
                    continue;
                const multipole_t mp = {.order = order,
                                        .center = nodes[ni_src].center,
                                        .coeffs_x = mp_slice,
                                        .coeffs_y = mp_slice + n_coeffs,
                                        .coeffs_z = mp_slice + 2u * n_coeffs};
                multipole_to_local(&mp, (local_expansion_t *)&local, work_order, my_shift_exp, my_pse);
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/* L2L downward sweep                                                 */
/* ------------------------------------------------------------------ */

/**
 * @brief Propagate local expansions downward via L2L (coarse → fine).
 *
 * For each internal node (root → leaves), L2L-shift its local expansion
 * into each child's local expansion (additive).  After this sweep, every
 * node's local expansion contains contributions from all coarser-level
 * ancestor interactions.
 */
static void fmm_downward_l2l_sweep(unsigned n_nodes, const octree_node_t CVL_ARRAY_ARG(nodes, restrict n_nodes),
                                   unsigned order, unsigned work_order, real_t *const *restrict local_slices,
                                   real_t *work_shift_exp, real_t *work_pse, unsigned n_threads)
{
    const size_t n_coeffs = multipole_num_coeffs(order);
    const size_t shift_per_thread = 3u * (size_t)(work_order + 1) * (size_t)(work_order + 1);
    const size_t pse_per_thread = 2u * multipole_num_coeffs(work_order);

    /* Find max depth. */
    unsigned max_depth = 0;
    for (unsigned i = 0; i < n_nodes; ++i)
        if (nodes[i].depth > max_depth)
            max_depth = nodes[i].depth;

    unsigned depth_start[FMM_DEPTH_SLOTS], depth_end[FMM_DEPTH_SLOTS];
    octree_compute_depth_ranges(n_nodes, nodes, max_depth, depth_start, depth_end);

    /*
     * Propagate local expansions root → leaves, one depth level at a time.
     *
     * WARNING: L2L has a parent→child data dependency — a child's local
     * expansion is READ when that child acts as a parent at the next level.
     * Processing all levels in a single parallel loop would race: thread A
     * could read child C's local expansion (as C is parent of D) while
     * thread B writes C's expansion (as C is child of B's node).
     *
     * We serialize over depth to guarantee each level is fully written
     * before the next level reads it.  Nodes within a single depth level
     * are independent (a child has exactly one parent), so they can be
     * processed in parallel within each level.
     */
    for (unsigned d = 0; d < max_depth; ++d)
    {
        const unsigned ds = depth_start[d];
        const unsigned de = depth_end[d];
        if (ds >= de)
            continue;

        unsigned thread_counter_l2l = 0;
#pragma omp parallel default(none) shared(ds, de, nodes, order, work_order, local_slices, work_shift_exp, work_pse,    \
                                              n_coeffs, shift_per_thread, pse_per_thread, thread_counter_l2l)          \
    num_threads(n_threads)
        {
            unsigned tid;
#pragma omp atomic capture
            tid = thread_counter_l2l++;
            real_t *my_shift_exp = work_shift_exp + (size_t)tid * shift_per_thread;
            real_t *my_pse = work_pse + (size_t)tid * pse_per_thread;
#pragma omp for schedule(dynamic, 16)
            for (unsigned ni = ds; ni < de; ++ni)
            {
                if (nodes[ni].kind != OCTREE_NODE_INTERNAL)
                    continue;

                real_t *parent_local = local_slices[ni];
                if (parent_local == NULL)
                    continue;

                const local_expansion_t parent_loc = {.order = order,
                                                      .center = nodes[ni].center,
                                                      .coeffs_x = parent_local,
                                                      .coeffs_y = parent_local + n_coeffs,
                                                      .coeffs_z = parent_local + 2u * n_coeffs};

                for (int oct = 0; oct < 8; ++oct)
                {
                    const octree_node_t *child = nodes[ni].data.internal.children[oct];
                    if (child == NULL)
                        continue;

                    const unsigned ci = (unsigned)(child - nodes);
                    real_t *child_local = local_slices[ci];
                    if (child_local == NULL)
                        continue;

                    local_expansion_t child_loc = {.order = order,
                                                   .center = child->center,
                                                   .coeffs_x = child_local,
                                                   .coeffs_y = child_local + n_coeffs,
                                                   .coeffs_z = child_local + 2u * n_coeffs};
                    local_expansion_shift(&parent_loc, &child_loc, work_order, my_shift_exp, my_pse);
                }
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/* Particle kernel                                                    */
/* ------------------------------------------------------------------ */

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

    /* Step 1: Descend to the target's leaf using geometric centers. */
    uint32_t leaf_node_idx = 0;
    {
        uint32_t idx = 0;
        while (tree->nodes[idx].kind == OCTREE_NODE_INTERNAL)
        {
            const real3_t p = point;
            const real3_t c = tree->nodes[idx].geom_center;
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
    const bool has_local = (tree->local_coeffs != NULL && tree->local_slices != NULL);
    const bool has_vlist = (tree->vlist_offsets && tree->vlist_indices && tree->leaf_indices);

    /* --- Far-field --- */
    if (eval_settings.mode == FMM_EVAL_FMM && has_local)
    {
        /* FMM mode: evaluate the leaf's precomputed local expansion (interior only). */
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
    else if (eval_settings.mode == FMM_EVAL_HYBRID && has_local)
    {
        /*
         * HYBRID mode: stack-based traversal from root, accepting the first
         * node whose local expansion converges (|r'| < alpha * half_size).
         * This works anywhere in space because we can always accept some
         * ancestor whose cell covers the evaluation point.  If no node is
         * accepted (pathological), fall back to tree-code.
         */
        static const unsigned STACK_MAX = 256;
        unsigned stack[STACK_MAX];
        unsigned sp = 0;
        stack[sp++] = 0; /* root */

        const real_t alpha = (eval_settings.hybrid_alpha > 0.0) ? eval_settings.hybrid_alpha : 1.5;

        while (sp > 0)
        {
            const unsigned ni = stack[--sp];
            const octree_node_t *node = &tree->nodes[ni];

            real_t *local_slice = tree->local_slices[ni];
            if (local_slice != NULL)
            {
                const real3_t rl = real3_sub(point, node->center);
                const real_t dist = sqrt(rl.x * rl.x + rl.y * rl.y + rl.z * rl.z);
                if (dist < alpha * node->half_size)
                {
                    /* Convergence criterion satisfied — evaluate and prune. */
                    const local_expansion_t local = {.order = order,
                                                     .center = node->center,
                                                     .coeffs_x = local_slice,
                                                     .coeffs_y = local_slice + n_coeffs,
                                                     .coeffs_z = local_slice + 2u * n_coeffs};
                    result = local_expansion_eval(&local, point);
                    goto hybrid_done;
                }
            }

            if (node->kind == OCTREE_NODE_INTERNAL && sp + 8 <= STACK_MAX)
            {
                for (int oct = 7; oct >= 0; --oct)
                {
                    const octree_node_t *child = node->data.internal.children[oct];
                    if (child != NULL)
                        stack[sp++] = (unsigned)(child - tree->nodes);
                }
            }
        }

        /* Fallback: no node's local expansion converged — use tree-code. */
        if (has_vlist && li >= 0)
        {
            const unsigned v_start = tree->vlist_offsets[(unsigned)li];
            const unsigned v_end = tree->vlist_offsets[(unsigned)li + 1];
            for (unsigned vi = v_start; vi < v_end; ++vi)
            {
                const unsigned other_ni = tree->vlist_indices[vi];
                real_t *slice = tree->mp_slices[other_ni];
                if (slice == NULL)
                    continue;
                const octree_node_t *other = &tree->nodes[other_ni];
                const multipole_t mp = {.order = order,
                                        .center = other->center,
                                        .coeffs_x = slice,
                                        .coeffs_y = slice + n_coeffs,
                                        .coeffs_z = slice + 2u * n_coeffs};
                result = real3_add(result, multipole_eval(&mp, point));
            }
        }
    hybrid_done:;
    }
    else if (has_vlist && li >= 0)
    {
        /* Tree-code mode: per-V-list multipole_eval.
         * V-list entries are NODE indices (leaf or internal), directly usable. */
        const unsigned v_start = tree->vlist_offsets[(unsigned)li];
        const unsigned v_end = tree->vlist_offsets[(unsigned)li + 1];
        for (unsigned vi = v_start; vi < v_end; ++vi)
        {
            const unsigned other_ni = tree->vlist_indices[vi];
            real_t *slice = tree->mp_slices[other_ni];
            if (slice == NULL)
                continue;
            const octree_node_t *other = &tree->nodes[other_ni];
            const real3_t center = other->center; /* works for both leaf and internal */
            const multipole_t mp = {.order = order,
                                    .center = center,
                                    .coeffs_x = slice,
                                    .coeffs_y = slice + n_coeffs,
                                    .coeffs_z = slice + 2u * n_coeffs};
            result = real3_add(result, multipole_eval(&mp, point));
        }
    }
    else
    {
        /* Fallback: direct sum over all sources. */
        for (unsigned kk = 0; kk < tree->n_sources; ++kk)
        {
            const real3_t dr = real3_sub(point, sources_coords[kk]);
            result = real3_add(result, particle_kernel(sources_values[kk], dr));
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
                result = real3_add(result, particle_kernel(sources_values[src], dr));
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
                result = real3_add(result, particle_kernel(sources_values[src], dr));
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
                                                  eval_settings) num_threads(n_threads) schedule(dynamic)
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
    unsigned *mlvl_offsets; /**< Per-node interaction-list CSR offsets (multi-level M2L). */
    unsigned *mlvl_indices; /**< Per-node interaction-list flat indices. */
    size_t mlvl_capacity;   /**< Capacity of mlvl_indices. */
    real_t *m2l_shift_exp;  /**< Per-thread shift_exp scratch for M2L sweep. */
    real_t *m2l_pse;        /**< Per-thread pse scratch for M2L sweep.       */
    /* Morton-order sort storage. */
    uint64_t *morton_codes;         /**< [n_total] Morton code for every node. */
    unsigned *morton_sorted_nodes;  /**< [n_total] Node indices sorted by Morton code (per depth). */
    unsigned *morton_depth_offsets; /**< [max_depth+2] depth_offsets + end sentinel. */
    uint8_t *morton_pairs_temp;     /**< [2 * n_total * pair_size] Radix sort ping-pong buffer. */
} fmm_work_t;

/* ------------------------------------------------------------------ */
/* Work buffer partition helper (FMM-specific)                        */
/* ------------------------------------------------------------------ */

static fmm_work_t fmm_partition_work(fmm_work_sizes_t sizes, void *buffer, unsigned n_leaves, unsigned n_total,
                                     unsigned max_depth)
{
    const size_t pair_size = sizeof(uint64_t) + sizeof(unsigned);
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

    /* Per-leaf V-list + NF-list (tree-code mode). */
    const size_t leaf_offsets_bytes = (size_t)(n_leaves + 1) * sizeof(unsigned);
    const size_t max_per_leaf = n_leaves > 0 ? (size_t)n_leaves * (size_t)(n_leaves - 1) : 1u;
    out.vlist_offsets = (unsigned *)bp;
    bp += leaf_offsets_bytes;
    out.vlist_indices = (unsigned *)bp;
    out.vlist_capacity = max_per_leaf;
    bp += max_per_leaf * sizeof(unsigned);
    out.nflist_offsets = (unsigned *)bp;
    bp += leaf_offsets_bytes;
    out.nflist_indices = (unsigned *)bp;
    out.nflist_capacity = max_per_leaf;
    bp += max_per_leaf * sizeof(unsigned);

    /* Per-node interaction lists (multi-level M2L). */
    const size_t node_offsets_bytes = (size_t)(n_total + 1) * sizeof(unsigned);
    const size_t max_per_node = (size_t)n_total * (size_t)n_total;
    out.mlvl_offsets = (unsigned *)bp;
    bp += node_offsets_bytes;
    out.mlvl_indices = (unsigned *)bp;
    out.mlvl_capacity = max_per_node;
    bp += max_per_node * sizeof(unsigned);

    /* M2L per-thread scratch (independent of the transient scratch buffer). */
    out.m2l_shift_exp = (real_t *)bp;
    bp += sizes.m2l_shift_exp_bytes;
    out.m2l_pse = (real_t *)bp;
    bp += sizes.m2l_pse_bytes;

    /* Morton sort storage. */
    out.morton_codes = (uint64_t *)bp;
    bp += (size_t)n_total * sizeof(uint64_t);
    out.morton_sorted_nodes = (unsigned *)bp;
    bp += (size_t)n_total * sizeof(unsigned);
    out.morton_depth_offsets = (unsigned *)bp;
    bp += (size_t)(max_depth + 2u) * sizeof(unsigned);
    out.morton_pairs_temp = (uint8_t *)bp;
    bp += (size_t)n_total * pair_size * 2u;

    return out;
}

/* ------------------------------------------------------------------ */
/* Staged build helpers                                               */
/* ------------------------------------------------------------------ */

size_t fmm_scratch_size(unsigned n_sources, const fmm_settings_t *settings, unsigned n_threads)
{
    return octree_scratch_size(n_sources, n_threads, (const octree_settings_t *)settings);
}

bool fmm_prepare_scratch(void *scratch_buffer, size_t scratch_size, unsigned n_sources, unsigned n_threads,
                         const real3_t sources_coords[restrict n_sources], const fmm_settings_t settings[restrict],
                         octree_count_t *out_count, octree_scratch_t *out_scratch)
{
    if (scratch_buffer == NULL || out_count == NULL || out_scratch == NULL)
        return false;
    if (n_sources == 0 || settings == NULL || sources_coords == NULL)
        return false;

    const octree_scratch_sizes_t sz = octree_size_scratch(n_sources, (const octree_settings_t *)settings);
    if (scratch_size < octree_total_scratch_size(sz, n_threads))
        return false;

    *out_scratch = octree_scratch_partition(n_threads, scratch_buffer, sz);
    const size_t topo_bytes = (8u * (size_t)n_sources + 1u) * sizeof(topo_node_t);
    memset(out_scratch->topo, 0, topo_bytes);

    *out_count = octree_count_pass(n_sources, sources_coords, (const octree_settings_t *)settings, out_scratch->topo,
                                   out_scratch->source_leaf_topo);
    return true;
}

size_t fmm_work_size(unsigned n_sources, const fmm_settings_t *settings, const octree_count_t *count,
                     unsigned n_threads)
{
    if (count == NULL || count->n_internal + count->n_multipole_leaves + count->n_particle_leaves == 0)
        return 0;
    const fmm_work_sizes_t ws = fmm_size_work_buffer(n_sources, settings, *count, n_threads);
    return fmm_total_work_size(ws);
}

/* ------------------------------------------------------------------ */
/* fmm_tree_insert (pre-counted, pre-partitioned scratch)             */
/* ------------------------------------------------------------------ */

bool fmm_tree_insert(unsigned n_sources, unsigned n_threads, const real3_t sources_coords[restrict n_sources],
                     const real3_t sources_values[restrict n_sources], const fmm_settings_t settings[restrict],
                     const octree_count_t *count, const octree_scratch_t *scratch, const allocator_t *allocator,
                     void *buffer, size_t buffer_size, fmm_tree_t *out)
{
    bool ret = false;

    if (!buffer || !out || !scratch || !count)
        return false;
    if (n_sources == 0 || settings == NULL)
        return false;

    const unsigned n_topo = count->n_internal + count->n_multipole_leaves + count->n_particle_leaves;
    if (n_topo == 0)
        return false;

    /* Size work buffer and validate. */
    const fmm_work_sizes_t ws = fmm_size_work_buffer(n_sources, settings, *count, n_threads);
    if (buffer_size < fmm_total_work_size(ws))
        return false;

    const unsigned n_leaves = count->n_multipole_leaves + count->n_particle_leaves;
    const fmm_work_t work = fmm_partition_work(ws, buffer, n_leaves, n_topo, count->max_depth);

    /* Zero the per-thread M2L/L2L scratch buffers in the work buffer.
     *
     * These come from PyMem_Malloc and may contain stale NaN / large-exponent
     * bit patterns when blocks are reused after prior trees are freed.  The
     * M2L/L2L operators do initialise these buffers before reading, but stale
     * values at indices that are read (via multipole_add_poly_to_order's
     * ``if (c == 0.0) continue``) before being written by the monomial-loop
     * l>=1 iterations can produce inf/NaN.  BH has no M2L/L2L scratch, hence
     * no issue. */
    memset(work.m2l_shift_exp, 0, ws.m2l_shift_exp_bytes);
    memset(work.m2l_pse, 0, ws.m2l_pse_bytes);

    /* === Shared pipeline stages === */
    octree_materialize(scratch->topo, n_topo, (const octree_settings_t *)settings, work.topo_to_real, work.nodes,
                       work.multipole_coeffs, work.mp_slices, n_threads);
    octree_descend(n_sources, sources_coords, work.nodes, scratch->source_leaf_real, n_threads);

    const unsigned n_mp = octree_compute_metadata(n_topo, work.nodes, count->max_depth, NULL, NULL, n_threads);

    octree_fill_particle_order(n_sources, scratch->source_leaf_real, work.nodes, work.particle_order, n_threads);
    octree_compute_leaf_centers(n_topo, work.nodes, work.particle_order, sources_coords, sources_values, n_threads);

    if (n_mp > 0)
    {
        if (!octree_build_leaf_multipoles(n_topo, work.nodes, work.particle_order, sources_coords, sources_values,
                                          (const octree_settings_t *)settings, *scratch, n_sources, n_threads,
                                          work.mp_slices, n_mp, allocator))
            goto cleanup;
    }

    /* Upward sweep (M2M). */
    octree_run_upward_sweep(n_topo, work.nodes, count->max_depth, (const octree_settings_t *)settings, work.mp_slices,
                            scratch, work.particle_order, sources_coords, sources_values, n_threads);

    /* === FMM-specific stages === */

    /* Build interaction lists. */
    {
        fmm_build_leaf_index_map(n_topo, work.nodes, work.leaf_indices);

        /* Build Morton-sorted index arrays using invariant geom_center. */
        unsigned max_depth_found = 0;
        if (!octree_build_morton_sorted(n_topo, work.nodes, work.morton_codes, work.morton_sorted_nodes,
                                        work.morton_depth_offsets, work.morton_pairs_temp, scratch->radix_hist,
                                        count->max_depth, &max_depth_found, n_threads))
            goto cleanup;

        size_t vlist_count = 0, nflist_count = 0;
        if (!fmm_build_leaf_vlists(work.nodes, n_leaves, work.leaf_indices, work.vlist_offsets, work.vlist_indices,
                                   work.vlist_capacity, work.nflist_offsets, work.nflist_indices, work.nflist_capacity,
                                   &vlist_count, &nflist_count, n_threads, settings->theta))
            goto cleanup;

        /* Build per-node interaction lists via Morton-accelerated range queries. */
        size_t mlvl_count = 0;
        if (!fmm_build_per_node_interaction_lists(work.nodes, n_topo, work.mlvl_offsets, work.mlvl_indices,
                                                  work.mlvl_capacity, &mlvl_count, n_threads, work.morton_codes,
                                                  work.morton_sorted_nodes, work.morton_depth_offsets, max_depth_found))
            goto cleanup;

        out->leaf_indices = work.leaf_indices;
        out->vlist_offsets = work.vlist_offsets;
        out->vlist_indices = work.vlist_indices;
        out->nflist_offsets = work.nflist_offsets;
        out->nflist_indices = work.nflist_indices;
        out->vlist_count = vlist_count;
        out->nflist_count = nflist_count;
        out->n_leaves = n_leaves;
    }

    /* Multi-level M2L + L2L. */
    {
        const unsigned order = settings->order;
        const unsigned work_order = settings->work_order ? settings->work_order : settings->order;

        fmm_assign_local_slices(n_topo, order, work.local_coeffs, work.local_slices);
        fmm_m2l_sweep_mlvl(n_topo, work.nodes, order, work_order, work.mp_slices, work.local_slices, work.mlvl_offsets,
                           work.mlvl_indices, work.m2l_shift_exp, work.m2l_pse, n_threads);
        fmm_downward_l2l_sweep(n_topo, work.nodes, order, work_order, work.local_slices, work.m2l_shift_exp,
                               work.m2l_pse, n_threads);

        out->local_coeffs = work.local_coeffs;
        out->local_slices = work.local_slices;
    }

    /* Populate tree handle. */
    out->settings = *settings;
    out->root_center = work.nodes[0].center;
    out->root_half_size = work.nodes[0].half_size;
    out->n_sources = n_sources;
    out->n_nodes = n_topo;
    out->n_internal = count->n_internal;
    out->n_multipole_leaves = n_mp;
    out->n_particle_leaves = count->n_particle_leaves;
    out->max_depth_reached = count->max_depth;
    out->buffer = (uint8_t *)buffer;
    out->buffer_size = buffer_size;
    out->nodes = work.nodes;
    out->particle_order = work.particle_order;
    out->multipole_coeffs = work.multipole_coeffs;
    out->mp_slices = work.mp_slices;

    ret = true;

cleanup:
    return ret;
}

/* ------------------------------------------------------------------ */
/* fmm_tree_build (convenience — allocates internally)                */
/* ------------------------------------------------------------------ */

bool fmm_tree_build(unsigned n_sources, unsigned n_threads, const real3_t sources_coords[restrict n_sources],
                    const real3_t sources_values[restrict n_sources], const fmm_settings_t settings[restrict],
                    const allocator_t *allocator, fmm_tree_t *out)
{
    void *scratch_buffer = NULL;
    bool ret = false;

    if (!out || n_sources == 0 || settings == NULL || sources_coords == NULL || sources_values == NULL)
        return false;

    /* 1. Size scratch. */
    const size_t needed_scratch = fmm_scratch_size(n_sources, settings, n_threads);
    if (needed_scratch == 0)
        return false;

    /* 2. Allocate scratch. */
    scratch_buffer = octree_alloc(allocator, needed_scratch);
    if (!scratch_buffer)
        return false;

    /* Zero scratch to prevent stale data in per-thread pse/shift_exp.
     * Valgrind confirms 5.7M uninitialised reads from multipole_add_poly_to_order
     * in the upward sweep when this is omitted.  BH skips this because its
     * eval accuracy checks are looser and the uninit patterns rarely produce
     * inf/NaN in that code path, but FMM's M2L chain amplifies the garbage. */
    memset(scratch_buffer, 0, needed_scratch);

    /* 3. Prepare scratch (partition + count). */
    octree_count_t count;
    octree_scratch_t scratch;
    if (!fmm_prepare_scratch(scratch_buffer, needed_scratch, n_sources, n_threads, sources_coords, settings, &count,
                             &scratch))
        goto cleanup;

    /* 4. Size work buffer. */
    const size_t total_work_size = fmm_work_size(n_sources, settings, &count, n_threads);
    if (total_work_size == 0)
        goto cleanup;

    /* 5. Allocate work buffer. */
    void *work_buffer = octree_alloc(allocator, total_work_size);
    if (!work_buffer)
        goto cleanup;

    /* 6. Full pipeline (insert). */
    if (!fmm_tree_insert(n_sources, n_threads, sources_coords, sources_values, settings, &count, &scratch, allocator,
                         work_buffer, total_work_size, out))
    {
        octree_free(allocator, work_buffer);
        goto cleanup;
    }

    ret = true;

cleanup:
    /* 7. Release scratch. */
    octree_free(allocator, scratch_buffer);
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
