#pragma once
/*
 * FMM local-expansion evaluation kernel (L2P) for OpenCL C.
 *
 * Phase 5 - GPU offload of the FMM evaluation step.
 *
 * The FMM tree is built on the CPU (fmm_tree_build), which performs the
 * full upward sweep (P2M + M2M) and downward sweep (M2L + L2L) and stores
 * the resulting local expansion coefficients in fmm_tree_t::local_coeffs.
 * This kernel only performs the final L2P step on the GPU:
 *
 *   1. Descend the flat octree from the root to the leaf containing the
 *      target point (using geometric cell centres and the child_mask /
 *      child_base index arithmetic).
 *   2. Evaluate the leaf's precomputed local expansion at the target
 *      point (Horner-style monomial evaluation, identical to
 *      local_expansion_eval in cvl_cl_fmm_ops.h.cl).  The expansion is
 *      centred at the leaf's Γ-weighted centroid (eval_centers), NOT the
 *      geometric centre stored in the flat node - the coefficients are
 *      relative to the centroid.
 *  3. Add the near-field contribution: a direct 1/|r|^2 sum over the
 *      particles stored in the leaf (the leaf's own sources).
 *
 * The kernel is self-contained - it does not call the device-side
 * functions from cvl_cl_fmm_ops.h.cl.  This avoids the private-memory
 * pressure of the shared shift_exp / pse scratch and the struct-setup
 * overhead, and gives full control over the descent + evaluation.
 *
 * The flat node layout matches cvl_cl_flat_node_t (64 bytes), but the
 * `center` field holds the GEOMETRIC centre (geom_center) used for
 * descent.  The Γ-weighted centroid used for local-expansion evaluation
 * is passed separately in @p eval_centers so the descent split planes
 * line up with the children's geometric positions:
 *   center (24B) | half_size (8B) | morton_code (8B) | child_base (4B)
 *   | particle_begin (4B) | child_mask (1B) | kind (1B)
 *   | particle_count (2B) | pad (12B)
 *
 * Guarded by __OPENCL_C_VERSION__ so the header is empty on the host.
 */

#ifdef __OPENCL_C_VERSION__

/* ------------------------------------------------------------------ */
/*  Types - flat node (64 bytes, matches cvl_cl_flat_node_t)           */
/* ------------------------------------------------------------------ */

typedef enum
{
    FMM_L2P_NODE_INTERNAL = 0,
    FMM_L2P_NODE_PARTICLE = 1,
    FMM_L2P_NODE_MULTIPOLE = 2,
} fmm_l2p_node_kind_t;

typedef struct
{
    real3_t center;       /* 24 bytes                                */
    real_t half_size;     /*  8 bytes                                */
    ulong morton_code;    /*  8 bytes  (unused by this kernel)        */
    int child_base;       /*  4 bytes  (index of first child, -1 leaf)*/
    int particle_begin;   /*  4 bytes  (index into particle_order)    */
    uchar child_mask;     /*  1 byte   (bitmask of occupied octants)  */
    uchar kind;           /*  1 byte   (fmm_l2p_node_kind_t)          */
    short particle_count; /*  2 bytes  (number of particles in leaf)  */
    int leaf_id;          /*  4 bytes  (leaf index for nflist, -1=internal) */
    uchar pad[8];         /*  8 bytes  → total 64 bytes               */
} fmm_l2p_flat_node_t;

/* ------------------------------------------------------------------ */
/*  Child-index helper (popcount trick)                               */
/* ------------------------------------------------------------------ */

/**
 * @brief Return the position of the given octant in the compact children
 *        array (i.e. the number of set bits in child_mask below octant).
 *
 * OpenCL C provides popcount() as a built-in.
 */
static inline int fmm_l2p_child_index(uchar child_mask, unsigned int octant)
{
    uchar mask_below = child_mask & ((uchar)((1u << octant) - 1u));
    return (int)popcount(mask_below);
}

/* ------------------------------------------------------------------ */
/*  Local-expansion evaluation (L2P) - inlined monomial Horner        */
/* ------------------------------------------------------------------ */

/**
 * @brief Evaluate a local expansion at a target point.
 *
 * Identical arithmetic to local_expansion_eval() in cvl_cl_fmm_ops.h.cl,
 * inlined here to keep the kernel self-contained.  The local expansion
 * is a regular polynomial (no 1/r^n scaling), so this is a straight
 * Horner-style monomial evaluation over the tetrahedral coefficient
 * layout (multipole_coeff_index ordering).
 *
 * @param order       Expansion order.
 * @param center      Expansion centre (absolute coordinates).
 * @param coeffs_x    x-component coefficients [n_coeffs].
 * @param coeffs_y    y-component coefficients [n_coeffs].
 * @param coeffs_z    z-component coefficients [n_coeffs].
 * @param point       Target point (absolute coordinates).
 * @return Vector value of the expansion at @p point.
 */
static inline real3_t fmm_l2p_local_eval(unsigned order, real3_t center, __global const real_t *coeffs_x,
                                         __global const real_t *coeffs_y, __global const real_t *coeffs_z,
                                         real3_t point)
{
    real3_t rel = real3_sub(point, center);
    real3_t res = {0, 0, 0};
    size_t idx = 0;

    for (unsigned m = 0; m <= order; ++m)
    {
        real3_t term = {0, 0, 0};
        real_t px = 1.0;
        for (unsigned p = 0; p <= m; ++p)
        {
            real_t py = px;
            for (unsigned q = 0; q <= m - p; ++q)
            {
                real_t pz = py;
                for (unsigned r = 0; r <= m - p - q; ++r)
                {
                    term.x += coeffs_x[idx] * pz;
                    term.y += coeffs_y[idx] * pz;
                    term.z += coeffs_z[idx] * pz;
                    pz *= rel.z;
                    idx += 1;
                }
                py *= rel.y;
            }
            px *= rel.x;
        }
        res.x += term.x;
        res.y += term.y;
        res.z += term.z;
    }
    return res;
}

/* ------------------------------------------------------------------ */
/*  Kernel: fmm_l2p_eval - one work-item per target                   */
/* ------------------------------------------------------------------ */

/**
 * @brief Evaluate the FMM local expansion + near-field for every target.
 *
 * Each work-item handles one target point:
 *   1. Descend the flat octree from the root (index 0) to the leaf
 *      containing the target, using geometric centres and the
 *      child_mask / child_base index arithmetic.  Descent stops at the
 *      first leaf (kind != INTERNAL) or when a child is missing in the
 *      required octant.
 *   2. Evaluate the leaf's precomputed local expansion at the target.
 *   3. Add the near-field direct sum over the leaf's own particles.
 *
 * The local expansion coefficients are laid out as
 *   [3 * n_coeffs * n_nodes]
 * with n_coeffs = (order+1)(order+2)(order+3)(order+4)/24 per component.
 * For node ni, component c (0=x, 1=y, 2=z), coefficient k:
 *   local_coeffs[ni * 3 * n_coeffs + c * n_coeffs + k]
 *
 * @param[in]  nodes           Flat node array [n_nodes] (64 B each).
 *                             The `center` field holds the GEOMETRIC
 *                             centre (used for descent).
 * @param[in]  eval_centers    Γ-weighted centroids used for local-
 *                             expansion evaluation [3 * n_nodes].
 *                             For leaves with a local expansion this is
 *                             the centre the coefficients are relative
 *                             to; for internal nodes it is unused by
 *                             this kernel.
 * @param[in]  particle_order  Per-leaf particle indices [n_sources].
 * @param[in]  local_coeffs    Local expansion coefficients
 *                             [3 * n_coeffs * n_nodes].
 * @param[in]  sources_pos     Source positions [3 * n_sources].
 * @param[in]  sources_val     Source strengths [3 * n_sources].
 * @param[in]  targets         Target positions [3 * n_targets].
 * @param[in]  n_targets       Number of targets (= global work size).
 * @param[in]  n_nodes         Total number of flat nodes.
 * @param[in]  max_depth       Maximum tree depth (descent cap).
 * @param[in]  order           Local expansion order.
 * @param[in]  n_coeffs        Coefficients per component per node
 *                             (= multipole_num_coeffs(order)).
 * @param[out] results         Output field [3 * n_targets].
 */
__kernel void fmm_l2p_eval(__global const fmm_l2p_flat_node_t *nodes, __global const real_t *eval_centers,
                           __global const unsigned int *particle_order, __global const real_t *local_coeffs,
                           __global const real_t *sources_pos, __global const real_t *sources_val,
                           __global const real_t *targets, unsigned int n_targets, unsigned int n_nodes,
                           unsigned int max_depth, unsigned int order, unsigned int n_coeffs,
                           __global const unsigned int *nflist_offsets, __global const unsigned int *nflist_indices,
                           __global const unsigned int *leaf_indices, __global const int *child_indices,
                           __global const real_t *mp_coeffs, __global real_t *results)
{
    unsigned int tid = get_global_id(0);
    if (tid >= n_targets)
        return;

    /* Load target point. */
    real3_t point;
    point.x = targets[3u * tid];
    point.y = targets[3u * tid + 1u];
    point.z = targets[3u * tid + 2u];

    /* ---- 1. Descend to leaf ---- */
    unsigned int ni = 0; /* root */
    for (unsigned int d = 0; d <= max_depth; ++d)
    {
        if (ni >= n_nodes)
            break;
        fmm_l2p_flat_node_t node = nodes[ni];
        if (node.kind != FMM_L2P_NODE_INTERNAL)
            break; /* leaf */

        /* Compute octant from geometric centre. */
        real3_t c = node.center;
        unsigned int oct = 0;
        if (point.x >= c.x)
            oct |= 1u;
        if (point.y >= c.y)
            oct |= 2u;
        if (point.z >= c.z)
            oct |= 4u;

        /* Look up child index directly from the child_indices array. */
        int next = child_indices[8u * ni + oct];
        if (next < 0 || (unsigned int)next >= n_nodes)
            break;
        ni = (unsigned int)next;
    }

    real3_t acc = {0, 0, 0};

    /* ---- 2. Evaluate local expansion at the leaf ---- */
    /* Debug: write leaf node index to results for debugging */
    /* ---- 2. Evaluate local expansion at the leaf ---- */
    /* Only evaluate if this is a leaf (leaf_id >= 0).  Internal nodes
     * have leaf_id = -1 and no local expansion to evaluate (matching
     * the CPU fmm_tree_eval FMM mode behavior). */
    if (ni < n_nodes && order > 0)
    {
        fmm_l2p_flat_node_t leaf = nodes[ni];
        if (leaf.leaf_id >= 0)
        {
            real3_t eval_center;
            eval_center.x = eval_centers[3u * ni];
            eval_center.y = eval_centers[3u * ni + 1u];
            eval_center.z = eval_centers[3u * ni + 2u];
            size_t base = (size_t)ni * 3u * (size_t)n_coeffs;
            __global const real_t *cx = local_coeffs + base;
            __global const real_t *cy = local_coeffs + base + (size_t)n_coeffs;
            __global const real_t *cz = local_coeffs + base + 2u * (size_t)n_coeffs;
            acc = real3_add(acc, fmm_l2p_local_eval(order, eval_center, cx, cy, cz, point));
        }
    }

    /* ---- 3. Near-field: direct sum over the leaf's own particles ---- */
    if (ni < n_nodes)
    {
        fmm_l2p_flat_node_t leaf = nodes[ni];
        /* Only leaves (leaf_id >= 0) have particles.  Internal nodes have
         * particle_begin=0, particle_count=0, so this is a no-op for them. */
        int begin = leaf.particle_begin;
        int count = leaf.particle_count;
        if (begin >= 0 && count > 0)
        {
            for (int k = 0; k < count; ++k)
            {
                unsigned int src = particle_order[begin + k];
                real3_t dr;
                dr.x = point.x - sources_pos[3u * src];
                dr.y = point.y - sources_pos[3u * src + 1u];
                dr.z = point.z - sources_pos[3u * src + 2u];
                real_t r2 = dr.x * dr.x + dr.y * dr.y + dr.z * dr.z;
                if (r2 > (real_t)1e-30)
                {
                    real_t inv_r2 = (real_t)1.0 / r2;
                    acc.x += sources_val[3u * src] * inv_r2;
                    acc.y += sources_val[3u * src + 1u] * inv_r2;
                    acc.z += sources_val[3u * src + 2u] * inv_r2;
                }
            }
        }

        /* ---- 3b. Near-field: neighbour leaves via nflist ---- */
        int leaf_id = leaf.leaf_id;
        if (leaf_id >= 0 && nflist_offsets != 0 && nflist_indices != 0 && leaf_indices != 0)
        {
            unsigned int nf_start = nflist_offsets[leaf_id];
            unsigned int nf_end = nflist_offsets[leaf_id + 1];
            for (unsigned int nfi = nf_start; nfi < nf_end; ++nfi)
            {
                unsigned int li_other = nflist_indices[nfi];
                unsigned int other_ni = leaf_indices[li_other];
                if (other_ni >= n_nodes)
                    continue;
                fmm_l2p_flat_node_t other = nodes[other_ni];
                int obegin = other.particle_begin;
                int ocount = other.particle_count;
                if (obegin < 0 || ocount <= 0)
                    continue;
                for (int k = 0; k < ocount; ++k)
                {
                    unsigned int src = particle_order[obegin + k];
                    real3_t dr;
                    dr.x = point.x - sources_pos[3u * src];
                    dr.y = point.y - sources_pos[3u * src + 1u];
                    dr.z = point.z - sources_pos[3u * src + 2u];
                    real_t r2 = dr.x * dr.x + dr.y * dr.y + dr.z * dr.z;
                    if (r2 > (real_t)1e-30)
                    {
                        real_t inv_r2 = (real_t)1.0 / r2;
                        acc.x += sources_val[3u * src] * inv_r2;
                        acc.y += sources_val[3u * src + 1u] * inv_r2;
                        acc.z += sources_val[3u * src + 2u] * inv_r2;
                    }
                }
            }
        }
    }

    /* ---- 4. Fallback: if leaf_id < 0 (internal node or outside bbox),
     *          evaluate root multipole (same as CPU fmm_tree_eval). ---- */
    if (ni < n_nodes)
    {
        fmm_l2p_flat_node_t leaf = nodes[ni];
        if (leaf.leaf_id < 0 && mp_coeffs != 0)
        {
            /* Evaluate root multipole at node 0. */
            real3_t root_center;
            root_center.x = eval_centers[0];
            root_center.y = eval_centers[1];
            root_center.z = eval_centers[2];
            size_t mp_base = 0;
            __global const real_t *cx = mp_coeffs + mp_base;
            __global const real_t *cy = mp_coeffs + mp_base + (size_t)n_coeffs;
            __global const real_t *cz = mp_coeffs + mp_base + 2u * (size_t)n_coeffs;

            /* multipole_eval: V = sum_{m=0}^{order} (1/|r|^2)^{m+1} * P_m(r_rel) */
            real3_t r_rel;
            r_rel.x = point.x - root_center.x;
            r_rel.y = point.y - root_center.y;
            r_rel.z = point.z - root_center.z;
            real_t rr = r_rel.x * r_rel.x + r_rel.y * r_rel.y + r_rel.z * r_rel.z;
            if (rr > (real_t)1e-30)
            {
                real_t inv_r = (real_t)1.0 / sqrt(rr);
                real_t inv_r2 = inv_r * inv_r;
                real_t scale = inv_r2;
                size_t idx = 0;
                for (unsigned m = 0; m <= order; ++m)
                {
                    real_t px = (real_t)1.0;
                    real3_t term = {0, 0, 0};
                    for (unsigned p = 0; p <= m; ++p)
                    {
                        real_t py = px;
                        for (unsigned q = 0; q <= m - p; ++q)
                        {
                            real_t pz = py;
                            for (unsigned r = 0; r <= m - p - q; ++r)
                            {
                                term.x += cx[idx] * pz;
                                term.y += cy[idx] * pz;
                                term.z += cz[idx] * pz;
                                pz *= r_rel.z;
                                idx += 1;
                            }
                            py *= r_rel.y;
                        }
                        px *= r_rel.x;
                    }
                    acc.x += term.x * scale;
                    acc.y += term.y * scale;
                    acc.z += term.z * scale;
                    scale *= inv_r2;
                }
            }
        }
    }

    results[3u * tid] = acc.x;
    results[3u * tid + 1u] = acc.y;
    results[3u * tid + 2u] = acc.z;
}

#endif /* __OPENCL_C_VERSION__ */
