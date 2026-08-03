#pragma once
/*
 * BH multipole-coefficient kernels (P2M + M2M) for OpenCL C device
 * compilation.
 *
 * These kernels complete the GPU BH pipeline: the GPU tree builder
 * (cvl_cl_gpu_tree_build_t / bh_build.cl.h) produces the flat node
 * structure (nodes + particle_order + depth_offsets) but deliberately
 * does NOT compute any multipole coefficients.  This file provides the
 * two missing stages:
 *
 *   1. kernel_p2m_leaves  - one work-item per leaf: computes the
 *      Gamma-weighted centroid, updates node.center to it (matching the
 *      CPU trees, where the multipole origin is the weighted centroid),
 *      and accumulates the leaf's particles into a fresh multipole
 *      expansion (P2M, via multipole_update).
 *
 *   2. kernel_build_internal_m2m - one work-item per internal node at a
 *      given depth: computes the parent node (centroid from the updated
 *      child centres, octant mask, child base) AND shifts each child's
 *      multipole to the parent's centre (M2M, via multipole_add_shift),
 *      P2M-ing any PARTICLE children's sources directly into the parent.
 *      Call once per depth level, bottom-up (max_depth-1 down to 0),
 *      mirroring the CPU upward sweep.  Must run AFTER kernel_p2m_leaves.
 *
 * The coefficient layout is the same as the CPU multipole library:
 * for each node the buffer holds 3 * n_coeffs real_t values, ordered
 * component-major (x block, y block, z block), with each block using
 * the tetrahedral monomial index from multipole_coeff_index().
 * bh_flat_eval.cl.h reads exactly this layout.
 *
 * Depends on the shared-header chain (cvl_cl_types -> cvl_cl_math ->
 * cvl_cl_multipole -> cvl_cl_multipole_ops), which provides
 * multipole_update (P2M), multipole_add_shift (M2M), and the
 * coefficient-index helpers.
 */

#include "cvl_cl_multipole_ops.h.cl"

#ifdef __OPENCL_C_VERSION__

/* ------------------------------------------------------------------ */
/*  Flat node struct (64 bytes, matches bh_build.cl.h / bh_flat_eval   */
/*  and cvl_cl_flat_node_t)                                            */
/* ------------------------------------------------------------------ */

#define BH_COEFF_KIND_INTERNAL 0
#define BH_COEFF_KIND_PARTICLE 1
#define BH_COEFF_KIND_MULTIPOLE 2

typedef struct
{
    real3_t center;       /* 24  */
    real_t half_size;     /*  8  */
    ulong morton_code;    /*  8  */
    int child_base;       /*  4  */
    int particle_begin;   /*  4  */
    uchar child_mask;     /*  1  */
    uchar kind;           /*  1  */
    short particle_count; /*  2  */
    uchar pad[12];        /* 12  */
} bh_coeff_node_t;        /* 64  */

/* ================================================================== */
/*  Kernel 1 - leaf P2M                                                */
/* ================================================================== */
/*  One work-item per leaf (any leaf kind).  Computes the Γ-weighted   */
/*  centroid of the leaf's particles, writes it to node.center, and    */
/*  accumulates the particles into the node's multipole expansion.     */
/*  The scratch regions (cur/nxt) are per-work-item and carved from a  */
/*  caller-provided global buffer: [n_leaves][2 * scratch_size].       */
/* ================================================================== */

__kernel void kernel_p2m_leaves(__global bh_coeff_node_t *nodes,         /* [n_nodes] - leaf centers updated in place */
                                unsigned n_leaves, unsigned leaf_offset, /* depth_offsets[max_depth] */
                                unsigned n_nodes,                        /* total nodes (sanity) */
                                __global const unsigned *particle_order, /* [n_sources] */
                                __global const real_t *sources_pos,      /* [3 * n_sources] */
                                __global const real_t *sources_val,      /* [3 * n_sources] */
                                unsigned order, unsigned work_order,     /* work_order 0 = use order */
                                __global real_t *coeffs,                 /* [n_nodes * 3 * n_coeffs] */
                                __global real_t *scratch)                /* [n_leaves * 2 * scratch_size] */
{
    uint lid = get_global_id(0);
    if (lid >= n_leaves)
        return;

    /* Work-order: 0 means "use order" (matches the CPU resolve). */
    unsigned wo = (work_order == 0) ? order : work_order;
    if (wo < order)
        wo = order;

    size_t n_coeffs = multipole_num_coeffs(order);
    size_t scratch_sz = multipole_scratch_size(wo);

    /* Per-work-item scratch: one cur/nxt pair (multipole_update writes
     * all three components with the same pair). */
    size_t base = (size_t)lid * 2u * scratch_sz;
    real_t *cur = scratch + base;
    real_t *nxt = cur + scratch_sz;

    uint ni = leaf_offset + lid; /* leaf nodes are contiguous at the deepest level */
    if (ni >= n_nodes)
        return;
    bh_coeff_node_t node = nodes[ni];

    uint n_p = (uint)node.particle_count;
    uint begin = (uint)node.particle_begin;

    /* Zero the leaf's coefficient blocks. */
    size_t cbase = (size_t)ni * 3u * n_coeffs;
    for (size_t i = 0; i < n_coeffs; ++i)
    {
        coeffs[cbase + i] = 0.0;
        coeffs[cbase + n_coeffs + i] = 0.0;
        coeffs[cbase + 2u * n_coeffs + i] = 0.0;
    }

    /* Γ-weighted centroid. */
    real_t wx = 0, wy = 0, wz = 0;
    real_t wsum = 0;
    for (uint k = 0; k < n_p; ++k)
    {
        uint src = particle_order[begin + k];
        real_t gx = sources_val[3u * src];
        real_t gy = sources_val[3u * src + 1u];
        real_t gz = sources_val[3u * src + 2u];
        real_t w = sqrt(gx * gx + gy * gy + gz * gz);
        wx += sources_pos[3u * src] * w;
        wy += sources_pos[3u * src + 1u] * w;
        wz += sources_pos[3u * src + 2u] * w;
        wsum += w;
    }
    real3_t center = node.center;
    if (wsum > (real_t)0.0)
    {
        center.x = wx / wsum;
        center.y = wy / wsum;
        center.z = wz / wsum;
    }
    /* Fall back to the geometric centroid (already in node.center) for
     * zero-strength leaves; multipole_update falls back to it as well. */

    /* Update node.center to the weighted centroid (like the CPU tree,
     * where the multipole origin is the Γ-weighted centroid). */
    nodes[ni].center = center;

    /* P2M: accumulate each particle into the expansion. */
    multipole_t mp;
    mp.order = order;
    mp.center = center;
    mp.coeffs_x = coeffs + cbase;
    mp.coeffs_y = coeffs + cbase + n_coeffs;
    mp.coeffs_z = coeffs + cbase + 2u * n_coeffs;

    for (uint k = 0; k < n_p; ++k)
    {
        uint src = particle_order[begin + k];
        real3_t pos = {sources_pos[3u * src], sources_pos[3u * src + 1u], sources_pos[3u * src + 2u]};
        real3_t val = {sources_val[3u * src], sources_val[3u * src + 1u], sources_val[3u * src + 2u]};
        multipole_update(&mp, pos, val, cur, nxt);
    }
}

/* ================================================================== */
/*  Kernel 2 - combined internal build + M2M (one depth level)         */
/* ================================================================== */
/*  One work-item per internal node at the given depth.  This is the   */
/*  counterpart of kernel_build_internal (bh_build.cl.h) but it ALSO   */
/*  shifts every child's multipole up to the parent (M2M) in the same  */
/*  launch.  It must run AFTER kernel_p2m_leaves so the child centres   */
/*  are the Γ-weighted centroids: the parent centre is computed from    */
/*  those, and multipole_add_shift uses exactly that parent centre.     */
/*  Children are found via the flat child index arithmetic.             */
/*  Scratch (shift_exp + pse) is per-work-item.                         */
/* ================================================================== */

__kernel void kernel_build_internal_m2m(
    __global bh_coeff_node_t *nodes,                               /* [n_nodes] - parents written, children read */
    unsigned depth,                                                /* parent depth */
    unsigned n_parents,                                            /* depth_counts[depth] */
    unsigned parent_offset,                                        /* depth_offsets[depth] */
    unsigned child_offset,                                         /* depth_offsets[depth + 1] */
    unsigned n_children,                                           /* depth_counts[depth + 1] */
    real_t root_half_size, __global const unsigned *parent_starts, /* [n_parents] child-range starts */
    __global const unsigned *particle_order,                       /* [n_sources] */
    __global const real_t *sources_pos,                            /* [3 * n_sources] */
    __global const real_t *sources_val,                            /* [3 * n_sources] */
    unsigned order, unsigned work_order, __global real_t *coeffs,  /* [n_nodes * 3 * n_coeffs] */
    __global real_t *scratch) /* [n_parents * (3 * (wo+1)^2 + 2*n_coeffs(wo) + 2*scratch)] */
{
    uint p = get_global_id(0);
    if (p >= n_parents)
        return;

    unsigned wo = (work_order == 0) ? order : work_order;
    if (wo < order)
        wo = order;

    size_t n_coeffs = multipole_num_coeffs(order);
    size_t wo_coeffs = multipole_num_coeffs(wo);
    size_t scratch_sz = multipole_scratch_size(wo);
    size_t shift_dim = (size_t)wo + 1;
    size_t shift_plane = shift_dim * shift_dim;
    /* Per-work-item scratch: shift_exp + pse (M2M) + one cur/nxt pair (P2M
     * of PARTICLE children into the parent). */
    size_t per_wg = 3u * shift_plane + 2u * wo_coeffs + 2u * scratch_sz;

    size_t sbase = (size_t)p * per_wg;
    real_t *shift_exp = scratch + sbase;
    real_t *pse = shift_exp + 3u * shift_plane;
    real_t *p2m_cur = pse + 2u * wo_coeffs;
    real_t *p2m_nxt = p2m_cur + scratch_sz;

    uint ni = parent_offset + p;
    if (ni >= child_offset)
        return;
    __global bh_coeff_node_t *parent = &nodes[ni];

    uint child_start = parent_starts[p];
    uint child_end = (p + 1 < n_parents) ? parent_starts[p + 1] : (child_offset + n_children);

    if (child_start >= child_end)
    {
        /* Degenerate parent - no children. */
        parent->child_base = -1;
        parent->child_mask = 0;
        parent->kind = BH_COEFF_KIND_INTERNAL;
        parent->morton_code = 0;
        return;
    }

    /* ---- accumulate child data: centroid + octant mask ---- */
    real_t cx = 0, cy = 0, cz = 0;
    real_t tot = 0;
    uchar mask = 0;

    for (uint ci = child_start; ci < child_end; ++ci)
    {
        bh_coeff_node_t ch = nodes[ci];
        ulong ccode = ch.morton_code >> (60u - 3u * depth);
        uchar oct = (uchar)(ccode & 0x7u);
        mask |= (uchar)(1u << oct);

        real_t w = (ch.kind != BH_COEFF_KIND_INTERNAL) ? (real_t)ch.particle_count : (real_t)1.0;
        cx += ch.center.x * w;
        cy += ch.center.y * w;
        cz += ch.center.z * w;
        tot += w;
    }

    real3_t pcenter;
    if (tot > (real_t)0.0)
    {
        pcenter.x = cx / tot;
        pcenter.y = cy / tot;
        pcenter.z = cz / tot;
    }
    else
    {
        pcenter = nodes[child_start].center;
    }

    /* ---- write the parent node ---- */
    parent->center = pcenter;
    parent->half_size = root_half_size / (real_t)(1u << depth);
    parent->morton_code = nodes[child_start].morton_code;
    parent->child_base = (int)child_start;
    parent->child_mask = mask;
    parent->particle_begin = 0;
    parent->particle_count = 0;
    parent->kind = BH_COEFF_KIND_INTERNAL;

    /* ---- zero the parent's coefficient blocks ---- */
    size_t cbase = (size_t)ni * 3u * n_coeffs;
    for (size_t i = 0; i < n_coeffs; ++i)
    {
        coeffs[cbase + i] = 0.0;
        coeffs[cbase + n_coeffs + i] = 0.0;
        coeffs[cbase + 2u * n_coeffs + i] = 0.0;
    }

    multipole_t parent_mp;
    parent_mp.order = order;
    parent_mp.center = pcenter;
    parent_mp.coeffs_x = coeffs + cbase;
    parent_mp.coeffs_y = coeffs + cbase + n_coeffs;
    parent_mp.coeffs_z = coeffs + cbase + 2u * n_coeffs;

    /* ---- M2M: shift each non-particle child into the parent; P2M the
     * particle children's sources directly into the parent (matching the
     * CPU upward sweep) ---- */
    for (uint ci = child_start; ci < child_end; ++ci)
    {
        bh_coeff_node_t child = nodes[ci];

        if (child.kind == BH_COEFF_KIND_PARTICLE)
        {
            /* P2M: accumulate the particle child's sources into the parent
             * expansion at the parent centre. */
            multipole_t mp;
            mp.order = order;
            mp.center = pcenter;
            mp.coeffs_x = parent_mp.coeffs_x;
            mp.coeffs_y = parent_mp.coeffs_y;
            mp.coeffs_z = parent_mp.coeffs_z;

            uint n_p = (uint)child.particle_count;
            uint begin = (uint)child.particle_begin;
            for (uint k = 0; k < n_p; ++k)
            {
                uint src = particle_order[begin + k];
                real3_t pos = {sources_pos[3u * src], sources_pos[3u * src + 1u], sources_pos[3u * src + 2u]};
                real3_t val = {sources_val[3u * src], sources_val[3u * src + 1u], sources_val[3u * src + 2u]};
                multipole_update(&mp, pos, val, p2m_cur, p2m_nxt);
            }
            continue;
        }

        size_t cb = (size_t)ci * 3u * n_coeffs;
        multipole_t child_mp;
        child_mp.order = order;
        child_mp.center = child.center;
        child_mp.coeffs_x = coeffs + cb;
        child_mp.coeffs_y = coeffs + cb + n_coeffs;
        child_mp.coeffs_z = coeffs + cb + 2u * n_coeffs;

        multipole_add_shift(&child_mp, &parent_mp, wo, shift_exp, pse);
    }
}

#endif /* __OPENCL_C_VERSION__ */
