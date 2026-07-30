#pragma once
/*
 * OpenCL C kernels for building the uniform flat octree on GPU.
 *
 * Pipeline (call in order):
 *   1. kernel_morton           – compute 63-bit Morton codes for all particles
 *   2. kernel_radix_hist       – per-pass digit histogram   (call 8× for 64-bit keys)
 *   3. kernel_radix_scatter    – per-pass radix scatter     (call 8×)
 *   4. kernel_boundary         – boundary-depth detection + bd_histogram
 *   [host] read bd_hist → compute depth_counts / depth_offsets / n_total / n_leaves
 *   5. kernel_leaf_prefix      – per-WG inclusive prefix sum of leaf-start flags
 *   [host] read wg_sums → compute wg_carry[]
 *   6. kernel_leaf_build       – apply carry, build leaf nodes, scatter particles
 *   7. kernel_build_internal   – build one depth level of parent nodes
 *                               (call for depth = max_depth-1 down to 0)
 *
 * All kernels assume global work size = n (or n_parents), local size = 256.
 */

#ifdef __OPENCL_C_VERSION__

/* ------------------------------------------------------------------ */
/*  Constants and type helpers                                         */
/* ------------------------------------------------------------------ */

#define BH_KIND_INTERNAL 0
#define BH_KIND_PARTICLE 1
#define BH_KIND_MULTIPOLE 2

#define BLD_WG 256 /* work-group size used for scan kernels */

/* ------------------------------------------------------------------ */
/*  Flat node struct (64 bytes, matches bh_flat_eval.cl.h)             */
/* ------------------------------------------------------------------ */

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
} bh_build_node_t;        /* 64  */

/* ------------------------------------------------------------------ */
/*  Morton-code helpers                                                */
/* ------------------------------------------------------------------ */

/* Spread the lower 21 bits of x so that zeros are interleaved with
   every 2 bits:  ---- ---- ---- ---- ---- ---- fedcba9876543210
   →  -f-e-d-c-b-a-9-8-7-6-5-4-3-2-1-0-                     */
static inline ulong bd_morton_split_21(ulong x)
{
    x &= 0x1fffffUL;
    x = (x | (x << 32)) & 0x1f00000000ffffUL;
    x = (x | (x << 16)) & 0x1f0000ff0000ffUL;
    x = (x | (x << 8)) & 0x100f00f00f00f00fUL;
    x = (x | (x << 4)) & 0x10c30c30c30c30c3UL;
    x = (x | (x << 2)) & 0x1249249249249249UL;
    return x;
}

/* Map a physical coordinate to its 63-bit Morton code.
   The root box has centre (cx,cy,cz) and half-side-length hs.       */
static inline ulong bd_morton_3d(real_t px, real_t py, real_t pz, real_t cx, real_t cy, real_t cz, real_t hs)
{
    real_t inv = (real_t)0.5 / hs;
    real_t nx = (px - cx) * inv + (real_t)0.5;
    real_t ny = (py - cy) * inv + (real_t)0.5;
    real_t nz = (pz - cz) * inv + (real_t)0.5;

    if (nx < 0)
        nx = 0;
    if (nx >= 1)
        nx = (real_t)0.999999;
    if (ny < 0)
        ny = 0;
    if (ny >= 1)
        ny = (real_t)0.999999;
    if (nz < 0)
        nz = 0;
    if (nz >= 1)
        nz = (real_t)0.999999;

    ulong ix = (ulong)(nx * (real_t)((1u << 21) - 1));
    ulong iy = (ulong)(ny * (real_t)((1u << 21) - 1));
    ulong iz = (ulong)(nz * (real_t)((1u << 21) - 1));

    return bd_morton_split_21(ix) | (bd_morton_split_21(iy) << 1) | (bd_morton_split_21(iz) << 2);
}

/* ================================================================== */
/*  Kernel 1 – Morton codes                                            */
/* ================================================================== */
/*  One work-item per particle.  Reads coords[3*gid] and writes        */
/*  morton_out[gid] = 63-bit Z-order code.                             */
/* ================================================================== */

__kernel void kernel_morton(__global const real_t *coords, /* [3 * n] */
                            unsigned n, real_t root_cx, real_t root_cy, real_t root_cz, real_t root_hs,
                            __global ulong *morton_out) /* [n] */
{
    uint gid = get_global_id(0);
    if (gid >= n)
        return;

    morton_out[gid] = bd_morton_3d(coords[3u * gid], coords[3u * gid + 1u], coords[3u * gid + 2u], root_cx, root_cy,
                                   root_cz, root_hs);
}

/* ================================================================== */
/*  Kernel 2 – Radix-sort digit histogram (one pass)                   */
/* ================================================================== */
/*  One work-item per particle.  Each WG builds a local 256-bin        */
/*  histogram of the digit `(key >> shift) & 0xFF`, then writes it     */
/*  out to hist_out[wg * 256 .. wg * 256 + 255].                       */
/*  Host concatenates per-WG histograms into a global histogram and    */
/*  computes the exclusive prefix that is fed to kernel_radix_scatter. */
/* ================================================================== */

__kernel void kernel_radix_hist(__global const ulong *keys,                      /* [n] */
                                unsigned n, uint shift, __global uint *hist_out) /* [n_groups * 256] */
{
    __local uint lhist[256];

    /* zero local memory — stride coverage for small WGs */
    for (uint t = get_local_id(0); t < 256; t += get_local_size(0))
        lhist[t] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    uint gid = get_global_id(0);
    if (gid < n)
    {
        ulong key = keys[gid];
        uint digit = (uint)((key >> shift) & 0xFFu);
        atomic_inc(&lhist[digit]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    uint g = get_group_id(0);
    for (uint t = get_local_id(0); t < 256; t += get_local_size(0))
        hist_out[g * 256u + t] = lhist[t];
}

/* ================================================================== */
/*  Kernel 3 – Radix-sort scatter (one pass)                           */
/* ================================================================== */
/*  One work-item per particle.  Uses pre-computed scatter offsets      */
/*  (prefix[]) to place each key + its companion index into sorted     */
/*  position.  prefix[] is shaped [n_groups × 256].                    */
/* ================================================================== */

__kernel void kernel_radix_scatter(__global const ulong *keys_in,                       /* [n] */
                                   __global const uint *indices_in,                     /* [n] */
                                   __global ulong *keys_out,                            /* [n] */
                                   __global uint *indices_out,                          /* [n] */
                                   unsigned n, uint shift, __global const uint *prefix) /* [n_groups * 256] */
{
    __local uint digit_pos[256];
    for (uint t = get_local_id(0); t < 256; t += get_local_size(0))
        digit_pos[t] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    uint gid = get_global_id(0);
    uint g = get_group_id(0);
    if (gid >= n)
        return;

    ulong key = keys_in[gid];
    uint idx = indices_in[gid];
    uint digit = (uint)((key >> shift) & 0xFFu);
    uint pos = atomic_inc(&digit_pos[digit]);
    barrier(CLK_LOCAL_MEM_FENCE);

    uint dst = prefix[g * 256u + digit] + pos;
    keys_out[dst] = key;
    indices_out[dst] = idx;
}

/* ================================================================== */
/*  Kernel 4 – Boundary-depth detection + histogram                    */
/* ================================================================== */
/*  One work-item per particle.  Computes the LCP-based boundary depth */
/*  (ZBH’s bd(i)) and atomics-increments bd_hist[bd].                  */
/*  bd_hist must be zeroed by the host before launching this kernel.   */
/*  After the kernel, host computes:                                    */
/*    depth_counts[d] = sum_{b=0..d} bd_hist[b]   (number of groups    */
/*                                                  at depth d)        */
/*    depth_offsets[0] = 0;                                            */
/*    depth_offsets[d+1] = depth_offsets[d] + depth_counts[d]          */
/*    n_total = depth_offsets[max_depth+1]                             */
/*    n_leaves = depth_counts[max_depth]                               */
/* ================================================================== */

__kernel void kernel_boundary(__global const ulong *morton_codes,                         /* [n] — sorted */
                              unsigned n, unsigned max_depth, __global int *boundary_out, /* [n] — bd(i) per particle */
                              __global volatile uint *bd_hist) /* [max_depth+2] — must be zeroed */
{
    uint gid = get_global_id(0);
    if (gid >= n)
        return;

    int bd;
    if (gid == 0)
    {
        bd = 0;
    }
    else
    {
        ulong diff = morton_codes[gid] ^ morton_codes[gid - 1];
        if (diff == 0)
        {
            bd = (int)(max_depth + 1);
        }
        else
        {
            uint lz = clz(diff); /* OpenCL built-in */
            bd = (int)((63u - lz) / 3u + 1u);
            if ((uint)bd > max_depth)
                bd = (int)(max_depth + 1);
        }
    }

    boundary_out[gid] = bd;
    atomic_inc(&bd_hist[bd]);
}

/* ================================================================== */
/*  Kernel 5 – Compact leaf-start positions (replaces leaf_prefix)     */
/* ================================================================== */
/*  One work-item per particle.  If boundary depth <= max_depth (or    */
/*  gid == 0), atomically records its position as a leaf start.        */
/*  Output: leaf_starts[n_leaves] contiguous array of start positions; */
/*  n_leaves_out[0] receives the total leaf count.                     */
/*  The caller must zero n_leaves_out before launching.                */
/* ================================================================== */

__kernel void kernel_compact_leaves(__global const int *boundary, /* [n] */
                                    unsigned n, unsigned max_depth,
                                    __global volatile unsigned *n_leaves_out, /* single uint, zeroed by host */
                                    __global unsigned *leaf_starts)           /* [n] (only first n_leaves valid) */
{
    uint gid = get_global_id(0);
    if (gid >= n)
        return;

    int is_start = (boundary[gid] <= max_depth) || (gid == 0);
    if (is_start)
    {
        uint leaf_id = atomic_inc(&n_leaves_out[0]);
        leaf_starts[leaf_id] = gid;
    }
}

/* ================================================================== */
/*  Kernel 6 – Leaf-node construction (replaces leaf_build)            */
/* ================================================================== */
/*  One work-item per leaf.  Reads leaf_starts[lid], finds leaf end by */
/*  scanning forward for the next start boundary, computes centroid     */
/*  from the original (unsorted) coords via indices_sorted, and writes */
/*  the leaf node + particle_order fragment.                           */
/*  A global particle_counter (second uint in leaf_counter, zeroed by  */
/*  host) tracks the running offset into particle_order.               */
/* ================================================================== */

__kernel void kernel_fill_leaves(__global const unsigned *leaf_starts,            /* [n_leaves] */
                                 unsigned n_leaves, __global const int *boundary, /* [n] */
                                 unsigned n, unsigned max_depth, __global bh_build_node_t *nodes,
                                 unsigned leaf_offset,                     /* depth_offsets[max_depth] */
                                 __global unsigned *particle_order,        /* [n] */
                                 __global volatile unsigned *leaf_counter, /* [2]: [n_leaves_out, particle_counter] */
                                 __global const real_t *coords,            /* [3*n] — original (unsorted) coords */
                                 __global const ulong *morton_sorted,      /* [n] — sorted Morton codes */
                                 __global const unsigned *indices_sorted,  /* [n] — sorted→original permutation */
                                 real_t root_half_size, unsigned critical_count)
{
    uint lid = get_global_id(0);
    if (lid >= n_leaves)
        return;

    uint start = leaf_starts[lid];

    /* Find leaf end: scan forward for the next boundary <= max_depth. */
    uint end = n;
    for (uint j = start + 1; j < n; ++j)
    {
        if (boundary[j] <= max_depth)
        {
            end = j;
            break;
        }
    }

    uint n_p = end - start;
    uint leaf_ni = leaf_offset + lid;

    /* Centroid from original (unsorted) coords via indices_sorted. */
    real_t cx = 0, cy = 0, cz = 0;
    for (uint k = 0; k < n_p; ++k)
    {
        uint orig = indices_sorted[start + k];
        cx += coords[3u * orig];
        cy += coords[3u * orig + 1u];
        cz += coords[3u * orig + 2u];
    }
    cx /= (real_t)n_p;
    cy /= (real_t)n_p;
    cz /= (real_t)n_p;

    /* Reserve particle_order slot using the second uint of leaf_counter. */
    uint base = atomic_add(&leaf_counter[1], n_p);

    /* Scatter particle indices (in sorted order -> contiguous). */
    for (uint k = 0; k < n_p; ++k)
        particle_order[base + k] = indices_sorted[start + k];

    /* Fill node struct (64 bytes, matches cvl_cl_flat_node_t layout). */
    bh_build_node_t node;
    node.center.x = cx;
    node.center.y = cy;
    node.center.z = cz;
    node.half_size = root_half_size / (real_t)(1u << max_depth);
    node.morton_code = morton_sorted[start];
    node.child_base = -1;
    node.particle_begin = (int)base;
    node.particle_count = (short)n_p;
    node.child_mask = 0;
    node.kind = (n_p > critical_count) ? BH_KIND_MULTIPOLE : BH_KIND_PARTICLE;
    node.pad[0] = node.pad[1] = node.pad[2] = node.pad[3] = 0;

    nodes[leaf_ni] = node;
}

/* ================================================================== */
/*  Kernel 7 – Build one internal level of the tree                    */
/* ================================================================== */
/*  Launch with n_parents work-items.  Each WI binary-searches the     */
/*  child array to find the contiguous range of children (at depth+1)  */
/*  belonging to its parent, then computes the parent's centroid and   */
/*  mask.                                                              */
/*  Call once per depth level, from max_depth-1 down to 0.            */
/* ================================================================== */

__kernel void kernel_build_internal(__global bh_build_node_t *nodes, unsigned depth, /* current parent depth        */
                                    unsigned n_parents,                              /* depth_counts[depth]         */
                                    unsigned parent_offset,                          /* depth_offsets[depth]        */
                                    unsigned child_offset,                           /* depth_offsets[depth+1]      */
                                    unsigned n_children,                             /* depth_counts[depth+1]       */
                                    real_t root_half_size)
{
    uint p = get_global_id(0);
    if (p >= n_parents)
        return;

    uint ni = parent_offset + p;
    __global bh_build_node_t *parent = &nodes[ni];

    /* Children belong to the same parent if the top 3*depth bits of
       their Morton codes match.  Since children are stored at a deeper
       level, we compare their code >> (64 - 3*depth). */
    uint shift = 64u - 3u * depth;

    /* The parent's representative code — take it from the first child.
       If the child range is empty this parent is degenerate. */
    if (child_offset >= child_offset + n_children || n_children == 0)
    {
        parent->child_base = -1;
        parent->child_mask = 0;
        parent->kind = BH_KIND_INTERNAL;
        return;
    }

    ulong parent_code = nodes[child_offset].morton_code;

    /* ---- lower bound ---- */
    uint lo = child_offset, hi = child_offset + n_children;
    while (lo < hi)
    {
        uint mid = lo + (hi - lo) / 2u;
        ulong cc = nodes[mid].morton_code;
        if ((cc >> shift) < (parent_code >> shift))
            lo = mid + 1u;
        else
            hi = mid;
    }
    uint child_start = lo;

    if (child_start >= child_offset + n_children)
    {
        parent->child_base = -1;
        parent->child_mask = 0;
        parent->kind = BH_KIND_INTERNAL;
        return;
    }

    /* ---- upper bound ---- */
    lo = child_start;
    hi = child_offset + n_children;
    while (lo < hi)
    {
        uint mid = lo + (hi - lo) / 2u;
        ulong cc = nodes[mid].morton_code;
        if ((cc >> shift) <= (parent_code >> shift))
            lo = mid + 1u;
        else
            hi = mid;
    }
    uint child_end = lo;

    /* ---- accumulate child data ---- */
    real_t cx = 0, cy = 0, cz = 0;
    real_t tot = 0;
    uchar mask = 0;

    for (uint ci = child_start; ci < child_end; ++ci)
    {
        bh_build_node_t ch = nodes[ci];
        ulong ccode = ch.morton_code >> shift;
        uchar oct = (uchar)(ccode & 0x7u);
        mask |= (uchar)(1u << oct);

        real_t w = (ch.kind != BH_KIND_INTERNAL) ? (real_t)ch.particle_count : (real_t)1.0;
        cx += ch.center.x * w;
        cy += ch.center.y * w;
        cz += ch.center.z * w;
        tot += w;
    }

    if (tot > 0)
    {
        parent->center.x = cx / tot;
        parent->center.y = cy / tot;
        parent->center.z = cz / tot;
    }

    parent->half_size = root_half_size / (real_t)(1u << depth);
    parent->morton_code = nodes[child_start].morton_code;
    parent->child_base = (int)child_start;
    parent->child_mask = mask;
    parent->particle_begin = 0;
    parent->particle_count = 0;
    parent->kind = BH_KIND_INTERNAL;
}

#endif /* __OPENCL_C_VERSION__ */
