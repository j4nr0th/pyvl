#pragma once
/*
 * BH tree evaluation on a uniform flat octree (GPU kernel).
 *
 * Each work-item evaluates one target: traverses the tree from root,
 * applying the Multipole Acceptance Criterion (theta) and accumulating
 * multipole_eval (for accepted non-leaf cells with multipole coefficients)
 * or direct particle kernel (for unaccepted leaves).
 *
 * The tree is stored as a flat array of nodes (see bh_flat_node_t).
 *
 * This file is self-contained OpenCL C -- no #include dependencies.
 * It must be guarded by __OPENCL_C_VERSION__ for compilation.
 */

#ifdef __OPENCL_C_VERSION__

/* ------------------------------------------------------------------ */
/*  Types -- same layout as cvl_cl_flat_node_t (64 bytes)             */
/* ------------------------------------------------------------------ */

typedef enum
{
    BH_NODE_INTERNAL = 0,
    BH_NODE_PARTICLE = 1,
    BH_NODE_MULTIPOLE = 2,
} bh_node_kind_t;

typedef struct
{
    real3_t center;       /* 24 bytes                                */
    real_t half_size;     /*  8 bytes                                */
    ulong morton_code;    /*  8 bytes                                */
    int child_base;       /*  4 bytes  (index of first child)        */
    int particle_begin;   /*  4 bytes  (index into particle_order)   */
    uchar child_mask;     /*  1 byte   (bitmask of occupied octants) */
    uchar kind;           /*  1 byte   (bh_node_kind_t)              */
    short particle_count; /*  2 bytes  (number of particles in leaf) */
    uchar pad[12];        /* 12 bytes  → total 64 bytes              */
} bh_flat_node_t;

/* ------------------------------------------------------------------ */
/*  MAC -- Multipole Acceptance Criterion                             */
/* ------------------------------------------------------------------ */

/**
 * @brief Decide whether a cell can be accepted (its multipole used)
 *        or must be opened and its children visited.
 *
 * @param node    The cell to test.
 * @param point   Target position.
 * @param theta   Opening angle (<= 0 means neighbour criterion).
 * @return true if the cell can be accepted, false if it must be opened.
 */
static inline bool bh_mac_accept(bh_flat_node_t node, real3_t point, real_t theta)
{
    real3_t diff;
    diff.x = point.x - node.center.x;
    diff.y = point.y - node.center.y;
    diff.z = point.z - node.center.z;

    if (theta <= (real_t)0.0)
    {
        /* Neighbour criterion: accept if point is outside 2× half_size
         * in any coordinate. */
        return fabs(diff.x) > (real_t)2.0 * node.half_size || fabs(diff.y) > (real_t)2.0 * node.half_size ||
               fabs(diff.z) > (real_t)2.0 * node.half_size;
    }

    /* Opening-angle criterion: half_size / distance < theta. */
    real_t dist = sqrt(diff.x * diff.x + diff.y * diff.y + diff.z * diff.z);
    if (dist < (real_t)1e-30)
        return false;
    return node.half_size / dist < theta;
}

/* ------------------------------------------------------------------ */
/*  Child-index helper (popcount trick)                               */
/* ------------------------------------------------------------------ */

/**
 * @brief Return the position of the given octant in the compact children
 *        array (i.e. the number of set bits in child_mask below octant).
 *
 * OpenCL C provides popcount() as a built-in function.
 */
static inline int bh_child_index(bh_flat_node_t node, unsigned int octant)
{
    uchar mask_below = node.child_mask & ((uchar)(1 << octant) - (uchar)1);
    return (int)popcount(mask_below);
}

/* ------------------------------------------------------------------ */
/*  BH tree evaluation kernel -- one work-item per target             */
/* ------------------------------------------------------------------ */

/**
 * @brief Evaluate the BH tree for every target point.
 *
 * Each work-item traverses the tree independently using a fixed-size
 * stack (BH_STACK_MAX = 128, well within the 256‑limit for most
 * devices) .
 *
 * @param[in]  nodes           Flat node array  (num_nodes entries).
 * @param[in]  particle_order  Per-leaf particle-index ordering
 *                             (num_particles entries).
 * @param[in]  depth_offsets   Start index of each depth level in nodes
 *                             (unused by this kernel, reserved).
 * @param[in]  sources_pos     Source particle positions
 *                             [3 * num_particles].
 * @param[in]  sources_val     Source particle strengths
 *                             [3 * num_particles].
 * @param[in]  n_targets       Number of targets (= global work size).
 * @param[in]  order           Multipole expansion order
 *                             (0 disables multipole evaluation).
 * @param[in]  theta           MAC opening angle (≤0 uses neighbour
 *                             criterion).
 * @param[in]  coeffs          Multipole coefficients
 *                             [3 * n_coeffs_per_node * num_nodes].
 *                             May be NULL if order == 0.
 * @param[in]  targets         Target positions [3 * n_targets].
 *                             May alias sources_pos to evaluate at the
 *                             source positions themselves.
 * @param[out] results         Output accumulator
 *                             [3 * n_targets].
 */
__kernel void bh_flat_eval(__global const bh_flat_node_t *nodes, __global const unsigned int *particle_order,
                           __global const unsigned int *depth_offsets, __global const real_t *sources_pos,
                           __global const real_t *sources_val, unsigned int n_targets, unsigned int order, real_t theta,
                           __global const real_t *coeffs, __global const real_t *targets, __global real_t *results)
{
    unsigned int tid = get_global_id(0);
    if (tid >= n_targets)
        return;

    /* Load target position. */
    real3_t point;
    point.x = targets[3u * tid];
    point.y = targets[3u * tid + 1u];
    point.z = targets[3u * tid + 2u];

    real3_t acc = {0, 0, 0};

    /* Fixed-size traversal stack.  Max octree depth = 21 for 64-bit
     * morton codes, so 128 entries is generous. */
    enum
    {
        BH_STACK_MAX = 128
    };
    int stack[BH_STACK_MAX];
    int sp = 0;
    stack[sp++] = 0; /* root node index */

    while (sp > 0)
    {
        sp -= 1;
        int ni = stack[sp];
        bh_flat_node_t node = nodes[ni];

        if (node.kind == BH_NODE_INTERNAL)
        {
            if (bh_mac_accept(node, point, theta))
            {
                /* Accepted: evaluate multipole if coefficients exist. */
                if (order > 0)
                {
                    /* Number of coefficients per component for given order:
                     *   n_coeffs = (order+1)(order+2)(order+3)(order+4) / 24
                     * This is the number of monomials in 3 variables up to
                     * total degree 'order'. */
                    size_t nc = (size_t)(order + 1) * (order + 2) * (order + 3) * (order + 4) / 24u;
                    size_t base = (size_t)ni * 3u * nc;

                    /* Evaluate multipole expansion at offset = point - center. */
                    real3_t r_rel;
                    r_rel.x = point.x - node.center.x;
                    r_rel.y = point.y - node.center.y;
                    r_rel.z = point.z - node.center.z;
                    real_t rr = r_rel.x * r_rel.x + r_rel.y * r_rel.y + r_rel.z * r_rel.z;

                    if (rr > (real_t)1e-30)
                    {
                        real_t inv_r = (real_t)1.0 / sqrt(rr);
                        real_t inv_r2 = inv_r * inv_r;
                        real_t scale = inv_r2;
                        size_t idx = 0;

                        for (unsigned int m = 0; m <= order; ++m)
                        {
                            real3_t term_m = {0, 0, 0};
                            real_t px = (real_t)1.0;

                            for (unsigned int p = 0; p <= m; ++p)
                            {
                                real_t py = px;

                                for (unsigned int q = 0; q <= m - p; ++q)
                                {
                                    real_t pz = py;

                                    for (unsigned int r = 0; r <= m - p - q; ++r)
                                    {
                                        term_m.x += coeffs[base + 0u * nc + idx] * pz;
                                        term_m.y += coeffs[base + 1u * nc + idx] * pz;
                                        term_m.z += coeffs[base + 2u * nc + idx] * pz;
                                        pz *= r_rel.z;
                                        idx += 1u;
                                    }
                                    py *= r_rel.y;
                                }
                                px *= r_rel.x;
                            }

                            acc.x += term_m.x * scale;
                            acc.y += term_m.y * scale;
                            acc.z += term_m.z * scale;
                            scale *= inv_r2;
                        }
                    }
                }
            }
            else
            {
                /* Not accepted: descend into children (reverse order so
                 * that octant 0 is processed first). */
                for (int oct = 7; oct >= 0; --oct)
                {
                    if (node.child_mask & (uchar)(1 << oct))
                    {
                        int child_idx = bh_child_index(node, (unsigned int)oct);
                        int child_ni = node.child_base + child_idx;
                        if (child_ni >= 0)
                        {
                            stack[sp++] = child_ni;
                        }
                    }
                }
            }
        }
        else
        {
            /* Leaf node (PARTICLE or MULTIPOLE kind). */
            if (node.kind == BH_NODE_MULTIPOLE && order > 0)
            {
                /* Try MAC on the leaf's own multipole. */
                if (bh_mac_accept(node, point, theta))
                {
                    /* Evaluate leaf multipole expansion. */
                    size_t nc = (size_t)(order + 1) * (order + 2) * (order + 3) * (order + 4) / 24u;
                    size_t base = (size_t)ni * 3u * nc;

                    real3_t r_rel;
                    r_rel.x = point.x - node.center.x;
                    r_rel.y = point.y - node.center.y;
                    r_rel.z = point.z - node.center.z;
                    real_t rr = r_rel.x * r_rel.x + r_rel.y * r_rel.y + r_rel.z * r_rel.z;

                    if (rr > (real_t)1e-30)
                    {
                        real_t inv_r = (real_t)1.0 / sqrt(rr);
                        real_t inv_r2 = inv_r * inv_r;
                        real_t scale = inv_r2;
                        size_t idx = 0;

                        for (unsigned int m = 0; m <= order; ++m)
                        {
                            real3_t term_m = {0, 0, 0};
                            real_t px = (real_t)1.0;

                            for (unsigned int p = 0; p <= m; ++p)
                            {
                                real_t py = px;

                                for (unsigned int q = 0; q <= m - p; ++q)
                                {
                                    real_t pz = py;

                                    for (unsigned int r = 0; r <= m - p - q; ++r)
                                    {
                                        term_m.x += coeffs[base + 0u * nc + idx] * pz;
                                        term_m.y += coeffs[base + 1u * nc + idx] * pz;
                                        term_m.z += coeffs[base + 2u * nc + idx] * pz;
                                        pz *= r_rel.z;
                                        idx += 1u;
                                    }
                                    py *= r_rel.y;
                                }
                                px *= r_rel.x;
                            }

                            acc.x += term_m.x * scale;
                            acc.y += term_m.y * scale;
                            acc.z += term_m.z * scale;
                            scale *= inv_r2;
                        }
                    }

                    continue; /* skip the direct-sum fallback below */
                }
            }

            /* Fallback: direct particle summation over leaf particles. */
            int begin = node.particle_begin;
            int count = node.particle_count;
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
    }

    results[3u * tid] = acc.x;
    results[3u * tid + 1u] = acc.y;
    results[3u * tid + 2u] = acc.z;
}

#endif /* __OPENCL_C_VERSION__ */
