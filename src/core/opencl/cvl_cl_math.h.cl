#pragma once
/*
 * Shared math helpers for C17 / OpenCL C dual-compilation kernel headers.
 *
 * OpenCL C path: defines all functions as static inline for GPU use.
 * C17 path:     functions are provided by common.h / multipole.h, so
 *               this header only includes the type header.
 */

#include "cvl_cl_types.h.cl"

#ifdef __OPENCL_C_VERSION__

/* ------------------------------------------------------------------ */
/*  Particle kernel: gamma / |r|^2                                     */
/* ------------------------------------------------------------------ */

static inline real3_t particle_kernel(real3_t gamma, real3_t r_vec)
{
    real_t r2 = real3_dot(r_vec, r_vec);
    if (r2 < 1e-30)
        return (real3_t){0, 0, 0};
    return real3_mul1(gamma, 1.0 / r2);
}

/* ------------------------------------------------------------------ */
/*  Morton (Z-order) 3D helpers                                        */
/* ------------------------------------------------------------------ */

static inline uint64_t morton_split_21(uint64_t x)
{
    x &= 0x1fffffULL;
    x = (x | (x << 32)) & 0x1f00000000ffffULL;
    x = (x | (x << 16)) & 0x1f0000ff0000ffULL;
    x = (x | (x << 8))  & 0x100f00f00f00f00fULL;
    x = (x | (x << 4))  & 0x10c30c30c30c30c3ULL;
    x = (x | (x << 2))  & 0x1249249249249249ULL;
    return x;
}

static inline uint64_t morton_3d(real3_t p, real3_t root_center, real_t root_half_size)
{
    real_t inv_cell = 1.0 / (2.0 * root_half_size);
    real_t scale    = (real_t)((1u << 21) - 1);
    real_t nx = (p.x - root_center.x) * inv_cell + 0.5;
    real_t ny = (p.y - root_center.y) * inv_cell + 0.5;
    real_t nz = (p.z - root_center.z) * inv_cell + 0.5;
    if (nx < 0) nx = 0;
    if (nx >= 1) nx = 0.999999;
    if (ny < 0) ny = 0;
    if (ny >= 1) ny = 0.999999;
    if (nz < 0) nz = 0;
    if (nz >= 1) nz = 0.999999;
    uint64_t ix = (uint64_t)(nx * scale);
    uint64_t iy = (uint64_t)(ny * scale);
    uint64_t iz = (uint64_t)(nz * scale);
    return morton_split_21(ix) | (morton_split_21(iy) << 1) | (morton_split_21(iz) << 2);
}

/* ------------------------------------------------------------------ */
/*  Multipole coefficient arithmetic (pure integer)                     */
/* ------------------------------------------------------------------ */

static inline size_t multipole_num_coeffs(unsigned order)
{
    return (size_t)(order + 1) * (order + 2) * (order + 3) * (order + 4) / 24;
}

static inline size_t multipole_scratch_size(unsigned order)
{
    size_t dim = (size_t)order + 1;
    return dim * dim * dim;
}

static inline size_t multipole_coeff_index(unsigned m, unsigned p, unsigned q, unsigned r)
{
    size_t idx = (size_t)m * (m + 1) * (m + 2) * (m + 3) / 24;
    for (unsigned pp = 0; pp < p; ++pp)
        idx += (size_t)(m - pp + 2) * (m - pp + 1) / 2;
    idx += (size_t)q * (m - p + 1) - (size_t)q * (q - 1) / 2;
    idx += r;
    return idx;
}

#endif /* __OPENCL_C_VERSION__ */
