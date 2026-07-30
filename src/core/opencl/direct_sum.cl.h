#pragma once
/*
 * Direct N-body sum kernel for OpenCL.
 *
 * Dual-compiled header: compiles in C17 (host) and OpenCL C (device).
 * Each work-item sums contributions from N sources for one target.
 *
 * real_t is supplied by cvl_cl_types.h.cl (or equivalent) when compiled
 * as OpenCL C.  On the host side this header is empty.
 *
 * Phase 3 — simplest GPU offload: O(N*M) work, no tree, no approximation.
 */

#include "cvl_cl_types.h.cl"

#ifdef __OPENCL_C_VERSION__

/* ------------------------------------------------------------------ */
/* Kernel-local 3D vector type                                        */
/* ------------------------------------------------------------------ */

typedef struct
{
    real_t x, y, z;
} ds_real3_t;

/* ------------------------------------------------------------------ */
/* Kernel: direct_sum                                                  */
/*                                                                     */
/* For each target point, accumulates 1/|r|^2 contributions from all   */
/* source particles.  Uses a small softening radius (~1e-30) to avoid  */
/* division by zero when target == source.                             */
/* ------------------------------------------------------------------ */

/**
 * @brief Direct N-body evaluation kernel.
 *
 * @param[in]  targets      Target positions  [3 * n_targets]
 * @param[in]  sources_pos  Source positions   [3 * n_sources]
 * @param[in]  sources_val  Source strengths   [3 * n_sources]
 * @param[in]  n_sources    Number of sources
 * @param[in]  n_targets    Number of targets
 * @param[out] results      Accumulated field  [3 * n_targets]
 */
__kernel void direct_sum(__global const real_t *targets, __global const real_t *sources_pos,
                         __global const real_t *sources_val, unsigned n_sources, unsigned n_targets,
                         __global real_t *results)
{
    unsigned tid = get_global_id(0);
    if (tid >= n_targets)
        return;

    /* Load target point. */
    ds_real3_t pt;
    pt.x = targets[3 * tid];
    pt.y = targets[3 * tid + 1];
    pt.z = targets[3 * tid + 2];

    ds_real3_t acc = {0, 0, 0};

    for (unsigned i = 0; i < n_sources; ++i)
    {
        ds_real3_t dr;
        dr.x = pt.x - sources_pos[3 * i];
        dr.y = pt.y - sources_pos[3 * i + 1];
        dr.z = pt.z - sources_pos[3 * i + 2];

        real_t r2 = dr.x * dr.x + dr.y * dr.y + dr.z * dr.z;

        /* Softening — avoid division by zero for coincident points. */
        if (r2 > (real_t)1e-30)
        {
            real_t inv_r2 = (real_t)1.0 / r2;
            acc.x += sources_val[3 * i] * inv_r2;
            acc.y += sources_val[3 * i + 1] * inv_r2;
            acc.z += sources_val[3 * i + 2] * inv_r2;
        }
    }

    results[3 * tid] = acc.x;
    results[3 * tid + 1] = acc.y;
    results[3 * tid + 2] = acc.z;
}

#endif /* __OPENCL_C_VERSION__ */
