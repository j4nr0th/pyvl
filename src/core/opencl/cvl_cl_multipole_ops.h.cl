#pragma once
/*
 * Multipole operator kernels (P2M, M2P, M2M) for OpenCL C device compilation.
 *
 * All functions are `static inline` and are ONLY compiled in OpenCL C mode.
 * The C17 path uses the existing library in multipole.c.
 *
 * Depends on cvl_cl_multipole.h.cl for coefficient arithmetic helpers.
 */

#include "cvl_cl_multipole.h.cl"

#ifdef __OPENCL_C_VERSION__

/* multipole_t struct for device */
typedef struct {
    unsigned order;
    real3_t center;
    real_t *coeffs_x;
    real_t *coeffs_y;
    real_t *coeffs_z;
} multipole_t;

/* ------------------------------------------------------------------ */
/*  multipole_update  (P2M)                                            */
/* ------------------------------------------------------------------ */

/**
 * @brief Accumulate a single source point into a multipole expansion.
 *
 * Computes (2 r·pos - pos·pos)^m for each order m and adds the weighted
 * source value to the expansion coefficients.
 *
 * @param multipole    Target multipole expansion (accumulated in-place).
 * @param source_pos   Position of the source point (relative to origin).
 * @param source_value Vector value of the source point.
 * @param cur          Scratch buffer, at least multipole_scratch_size(order) elements.
 * @param nxt          Scratch buffer, at least multipole_scratch_size(order) elements.
 */
static inline void multipole_update(multipole_t *multipole, real3_t source_pos, real3_t source_value,
                                    real_t *cur, real_t *nxt)
{
    unsigned order = multipole->order;
    size_t scratch = multipole_scratch_size(order);
    unsigned dim = order + 1;
    size_t dim2 = (size_t)dim * dim;

    real3_t rel_pos = real3_sub(source_pos, multipole->center);
    real_t s2 = real3_dot(rel_pos, rel_pos);

    real_t *coeffs_x = multipole->coeffs_x;
    real_t *coeffs_y = multipole->coeffs_y;
    real_t *coeffs_z = multipole->coeffs_z;

    cur[0] = 1.0;
    size_t idx = 0;
    for (unsigned m = 0; m <= order; ++m)
    {
        for (unsigned p = 0; p <= m; ++p)
        {
            for (unsigned q = 0; q <= m - p; ++q)
            {
                unsigned r_max = m - p - q;
                for (unsigned r = 0; r <= r_max; ++r)
                {
                    real_t c = cur[p * dim2 + q * dim + r];
                    coeffs_x[idx + r] += source_value.x * c;
                    coeffs_y[idx + r] += source_value.y * c;
                    coeffs_z[idx + r] += source_value.z * c;
                }
                idx += (size_t)(r_max + 1);
            }
        }

        if (m == order) break;

        /* Zero nxt — manual loop instead of memset for OpenCL compat */
        for (size_t i = 0; i < scratch; ++i) nxt[i] = 0.0;

        for (unsigned p = 0; p <= m; ++p)
        {
            for (unsigned q = 0; q <= m - p; ++q)
            {
                for (unsigned r = 0; r <= m - p - q; ++r)
                {
                    size_t cur_idx = p * dim2 + q * dim + r;
                    real_t c = cur[cur_idx];
                    if (c == 0.0) continue;

                    nxt[cur_idx] -= s2 * c;
                    nxt[(p + 1) * dim2 + q * dim + r] += 2.0 * rel_pos.x * c;
                    nxt[p * dim2 + (q + 1) * dim + r] += 2.0 * rel_pos.y * c;
                    nxt[p * dim2 + q * dim + (r + 1)] += 2.0 * rel_pos.z * c;
                }
            }
        }
        /* Swap cur and nxt */
        real_t *tmp = cur; cur = nxt; nxt = tmp;
    }
}

/* ------------------------------------------------------------------ */
/*  multipole_eval  (M2P)                                              */
/* ------------------------------------------------------------------ */

/**
 * @brief Evaluate the multipole expansion at a given target point.
 *
 * @param multipole  Multipole expansion to evaluate.
 * @param point      Target point (absolute coordinates).
 * @return Vector value of the expansion at @p point.
 */
static inline real3_t multipole_eval(multipole_t *multipole, real3_t point)
{
    real3_t rel_point = real3_sub(point, multipole->center);
    real_t inv_r = 1.0 / sqrt(rel_point.x * rel_point.x + rel_point.y * rel_point.y + rel_point.z * rel_point.z);
    real_t inv_r2 = inv_r * inv_r;
    real_t scale = inv_r2;
    real3_t res = {0, 0, 0};
    size_t idx = 0;

    for (unsigned m = 0; m <= multipole->order; ++m)
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
                    term.x += multipole->coeffs_x[idx] * pz;
                    term.y += multipole->coeffs_y[idx] * pz;
                    term.z += multipole->coeffs_z[idx] * pz;
                    pz *= rel_point.z;
                    idx += 1;
                }
                py *= rel_point.y;
            }
            px *= rel_point.x;
        }
        res.x += term.x * scale;
        res.y += term.y * scale;
        res.z += term.z * scale;
        scale *= inv_r2;
    }
    return res;
}

/* ------------------------------------------------------------------ */
/*  multipole_add_shift  (M2M)                                        */
/* ------------------------------------------------------------------ */

/**
 * @brief Shift (translate) one multipole expansion and add into another.
 *
 * Implements M2M: shifts @p in so that its new centre is @p out->center,
 * then accumulates the result into @p out.
 *
 * @param in          Source multipole expansion (old centre).
 * @param out         Target multipole expansion (new centre, accumulated).
 * @param work_order  Internal expansion order for the shift series.
 * @param shift_exp   Work buffer for binomial expansions [3 * (work_order+1)^2].
 * @param pse         Work buffer for polynomial series expansion
 *                    [2 * multipole_num_coeffs(work_order)].
 */
static inline void multipole_add_shift(multipole_t *in, multipole_t *out, unsigned work_order,
                                       real_t *shift_exp, real_t *pse)
{
    unsigned in_order = in->order;
    unsigned out_order = out->order;

    real3_t shift = real3_sub(in->center, out->center);
    real_t s2 = real3_dot(shift, shift);

    real_t lx = 2.0 * shift.x;
    real_t ly = 2.0 * shift.y;
    real_t lz = 2.0 * shift.z;
    real_t lc = -s2;

    size_t shift_dim = (size_t)work_order + 1;
    size_t shift_plane = shift_dim * shift_dim;
    build_binomial_expansion(shift_exp, -shift.x, -shift.y, -shift.z, work_order, shift_dim, shift_plane);

    size_t n_coeffs = multipole_num_coeffs(work_order);

    for (unsigned m = 0; m <= in_order; ++m)
    {
        for (unsigned p = 0; p <= m; ++p)
        {
            for (unsigned q = 0; q <= m - p; ++q)
            {
                for (unsigned r = 0; r <= m - p - q; ++r)
                {
                    size_t in_idx = multipole_coeff_index(m, p, q, r);
                    real_t cx = in->coeffs_x[in_idx];
                    real_t cy = in->coeffs_y[in_idx];
                    real_t cz = in->coeffs_z[in_idx];
                    if (cx == 0.0 && cy == 0.0 && cz == 0.0) continue;

                    /* Numerator shift polynomial */
                    for (size_t i = 0; i < n_coeffs; ++i) pse[i] = 0.0;
                    for (unsigned i = 0; i <= p; ++i)
                        for (unsigned j = 0; j <= q; ++j)
                            for (unsigned k = 0; k <= r; ++k)
                            {
                                real_t factor = shift_exp[0 * shift_plane + p * shift_dim + i] *
                                                shift_exp[1 * shift_plane + q * shift_dim + j] *
                                                shift_exp[2 * shift_plane + r * shift_dim + k];
                                size_t pse_idx = multipole_coeff_index(i + j + k, i, j, k);
                                pse[pse_idx] = factor;
                            }

                    /* l = 0 */
                    if (m <= out_order)
                        multipole_add_poly_to_order(pse, m, 1.0, cx, cy, cz,
                            out->coeffs_x, out->coeffs_y, out->coeffs_z);

                    /* l >= 1: binomial series */
                    unsigned cur = 0, nxt = 1;
                    real_t binom = 1.0;
                    unsigned max_l = (work_order > m) ? work_order - m : 0;
                    for (unsigned l = 1; l <= max_l; ++l)
                    {
                        multipole_poly_mul_linear(pse + cur * n_coeffs, pse + nxt * n_coeffs,
                            lx, ly, lz, lc, work_order);
                        binom = binom * (real_t)(m + l) / (real_t)l;
                        if (m + l <= out_order)
                            multipole_add_poly_to_order(pse + nxt * n_coeffs, m + l, binom, cx, cy, cz,
                                out->coeffs_x, out->coeffs_y, out->coeffs_z);
                        unsigned tmp2 = cur; cur = nxt; nxt = tmp2;
                    }
                }
            }
        }
    }
}

#endif /* __OPENCL_C_VERSION__ */
