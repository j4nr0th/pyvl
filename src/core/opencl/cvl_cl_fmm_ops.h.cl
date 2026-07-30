#pragma once
/*
 * FMM operator kernels (M2L, L2L, L2P, P2L) for OpenCL C device compilation.
 *
 * All functions are `static inline` and are ONLY compiled in OpenCL C mode.
 * The C17 path uses the operators in fmm_operators.c directly.
 *
 * Depends on cvl_cl_multipole_ops.h.cl for the multipole polynomial
 * arithmetic helpers.
 */

#include "cvl_cl_multipole_ops.h.cl"

#ifdef __OPENCL_C_VERSION__

/* ------------------------------------------------------------------ */
/*  local_expansion_t — device-side struct                            */
/* ------------------------------------------------------------------ */

/**
 * @brief Device-side representation of a local (Taylor-like) expansion.
 *
 * Dual of multipole_t.  Coefficients use the same tetrahedral layout
 * as the multipole expansion (multipole_coeff_index).
 *
 * Plain pointers (no restrict) for OpenCL C compatibility.
 */
typedef struct {
    unsigned order;     /**< Expansion order. */
    real3_t center;     /**< Centre of the local expansion. */
    real_t *coeffs_x;   /**< x-component coefficients. */
    real_t *coeffs_y;   /**< y-component coefficients. */
    real_t *coeffs_z;   /**< z-component coefficients. */
} local_expansion_t;

/* ------------------------------------------------------------------ */
/*  local_add_poly_to_order — helper                                   */
/* ------------------------------------------------------------------ */

/**
 * @brief Accumulate a scaled polynomial into a local expansion.
 *
 * Thin wrapper that calls @ref multipole_add_poly_to_order with the
 * local expansion's coefficient arrays directly (the OpenCL C version
 * takes separate arrays rather than a multipole_t struct).
 *
 * @param poly      Polynomial coefficients (dense tetrahedral layout).
 * @param out_order Maximum order to accumulate (orders 0..out_order).
 * @param scale     Scalar multiplier.
 * @param cx        x-component of the vector multiplier.
 * @param cy        y-component of the vector multiplier.
 * @param cz        z-component of the vector multiplier.
 * @param out       Target local expansion (accumulated in-place).
 */
static inline void local_add_poly_to_order(real_t *poly, unsigned out_order, real_t scale,
    real_t cx, real_t cy, real_t cz, local_expansion_t *out)
{
    multipole_add_poly_to_order(poly, out_order, scale, cx, cy, cz,
        out->coeffs_x, out->coeffs_y, out->coeffs_z);
}

/* ------------------------------------------------------------------ */
/*  multipole_to_local  (M2L)                                          */
/* ------------------------------------------------------------------ */

/**
 * @brief Convert a multipole expansion into a local expansion (M2L).
 *
 * Given a multipole expansion of a source cluster centred at
 * in->center, produce the equivalent local expansion centred at
 * out->center and accumulate it into @p out.
 *
 * The denominator series uses the quadratic form (2 R'·r' + r'^2)
 * where R' = R - S (local centre minus source centre).
 *
 * @param in         Source multipole expansion (read-only).
 * @param out        Target local expansion (accumulated, not zeroed).
 * @param work_order Internal expansion order for the series
 *                   (>= max(in->order, out->order)).
 * @param shift_exp  Scratch buffer for binomial expansions
 *                   [3 * (work_order+1)^2].
 * @param pse        Double-buffered polynomial scratch buffer
 *                   [2 * multipole_num_coeffs(work_order)].
 */
static inline void multipole_to_local(multipole_t *in, local_expansion_t *out, unsigned work_order,
                                      real_t *shift_exp, real_t *pse)
{
    unsigned in_order = in->order;
    unsigned out_order = out->order;

    real3_t R_prime = real3_sub(out->center, in->center);
    real_t Rp2 = real3_dot(R_prime, R_prime);
    if (Rp2 < 1e-30) return;

    real_t qlx = 2.0 * R_prime.x;
    real_t qly = 2.0 * R_prime.y;
    real_t qlz = 2.0 * R_prime.z;
    real_t qx = 1.0, qy = 1.0, qz = 1.0, qc = 0.0;

    size_t shift_dim = (size_t)work_order + 1;
    size_t shift_plane = shift_dim * shift_dim;
    build_binomial_expansion(shift_exp, R_prime.x, R_prime.y, R_prime.z, work_order, shift_dim, shift_plane);

    size_t n_coeffs = multipole_num_coeffs(work_order);
    real_t inv_Rp2 = 1.0 / Rp2;

    /* Precompute inv_Rp2^k for k = 0..work_order+1 */
    enum { INV_MAX = 256 };
    real_t invRp2_pow[INV_MAX];
    invRp2_pow[0] = 1.0;
    for (unsigned k = 1; k <= work_order + 1; ++k)
        invRp2_pow[k] = invRp2_pow[k - 1] * inv_Rp2;

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

                    /* Numerator: (x + R'_x)^p (y + R'_y)^q (z + R'_z)^r */
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
                    {
                        real_t scale = invRp2_pow[m + 1];
                        local_add_poly_to_order(pse, m, scale, cx, cy, cz, out);
                    }

                    /* l >= 1: denominator binomial series with quadratic form */
                    unsigned cur = 0, nxt = 1;
                    real_t binom = 1.0;
                    unsigned max_l = (work_order > m) ? work_order - m : 0;
                    for (unsigned l = 1; l <= max_l; ++l)
                    {
                        multipole_poly_mul_quadratic(pse + cur * n_coeffs, pse + nxt * n_coeffs,
                            qx, qy, qz, qlx, qly, qlz, qc, work_order);
                        binom = binom * (real_t)(m + l) / (real_t)l;
                        if (m + l <= out_order)
                        {
                            real_t scale = invRp2_pow[m + l + 1];
                            if (l & 1u) scale = -scale;
                            local_add_poly_to_order(pse + nxt * n_coeffs, m + l, scale * binom, cx, cy, cz, out);
                        }
                        unsigned tmp2 = cur; cur = nxt; nxt = tmp2;
                    }
                }
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/*  local_expansion_shift  (L2L)                                       */
/* ------------------------------------------------------------------ */

/**
 * @brief Shift (translate) a local expansion to a new centre (L2L).
 *
 * Translates a local expansion from in->center to out->center and
 * accumulates the result into @p out.
 *
 * The shift is a pure polynomial substitution r'_in = r'_out + d
 * where d = out->center - in->center.  No series expansion is needed
 * because the local expansion is a regular polynomial.
 *
 * @param in         Source local expansion (read-only).
 * @param out        Target local expansion (accumulated, not zeroed).
 * @param work_order Internal expansion order for the shift
 *                   (>= max(in->order, out->order)).
 * @param shift_exp  Scratch buffer for binomial expansions
 *                   [3 * (work_order+1)^2].
 * @param pse        Polynomial scratch buffer
 *                   [multipole_num_coeffs(work_order)].
 */
static inline void local_expansion_shift(local_expansion_t *in, local_expansion_t *out, unsigned work_order,
                                         real_t *shift_exp, real_t *pse)
{
    unsigned in_order = in->order;
    unsigned out_order = out->order;

    real3_t d = real3_sub(out->center, in->center);
    real_t d2 = real3_dot(d, d);
    if (d2 < 1e-30) return;

    size_t shift_dim = (size_t)work_order + 1;
    size_t shift_plane = shift_dim * shift_dim;
    build_binomial_expansion(shift_exp, d.x, d.y, d.z, work_order, shift_dim, shift_plane);

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

                    if (m <= out_order)
                        local_add_poly_to_order(pse, m, 1.0, cx, cy, cz, out);
                }
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/*  local_expansion_eval  (L2P)                                        */
/* ------------------------------------------------------------------ */

/**
 * @brief Evaluate a local expansion at a given target point (L2P).
 *
 * The local expansion is a regular polynomial (no 1/r^n scaling),
 * so this is a straight Horner-style evaluation.
 *
 * @param local  Local expansion to evaluate.
 * @param point  Target point (absolute coordinates).
 * @return Vector value of the expansion at @p point.
 */
static inline real3_t local_expansion_eval(local_expansion_t *loc_exp, real3_t point)
{
    real3_t rel_point = real3_sub(point, loc_exp->center);
    real3_t res = {0, 0, 0};
    size_t idx = 0;

    for (unsigned m = 0; m <= loc_exp->order; ++m)
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
                    term.x += loc_exp->coeffs_x[idx] * pz;
                    term.y += loc_exp->coeffs_y[idx] * pz;
                    term.z += loc_exp->coeffs_z[idx] * pz;
                    pz *= rel_point.z;
                    idx += 1;
                }
                py *= rel_point.y;
            }
            px *= rel_point.x;
        }
        res.x += term.x;
        res.y += term.y;
        res.z += term.z;
    }
    return res;
}

/* ------------------------------------------------------------------ */
/*  particle_to_local  (P2L)                                           */
/* ------------------------------------------------------------------ */

/**
 * @brief Accumulate a single source particle into a local expansion (P2L).
 *
 * Builds the polynomial P_m(r') = (2 R'·r' + r'^2)^m iteratively and
 * accumulates source_value * P_m * inv_Rp2^{m+1} into the local expansion
 * coefficients.  This is the dual of @ref multipole_update (P2M).
 *
 * @param out          Target local expansion (accumulated in-place).
 * @param source_pos   Position of the source particle.
 * @param source_value Vector value of the source particle.
 * @param cur          Scratch buffer, at least multipole_scratch_size(order) elements.
 * @param nxt          Scratch buffer, at least multipole_scratch_size(order) elements.
 */
static inline void particle_to_local(local_expansion_t *out, real3_t source_pos, real3_t source_value,
                                     real_t *cur, real_t *nxt)
{
    unsigned order = out->order;
    unsigned dim = order + 1;
    size_t dim2 = (size_t)dim * dim;
    size_t scratch = multipole_scratch_size(order);

    real3_t R_prime = real3_sub(out->center, source_pos);
    real_t Rp2 = real3_dot(R_prime, R_prime);
    if (Rp2 < 1e-30) return;

    real_t inv_Rp2 = 1.0 / Rp2;

    /* Precompute inv_Rp2^k for k = 0..order+1 */
    enum { P2L_MAX = 256 };
    real_t invRp2_pow[P2L_MAX];
    invRp2_pow[0] = 1.0;
    for (unsigned k = 1; k <= order + 1; ++k)
        invRp2_pow[k] = invRp2_pow[k - 1] * inv_Rp2;

    real_t *coeffs_x = out->coeffs_x;
    real_t *coeffs_y = out->coeffs_y;
    real_t *coeffs_z = out->coeffs_z;

    cur[0] = 1.0;
    size_t idx = 0;
    for (unsigned m = 0; m <= order; ++m)
    {
        real_t scale = invRp2_pow[m + 1];
        if (m & 1u) scale = -scale;

        for (unsigned p = 0; p <= m; ++p)
        {
            for (unsigned q = 0; q <= m - p; ++q)
            {
                unsigned r_max = m - p - q;
                for (unsigned r = 0; r <= r_max; ++r)
                {
                    real_t c = cur[p * dim2 + q * dim + r];
                    coeffs_x[idx + r] += source_value.x * c * scale;
                    coeffs_y[idx + r] += source_value.y * c * scale;
                    coeffs_z[idx + r] += source_value.z * c * scale;
                }
                idx += (size_t)(r_max + 1);
            }
        }

        if (m == order) break;

        /* nxt = (2 R'·r' + r'^2) * cur */
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

                    nxt[(p + 1) * dim2 + q * dim + r] += 2.0 * R_prime.x * c;
                    nxt[p * dim2 + (q + 1) * dim + r] += 2.0 * R_prime.y * c;
                    nxt[p * dim2 + q * dim + (r + 1)] += 2.0 * R_prime.z * c;

                    if (p + 2 <= order)
                        nxt[(p + 2) * dim2 + q * dim + r] += c;
                    if (q + 2 <= order)
                        nxt[p * dim2 + (q + 2) * dim + r] += c;
                    if (r + 2 <= order)
                        nxt[p * dim2 + q * dim + (r + 2)] += c;
                }
            }
        }
        {
            real_t *tmp = cur; cur = nxt; nxt = tmp;
        }
    }
}

#endif /* __OPENCL_C_VERSION__ */
