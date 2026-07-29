#include "fmm_operators.h"

#include <assert.h>
#include <math.h>
#include <string.h>

size_t local_expansion_num_coeffs(unsigned order)
{
    /* Same tetrahedral layout as the multipole expansion. */
    return multipole_num_coeffs(order);
}

size_t local_expansion_shift_exp_size(unsigned work_order)
{
    const size_t dim = (size_t)work_order + 1;
    return 3u * dim * dim;
}

size_t local_expansion_pse_size(unsigned work_order)
{
    return 2u * multipole_num_coeffs(work_order);
}

size_t local_expansion_m2l_scratch_size(unsigned work_order)
{
    const size_t pse = local_expansion_pse_size(work_order);
    const size_t shift_exp = local_expansion_shift_exp_size(work_order);
    return pse > shift_exp ? pse : shift_exp;
}

/* ------------------------------------------------------------------ */
/* Internal: accumulate a polynomial into a local expansion.          */
/* ------------------------------------------------------------------ */

/**
 * @brief Thin wrapper around @ref multipole_add_poly_to_order for
 *        local expansion targets.
 *
 * Creates a temporary multipole_t view over the local expansion's
 * coefficient arrays so the shared polynomial helper can be reused.
 */
static inline void local_add_poly_to_order(const real_t *poly, unsigned out_order, real_t scale, const real_t cx,
                                           const real_t cy, const real_t cz, const local_expansion_t *out)
{
    const multipole_t mp_view = {
        .coeffs_x = out->coeffs_x,
        .coeffs_y = out->coeffs_y,
        .coeffs_z = out->coeffs_z,
    };
    multipole_add_poly_to_order(poly, out_order, scale, cx, cy, cz, &mp_view);
}

/* ------------------------------------------------------------------ */
/* M2L: multipole_to_local                                            */
/* ------------------------------------------------------------------ */

void multipole_to_local(const multipole_t *in, local_expansion_t *out, unsigned work_order,
                        real_t CVL_ARRAY_ARG(shift_exp, restrict), real_t CVL_ARRAY_ARG(pse, restrict))
{
    const unsigned in_order = in->order;
    const unsigned out_order = out->order;

    /* R' = R - S  (local centre minus source centre). */
    const real3_t R_prime = real3_sub(out->center, in->center);
    const real_t Rp2 = real3_dot(R_prime, R_prime);

    if (Rp2 < 1e-30)
        return; /* Source and local centres coincide — nothing to do. */

    /*
     * The local expansion polynomial for order m is:
     *
     *   P_m(r') = (2 R'·r' + r'^2)^m
     *
     * The full field is:
     *
     *   V(r) = (1/|R-S|^2) * sum_{m=0}^{order} (1/|R-S|^{2m}) * P_m(r')
     *
     * The multipole stores coefficients c_{pqr}^{(m)} for the polynomial
     * (2 r·s - s^2)^m.  To convert, we re-express each source monomial
     * in terms of the local variable r' using the identity:
     *
     *   s = r - R'   (since r = R + r', s = S + s_rel, R' = R - S,
     *                 so s_rel = r - R = r')
     *
     * Wait — the multipole is expanded about S (source centre), so its
     * variable is r - S.  The local is expanded about R, variable r' = r - R.
     * Thus r - S = r' + (R - S) = r' + R'.
     *
     * The multipole coefficient c_{pqr}^{(m)} multiplies (r-S)^p (r-S)^q ...
     * = (r' + R')^p (r' + R')^q ...  We substitute r' + R' for each factor
     * using binomial expansion, then the denominator series introduces
     * powers of (2 R'·r' + r'^2).
     *
     * This is structurally identical to multipole_add_shift but with a
     * QUADRATIC denominator factor (2 R'·r' + r'^2) instead of the linear
     * (2 r·d - d^2).
     */

    /* Quadratic form coefficients for (2 R'·r' + r'^2):
     *   = 2*R'_x * x + 2*R'_y * y + 2*R'_z * z + x^2 + y^2 + z^2
     * So: qc = 0, qlx = 2*R'_x, qly = 2*R'_y, qlz = 2*R'_z,
     *     qx = 1, qy = 1, qz = 1. */
    const real_t qlx = 2.0 * R_prime.x;
    const real_t qly = 2.0 * R_prime.y;
    const real_t qlz = 2.0 * R_prime.z;
    const real_t qx = 1.0;
    const real_t qy = 1.0;
    const real_t qz = 1.0;
    const real_t qc = 0.0;

    /* Build binomial expansions of (x + R'_x)^e etc up to work_order.
     * These re-express source-centred monomials in local-centred coords. */
    const size_t shift_dim = (size_t)work_order + 1;
    const size_t shift_plane = shift_dim * shift_dim;
    build_binomial_expansion(shift_exp, R_prime.x, R_prime.y, R_prime.z, work_order, shift_dim, shift_plane);

    const size_t n_coeffs = multipole_num_coeffs(work_order);
    const real_t inv_Rp2 = 1.0 / Rp2;

    /* Precompute inv_Rp2^k for k = 0..work_order+1. */
    enum
    {
        INV_RP2_POW_MAX = 256
    };
    real_t invRp2_pow[INV_RP2_POW_MAX];
    assert(work_order + 2 <= INV_RP2_POW_MAX);
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
                    const size_t in_idx = multipole_coeff_index(m, p, q, r);
                    const real_t cx = in->coeffs_x[in_idx];
                    const real_t cy = in->coeffs_y[in_idx];
                    const real_t cz = in->coeffs_z[in_idx];
                    if (cx == 0.0 && cy == 0.0 && cz == 0.0)
                        continue;

                    /* Numerator: (x + R'_x)^p (y + R'_y)^q (z + R'_z)^r
                     * expressed in local coords r' = (x,y,z). */
                    for (size_t i = 0; i < n_coeffs; ++i)
                    {
                        pse[i] = 0.0;
                    }
                    for (unsigned i = 0; i <= p; ++i)
                    {
                        for (unsigned j = 0; j <= q; ++j)
                        {
                            for (unsigned k = 0; k <= r; ++k)
                            {
                                const real_t factor = shift_exp[0 * shift_plane + p * shift_dim + i] *
                                                      shift_exp[1 * shift_plane + q * shift_dim + j] *
                                                      shift_exp[2 * shift_plane + r * shift_dim + k];
                                const size_t idx = multipole_coeff_index(i + j + k, i, j, k);
                                pse[idx] = factor;
                            }
                        }
                    }

                    /* l = 0: denominator factor = 1.
                     * Scale = inv_Rp2^{m+1}. */
                    if (m <= out_order)
                    {
                        const real_t scale = invRp2_pow[m + 1];
                        local_add_poly_to_order(pse, m, scale, cx, cy, cz, out);
                    }

                    /* l >= 1: denominator binomial series
                     * (2 R'·r' + r'^2)^l with binomial coefficient C(m+l, l)
                     * and alternating sign (-1)^l from 1/(1+u) = sum (-1)^l u^l.
                     * Each step multiplies the running polynomial by the
                     * quadratic form. */
                    unsigned cur = 0;
                    unsigned nxt = 1;
                    real_t binom = 1.0;
                    const unsigned max_l = (work_order > m) ? work_order - m : 0;
                    for (unsigned l = 1; l <= max_l; ++l)
                    {
                        multipole_poly_mul_quadratic(pse + cur * n_coeffs, pse + nxt * n_coeffs, qx, qy, qz, qlx, qly,
                                                     qlz, qc, work_order);
                        binom = binom * (m + l) / (real_t)l;
                        if (m + l <= out_order)
                        {
                            real_t scale = invRp2_pow[m + l + 1];
                            /* (-1)^l alternating sign from 1/(1+u) series. */
                            if (l & 1u)
                                scale = -scale;
                            local_add_poly_to_order(pse + nxt * n_coeffs, m + l, scale * binom, cx, cy, cz, out);
                        }
                        const unsigned tmp = cur;
                        cur = nxt;
                        nxt = tmp;
                    }
                }
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/* L2L: local_expansion_shift                                         */
/* ------------------------------------------------------------------ */

void local_expansion_shift(const local_expansion_t *in, local_expansion_t *out, unsigned work_order,
                           real_t CVL_ARRAY_ARG(shift_exp, restrict), real_t CVL_ARRAY_ARG(pse, restrict))
{
    const unsigned in_order = in->order;
    const unsigned out_order = out->order;

    /* d = out.center - in.center (shift from old centre to new centre). */
    const real3_t d = real3_sub(out->center, in->center);
    const real_t d2 = real3_dot(d, d);

    if (d2 < 1e-30)
        return; /* Centres coincide — copy coefficients directly. */

    /*
     * Shifting a local expansion from centre C_in to C_out:
     *
     *   r'_in  = r - C_in
     *   r'_out = r - C_out = r'_in - d
     *
     * So r'_in = r'_out + d.  Each monomial (r'_in)^p = (r'_out + d)^p,
     * re-expressed via binomial expansion (same as M2M numerator shift).
     *
     * The denominator factor for the local series is
     *   (2 R'·r' + r'^2)
     * where R' = R - S is FIXED (determined by source/local geometry).
     * When we shift the local centre, R' changes:
     *   R'_out = R_out - S = (R_in + d) - S = R'_in + d
     *
     * So the new denominator factor becomes:
     *   2 (R'_in + d)·r'_out + r'_out^2
     *   = (2 R'_in·r'_out + r'_out^2) + 2 d·r'_out
     *   = Q(r'_out) + L_d(r'_out)
     *
     * where Q is the original quadratic and L_d = 2 d·r' is linear.
     *
     * This is more complex than M2M (which has a purely linear denominator).
     * However, there's a simpler approach: the local expansion is just a
     * polynomial in r'.  Shifting the centre means substituting
     * r'_in = r'_out + d into the polynomial.  This is a pure polynomial
     * substitution — no series expansion needed!
     *
     * For each stored coefficient c_{pqr} at order m, the monomial
     * x^p y^q z^r (in old coords) becomes (x+d_x)^p (y+d_y)^q (z+d_z)^r
     * (in new coords).  We expand via binomial and accumulate.
     *
     * The scale factor 1/|R-S|^{2(m+1)} is absorbed into the coefficients
     * during M2L, so it's already part of c_{pqr} — no extra scaling here.
     */

    /* Build binomial expansions of (x + d_x)^e etc up to work_order. */
    const size_t shift_dim = (size_t)work_order + 1;
    const size_t shift_plane = shift_dim * shift_dim;
    build_binomial_expansion(shift_exp, d.x, d.y, d.z, work_order, shift_dim, shift_plane);

    const size_t n_coeffs = multipole_num_coeffs(work_order);

    for (unsigned m = 0; m <= in_order; ++m)
    {
        for (unsigned p = 0; p <= m; ++p)
        {
            for (unsigned q = 0; q <= m - p; ++q)
            {
                for (unsigned r = 0; r <= m - p - q; ++r)
                {
                    const size_t in_idx = multipole_coeff_index(m, p, q, r);
                    const real_t cx = in->coeffs_x[in_idx];
                    const real_t cy = in->coeffs_y[in_idx];
                    const real_t cz = in->coeffs_z[in_idx];
                    if (cx == 0.0 && cy == 0.0 && cz == 0.0)
                        continue;

                    /* Expand (x+d_x)^p (y+d_y)^q (z+d_z)^r into new coords. */
                    for (size_t i = 0; i < n_coeffs; ++i)
                    {
                        pse[i] = 0.0;
                    }
                    for (unsigned i = 0; i <= p; ++i)
                    {
                        for (unsigned j = 0; j <= q; ++j)
                        {
                            for (unsigned k = 0; k <= r; ++k)
                            {
                                const real_t factor = shift_exp[0 * shift_plane + p * shift_dim + i] *
                                                      shift_exp[1 * shift_plane + q * shift_dim + j] *
                                                      shift_exp[2 * shift_plane + r * shift_dim + k];
                                const size_t idx = multipole_coeff_index(i + j + k, i, j, k);
                                pse[idx] = factor;
                            }
                        }
                    }

                    /* Accumulate into out at the same order m (no series). */
                    if (m <= out_order)
                    {
                        local_add_poly_to_order(pse, m, 1.0, cx, cy, cz, out);
                    }
                }
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/* L2P: local_expansion_eval                                          */
/* ------------------------------------------------------------------ */

real3_t local_expansion_eval(const local_expansion_t *local, const real3_t point)
{
    const real3_t rel_point = real3_sub(point, local->center);
    real3_t res = {.x = 0, .y = 0, .z = 0};
    size_t idx = 0;

    /*
     * V(r) = sum_{m=0}^{order} P_m(r')
     *
     * where P_m is the stored polynomial of order m.  The 1/|R-S|^{2(m+1)}
     * scaling is already baked into the coefficients during M2L/P2L.
     *
     * This uses the same Horner-style nested evaluation as multipole_eval,
     * but without the inv_r^2 scaling (the local expansion is a regular
     * polynomial, not a Laurent series).
     */
    for (unsigned m = 0; m <= local->order; ++m)
    {
        real3_t term = {.x = 0, .y = 0, .z = 0};

        real_t px = 1.0;
        for (unsigned p = 0; p <= m; ++p)
        {
            real_t py = px;
            for (unsigned q = 0; q <= m - p; ++q)
            {
                real_t pz = py;
                for (unsigned r = 0; r <= m - p - q; ++r)
                {
                    term.x += local->coeffs_x[idx] * pz;
                    term.y += local->coeffs_y[idx] * pz;
                    term.z += local->coeffs_z[idx] * pz;

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
/* P2L: particle_to_local                                             */
/* ------------------------------------------------------------------ */

void particle_to_local(local_expansion_t *out, const real3_t source_pos, const real3_t source_value,
                       real_t CVL_ARRAY_ARG(cur, restrict), real_t CVL_ARRAY_ARG(nxt, restrict))
{
    const unsigned order = out->order;
    const unsigned dim = order + 1;
    const size_t dim2 = (size_t)dim * dim;
    const size_t scratch = multipole_scratch_size(order);

    /* R' = R - S (local centre minus source position). */
    const real3_t R_prime = real3_sub(out->center, source_pos);
    const real_t Rp2 = real3_dot(R_prime, R_prime);

    if (Rp2 < 1e-30)
        return;

    const real_t inv_Rp2 = 1.0 / Rp2;

    /* Precompute inv_Rp2^k for k = 0..order+1. */
    enum
    {
        P2L_INV_RP2_POW_MAX = 256
    };
    real_t invRp2_pow[P2L_INV_RP2_POW_MAX];
    assert(order + 2 <= P2L_INV_RP2_POW_MAX);
    invRp2_pow[0] = 1.0;
    for (unsigned k = 1; k <= order + 1; ++k)
        invRp2_pow[k] = invRp2_pow[k - 1] * inv_Rp2;

    real_t *restrict const coeffs_x = out->coeffs_x;
    real_t *restrict const coeffs_y = out->coeffs_y;
    real_t *restrict const coeffs_z = out->coeffs_z;

    /*
     * Build the polynomial P_m(r') = (2 R'·r' + r'^2)^m iteratively,
     * accumulating source_value * P_m * inv_Rp2^{m+1} into the coefficients.
     *
     * This mirrors multipole_update but:
     *   - The polynomial variable is r' (target relative to local centre).
     *   - The generator is (2 R'·r' + r'^2) instead of (2 r·s - s^2).
     *   - Each order m gets an extra inv_Rp2^{m+1} scale factor.
     *
     * cur holds P_m as a dense (order+1)^3 array; nxt is the write target.
     */
    cur[0] = 1.0; /* P_0 = 1 */
    size_t idx = 0;
    for (unsigned m = 0; m <= order; ++m)
    {
        /* Scale for this order: inv_Rp2^{m+1}. */
        real_t scale = invRp2_pow[m + 1];

        /* Accumulate P_m into the coefficient arrays, with (-1)^m
         * alternating sign from the 1/(1+u) local series expansion. */
        if (m & 1u)
            scale = -scale;
        for (unsigned p = 0; p <= m; ++p)
        {
            for (unsigned q = 0; q <= m - p; ++q)
            {
                const unsigned r_max = m - p - q;
                for (unsigned r = 0; r <= r_max; ++r)
                {
                    const real_t c = cur[p * dim2 + q * dim + r];
                    coeffs_x[idx + r] += source_value.x * c * scale;
                    coeffs_y[idx + r] += source_value.y * c * scale;
                    coeffs_z[idx + r] += source_value.z * c * scale;
                }
                idx += (size_t)(r_max + 1);
            }
        }

        if (m == order)
            break;

        /* nxt = (2 R'·r' + r'^2) * cur
         *   = r'^2 * cur + 2 R'_x * x * cur + 2 R'_y * y * cur + 2 R'_z * z * cur
         *
         * r'^2 = x^2 + y^2 + z^2, so the r'^2 term contributes to (p+2,q,r),
         * (p,q+2,r), (p,q,r+2).  The linear terms contribute to (p+1,q,r) etc. */
        memset(nxt, 0, scratch * sizeof(real_t));
        for (unsigned p = 0; p <= m; ++p)
        {
            for (unsigned q = 0; q <= m - p; ++q)
            {
                for (unsigned r = 0; r <= m - p - q; ++r)
                {
                    const size_t cur_idx = p * dim2 + q * dim + r;
                    const real_t c = cur[cur_idx];
                    if (c == 0.0)
                        continue;

                    /* Linear terms: 2 R'·r' * cur. */
                    nxt[(p + 1) * dim2 + q * dim + r] += 2.0 * R_prime.x * c;
                    nxt[p * dim2 + (q + 1) * dim + r] += 2.0 * R_prime.y * c;
                    nxt[p * dim2 + q * dim + (r + 1)] += 2.0 * R_prime.z * c;

                    /* Quadratic terms: r'^2 * cur = (x^2+y^2+z^2) * cur. */
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
            real_t *tmp = cur;
            cur = nxt;
            nxt = tmp;
        }
    }
}
