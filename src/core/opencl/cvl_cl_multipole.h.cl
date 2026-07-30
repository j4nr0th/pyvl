#pragma once
/*
 * Multipole polynomial arithmetic helpers for C17 / OpenCL C
 * dual-compilation.
 *
 * All functions are `static inline` and may be included from both
 * C17 (host) and OpenCL C (device) translation units.
 *
 * Depends on cvl_cl_math.h.cl for real_t, real3_t,
 * multipole_num_coeffs, multipole_coeff_index.
 */

#include "cvl_cl_math.h.cl"

#ifdef __OPENCL_C_VERSION__

/* ------------------------------------------------------------------ */
/*  build_binomial_expansion                                           */
/* ------------------------------------------------------------------ */

/**
 * @brief Build binomial expansion coefficients (x + s)^e for e = 0..work_order.
 *
 * For each dimension d, fills @p shift_exp with coefficients such that
 * shift_exp[d * shift_plane + e * shift_dim + i] is the coefficient of
 * x^i in (x + s_d)^e, for e = 0..work_order, i = 0..e, where s_d are
 * the @p sx / @p sy / @p sz arguments.
 *
 * Pass pre-signed shifts: e.g. -shift.x for (x - shift_x) expansion,
 * +R_prime.x for (x + R'_x) expansion.
 *
 * @param shift_exp  Output array [3 * shift_plane] where shift_plane = shift_dim^2.
 * @param sx         x-shift (pre-signed).
 * @param sy         y-shift (pre-signed).
 * @param sz         z-shift (pre-signed).
 * @param work_order Max expansion order.
 * @param shift_dim  work_order + 1.
 * @param shift_plane shift_dim * shift_dim.
 */
static inline void build_binomial_expansion(real_t *CVL_CL_RESTRICT shift_exp,
    real_t sx, real_t sy, real_t sz,
    unsigned work_order, size_t shift_dim, size_t shift_plane)
{
    const real_t s[3] = {sx, sy, sz};
    for (unsigned d = 0; d < 3; ++d)
    {
        shift_exp[d * shift_plane] = 1.0;
        for (unsigned e = 1; e <= work_order; ++e)
        {
            shift_exp[d * shift_plane + e * shift_dim] = s[d] * shift_exp[d * shift_plane + (e - 1) * shift_dim];
            for (unsigned i = 1; i <= e; ++i)
            {
                shift_exp[d * shift_plane + e * shift_dim + i] =
                    shift_exp[d * shift_plane + (e - 1) * shift_dim + (i - 1)] +
                    s[d] * shift_exp[d * shift_plane + (e - 1) * shift_dim + i];
            }
            for (unsigned i = e + 1; i <= work_order; ++i)
                shift_exp[d * shift_plane + e * shift_dim + i] = 0.0;
        }
    }
}

/* ------------------------------------------------------------------ */
/*  multipole_poly_mul_linear                                          */
/* ------------------------------------------------------------------ */

/**
 * @brief Multiply polynomial @p a by the linear form lc + lx*x + ly*y + lz*z,
 *        writing the result to @p b.
 *
 * @param a         Input polynomial (dense, multipole_num_coeffs layout).
 * @param b         Output polynomial (overwritten, not accumulated).
 * @param lx        Coefficient of x.
 * @param ly        Coefficient of y.
 * @param lz        Coefficient of z.
 * @param lc        Constant coefficient.
 * @param max_order Maximum polynomial order.
 */
static inline void multipole_poly_mul_linear(const real_t *CVL_CL_RESTRICT a, real_t *CVL_CL_RESTRICT b,
    real_t lx, real_t ly, real_t lz, real_t lc, unsigned max_order)
{
    size_t n_coeffs = multipole_num_coeffs(max_order);
    for (size_t i = 0; i < n_coeffs; ++i) b[i] = 0.0;

    for (unsigned deg = 0; deg <= max_order; ++deg)
    {
        for (unsigned p = 0; p <= deg; ++p)
        {
            for (unsigned q = 0; q <= deg - p; ++q)
            {
                unsigned r = deg - p - q;
                size_t idx = multipole_coeff_index(deg, p, q, r);
                real_t c = a[idx];
                if (c == 0.0) continue;

                b[idx] += lc * c;
                if (deg < max_order)
                {
                    b[multipole_coeff_index(deg + 1, p + 1, q, r)] += lx * c;
                    b[multipole_coeff_index(deg + 1, p, q + 1, r)] += ly * c;
                    b[multipole_coeff_index(deg + 1, p, q, r + 1)] += lz * c;
                }
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/*  multipole_poly_mul_quadratic                                       */
/* ------------------------------------------------------------------ */

/**
 * @brief Multiply polynomial @p a by the quadratic form
 *        qc + qlx*x + qly*y + qlz*z + qx*x^2 + qy*y^2 + qz*z^2,
 *        writing the result to @p b.
 *
 * Used by M2L (multipole_to_local) where the denominator factor is
 * (2 R'·r' + r'^2).
 *
 * @param a         Input polynomial (dense, multipole_num_coeffs layout).
 * @param b         Output polynomial (overwritten, not accumulated).
 * @param qx        Coefficient of x^2.
 * @param qy        Coefficient of y^2.
 * @param qz        Coefficient of z^2.
 * @param qlx       Coefficient of x.
 * @param qly       Coefficient of y.
 * @param qlz       Coefficient of z.
 * @param qc        Constant coefficient.
 * @param max_order Maximum polynomial order.
 */
static inline void multipole_poly_mul_quadratic(const real_t *CVL_CL_RESTRICT a, real_t *CVL_CL_RESTRICT b,
    real_t qx, real_t qy, real_t qz, real_t qlx, real_t qly, real_t qlz, real_t qc, unsigned max_order)
{
    size_t n_coeffs = multipole_num_coeffs(max_order);
    for (size_t i = 0; i < n_coeffs; ++i) b[i] = 0.0;

    for (unsigned deg = 0; deg <= max_order; ++deg)
    {
        for (unsigned p = 0; p <= deg; ++p)
        {
            for (unsigned q = 0; q <= deg - p; ++q)
            {
                unsigned r = deg - p - q;
                size_t idx = multipole_coeff_index(deg, p, q, r);
                real_t c = a[idx];
                if (c == 0.0) continue;

                /* Constant term */
                b[idx] += qc * c;

                /* Linear terms (degree + 1) */
                if (deg < max_order)
                {
                    b[multipole_coeff_index(deg + 1, p + 1, q, r)] += qlx * c;
                    b[multipole_coeff_index(deg + 1, p, q + 1, r)] += qly * c;
                    b[multipole_coeff_index(deg + 1, p, q, r + 1)] += qlz * c;
                }

                /* Quadratic terms (degree + 2) */
                if (deg + 1 < max_order)
                {
                    b[multipole_coeff_index(deg + 2, p + 2, q, r)] += qx * c;
                    b[multipole_coeff_index(deg + 2, p, q + 2, r)] += qy * c;
                    b[multipole_coeff_index(deg + 2, p, q, r + 2)] += qz * c;
                }
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/*  multipole_add_poly_to_order                                        */
/* ------------------------------------------------------------------ */

/**
 * @brief Scale a polynomial and accumulate it into coefficient arrays
 *        at the appropriate orders.
 *
 * The original C API takes a multipole_t struct; this version takes
 * the three coefficient arrays directly so it compiles in OpenCL C
 * where the multipole_t type is unavailable.
 *
 * @param poly      Polynomial coefficients (dense, multipole_num_coeffs layout).
 * @param out_order Maximum order to accumulate (orders 0..out_order).
 * @param scale     Scalar multiplier.
 * @param cx        x-component multiplier.
 * @param cy        y-component multiplier.
 * @param cz        z-component multiplier.
 * @param coeffs_x  x-component coefficient array (accumulated, not overwritten).
 * @param coeffs_y  y-component coefficient array (accumulated, not overwritten).
 * @param coeffs_z  z-component coefficient array (accumulated, not overwritten).
 */
static inline void multipole_add_poly_to_order(const real_t *CVL_CL_RESTRICT poly, unsigned out_order, real_t scale,
    real_t cx, real_t cy, real_t cz, real_t *CVL_CL_RESTRICT coeffs_x,
    real_t *CVL_CL_RESTRICT coeffs_y, real_t *CVL_CL_RESTRICT coeffs_z)
{
    for (unsigned deg = 0; deg <= out_order; ++deg)
    {
        for (unsigned p = 0; p <= deg; ++p)
        {
            for (unsigned q = 0; q <= deg - p; ++q)
            {
                unsigned r = deg - p - q;
                size_t poly_idx = multipole_coeff_index(deg, p, q, r);
                real_t c = poly[poly_idx];
                if (c == 0.0) continue;
                size_t out_idx = multipole_coeff_index(out_order, p, q, r);
                coeffs_x[out_idx] += scale * cx * c;
                coeffs_y[out_idx] += scale * cy * c;
                coeffs_z[out_idx] += scale * cz * c;
            }
        }
    }
}

#endif /* __OPENCL_C_VERSION__ */
