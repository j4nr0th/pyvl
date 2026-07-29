#include "multipole.h"

#include <math.h>
#include <string.h>

size_t multipole_num_coeffs(unsigned order)
{
    // Full expansion stores the polynomial (2 r·s - s·s)^m for each order m.
    // Order m needs C(m+3, 3) monomials; summing m=0..order gives C(order+4, 4).
    return (size_t)(order + 1) * (order + 2) * (order + 3) * (order + 4) / 24;
}

size_t multipole_scratch_size(unsigned order)
{
    const size_t dim = (size_t)order + 1;
    return dim * dim * dim;
}

void multipole_update(const multipole_t *multipole, const real3_t source_pos, const real3_t source_value,
                      real_t CVL_ARRAY_ARG(cur, restrict), real_t CVL_ARRAY_ARG(nxt, restrict))
{
    const size_t order = multipole->order;
    const size_t scratch = multipole_scratch_size(order);
    const unsigned dim = order + 1;
    const size_t dim2 = (size_t)dim * dim;

    const real3_t rel_pos = real3_sub(source_pos, multipole->center);
    const real_t s2 = real3_dot(rel_pos, rel_pos);

    real_t *restrict const coeffs_x = multipole->coeffs_x;
    real_t *restrict const coeffs_y = multipole->coeffs_y;
    real_t *restrict const coeffs_z = multipole->coeffs_z;

    // Initialize cur to represent the polynomial (2 r·pos - pos·pos)^0 = 1
    cur[0] = 1.0;
    size_t idx = 0;
    for (unsigned m = 0; m <= order; ++m)
    {
        // Accumulate the full polynomial for order m into the coefficient buffer.
        for (unsigned p = 0; p <= m; ++p)
        {
            for (unsigned q = 0; q <= m - p; ++q)
            {
                const unsigned r_max = m - p - q;
                for (unsigned r = 0; r <= r_max; ++r)
                {
                    const real_t c = cur[p * dim2 + q * dim + r];
                    coeffs_x[idx + r] += source_value.x * c;
                    coeffs_y[idx + r] += source_value.y * c;
                    coeffs_z[idx + r] += source_value.z * c;
                }
                idx += (size_t)(r_max + 1);
            }
        }

        if (m == order)
            break;

        // nxt = (2 r·pos - pos·pos) * cur
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

                    nxt[cur_idx] -= s2 * c;
                    nxt[(p + 1) * dim2 + q * dim + r] += 2.0 * rel_pos.x * c;
                    nxt[p * dim2 + (q + 1) * dim + r] += 2.0 * rel_pos.y * c;
                    nxt[p * dim2 + q * dim + (r + 1)] += 2.0 * rel_pos.z * c;
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
CVL_INTERNAL size_t multipole_coeff_index(unsigned m, unsigned p, unsigned q, unsigned r)
{
    // Coefficients are stored in tetrahedral blocks: block k contains all monomials
    // x^p y^q z^r with p + q + r <= k, ordered by the same nested loops used in
    // multipole_update and multipole_eval. This helper returns the linear index of
    // monomial (p,q,r) inside block m, ignoring the vector-component offset.

    // Offset of block m: sum_{k=0}^{m-1} C(k+3,3) = C(m+3,4).
    size_t idx = (size_t)m * (m + 1) * (m + 2) * (m + 3) / 24;

    // Monomials with x-degree less than p.
    for (unsigned pp = 0; pp < p; ++pp)
    {
        idx += (size_t)(m - pp + 2) * (m - pp + 1) / 2;
    }

    // Monomials with x-degree p and y-degree less than q.
    idx += (size_t)q * (m - p + 1) - (size_t)q * (q - 1) / 2;

    // Remaining r offset.
    idx += r;
    return idx;
}

CVL_INTERNAL void multipole_add_poly_to_order(const real_t *poly, unsigned out_order, real_t scale, const real_t cx,
                                              const real_t cy, const real_t cz, const multipole_t *out)
{
    for (unsigned deg = 0; deg <= out_order; ++deg)
    {
        for (unsigned p = 0; p <= deg; ++p)
        {
            for (unsigned q = 0; q <= deg - p; ++q)
            {
                const unsigned r = deg - p - q;
                const size_t poly_idx = multipole_coeff_index(deg, p, q, r);
                const real_t c = poly[poly_idx];
                if (c == 0.0)
                    continue;
                const size_t out_idx = multipole_coeff_index(out_order, p, q, r);
                out->coeffs_x[out_idx] += scale * cx * c;
                out->coeffs_y[out_idx] += scale * cy * c;
                out->coeffs_z[out_idx] += scale * cz * c;
            }
        }
    }
}

CVL_INTERNAL void multipole_poly_mul_linear(const real_t *a, real_t *b, real_t lx, real_t ly, real_t lz, real_t lc,
                                            unsigned max_order)
{
    const size_t n_coeffs = multipole_num_coeffs(max_order);
    for (size_t i = 0; i < n_coeffs; ++i)
    {
        b[i] = 0.0;
    }

    for (unsigned deg = 0; deg <= max_order; ++deg)
    {
        for (unsigned p = 0; p <= deg; ++p)
        {
            for (unsigned q = 0; q <= deg - p; ++q)
            {
                const unsigned r = deg - p - q;
                const size_t idx = multipole_coeff_index(deg, p, q, r);
                const real_t c = a[idx];
                if (c == 0.0)
                    continue;

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

CVL_INTERNAL void multipole_poly_mul_quadratic(const real_t *a, real_t *b, real_t qx, real_t qy, real_t qz, real_t qlx,
                                               real_t qly, real_t qlz, real_t qc, unsigned max_order)
{
    const size_t n_coeffs = multipole_num_coeffs(max_order);
    for (size_t i = 0; i < n_coeffs; ++i)
    {
        b[i] = 0.0;
    }

    for (unsigned deg = 0; deg <= max_order; ++deg)
    {
        for (unsigned p = 0; p <= deg; ++p)
        {
            for (unsigned q = 0; q <= deg - p; ++q)
            {
                const unsigned r = deg - p - q;
                const size_t idx = multipole_coeff_index(deg, p, q, r);
                const real_t c = a[idx];
                if (c == 0.0)
                    continue;

                /* Constant term. */
                b[idx] += qc * c;

                /* Linear terms (degree + 1). */
                if (deg < max_order)
                {
                    b[multipole_coeff_index(deg + 1, p + 1, q, r)] += qlx * c;
                    b[multipole_coeff_index(deg + 1, p, q + 1, r)] += qly * c;
                    b[multipole_coeff_index(deg + 1, p, q, r + 1)] += qlz * c;
                }

                /* Quadratic terms (degree + 2). */
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

void multipole_add_shift(const multipole_t *in, const multipole_t *out, unsigned work_order,
                         real_t CVL_ARRAY_ARG(shift_exp, restrict), real_t CVL_ARRAY_ARG(pse, restrict))
{
    const unsigned in_order = in->order;
    const unsigned out_order = out->order;

    const real3_t shift = real3_sub(in->center, out->center);
    const real_t s2 = real3_dot(shift, shift);

    const real_t lx = 2.0 * shift.x;
    const real_t ly = 2.0 * shift.y;
    const real_t lz = 2.0 * shift.z;
    const real_t lc = -s2;

    // Build binomial expansions of (x - shift_x)^e etc up to work_order.
    const size_t shift_dim = (size_t)work_order + 1;
    const size_t shift_plane = shift_dim * shift_dim;
    build_binomial_expansion(shift_exp, -shift.x, -shift.y, -shift.z, work_order, shift_dim, shift_plane);

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

                    // Numerator shift polynomial: (x - sx)^p (y - sy)^q (z - sz)^r
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

                    // l = 0: denominator factor = 1
                    if (m <= out_order)
                    {
                        multipole_add_poly_to_order(pse, m, 1.0, cx, cy, cz, out);
                    }

                    // l >= 1: denominator binomial series (2 r_B·shift - |shift|^2)^l
                    unsigned cur = 0;
                    unsigned nxt = 1;
                    real_t binom = 1.0;
                    const unsigned max_l = (work_order > m) ? work_order - m : 0;
                    for (unsigned l = 1; l <= max_l; ++l)
                    {
                        multipole_poly_mul_linear(pse + cur * n_coeffs, pse + nxt * n_coeffs, lx, ly, lz, lc,
                                                  work_order);
                        binom = binom * (m + l) / (real_t)l;
                        if (m + l <= out_order)
                        {
                            multipole_add_poly_to_order(pse + nxt * n_coeffs, m + l, binom, cx, cy, cz, out);
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

bool multipole_create(unsigned order, unsigned num_coeffs, real_t CVL_ARRAY_ARG(coeffs, restrict num_coeffs),
                      const real3_t center, unsigned sources,
                      const real3_t CVL_ARRAY_ARG(sources_coords, restrict sources),
                      const real3_t CVL_ARRAY_ARG(sources_values, restrict sources),
                      real_t CVL_ARRAY_ARG(cur, restrict num_coeffs), real_t CVL_ARRAY_ARG(nxt, restrict num_coeffs),
                      multipole_t *out)
{
    const size_t needed_coeffs = multipole_num_coeffs(order);
    const size_t scratch = multipole_scratch_size(order);
    if (num_coeffs < 3 * needed_coeffs || num_coeffs < scratch)
    {
        return false;
    }

    // Zero the coefficient array
    for (size_t i = 0; i < 3 * needed_coeffs; ++i)
    {
        coeffs[i] = 0.0;
    }
    // Prepare coeff arrays
    const multipole_t mp_built = {
        .order = order,
        .center = center,
        .coeffs_x = coeffs,
        .coeffs_y = coeffs + needed_coeffs,
        .coeffs_z = coeffs + 2 * needed_coeffs,
    };

    for (size_t i = 0; i < sources; ++i)
    {
        memset(cur, 0, scratch * sizeof(real_t));
        memset(nxt, 0, scratch * sizeof(real_t));

        multipole_update(&mp_built, sources_coords[i], sources_values[i], cur, nxt);
    }

    // Fill output field-by-field to avoid strict-aliasing UB when `out`
    // aliases into a union member (octree_node_t.data.mp in particular).
    out->order = mp_built.order;
    out->center = mp_built.center;
    out->coeffs_x = mp_built.coeffs_x;
    out->coeffs_y = mp_built.coeffs_y;
    out->coeffs_z = mp_built.coeffs_z;

    return true;
}

real3_t multipole_eval(const multipole_t *multipole, const real3_t point)
{
    const real3_t rel_point = real3_sub(point, multipole->center);
    const real_t inv_r = 1.0 / sqrt(rel_point.x * rel_point.x + rel_point.y * rel_point.y + rel_point.z * rel_point.z);
    const real_t inv_r2 = inv_r * inv_r;
    real_t scale = inv_r2;
    real3_t res = {.x = 0, .y = 0, .z = 0};
    size_t idx = 0;

    for (unsigned m = 0; m <= multipole->order; ++m)
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
