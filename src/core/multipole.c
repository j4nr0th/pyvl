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

void multipole_update(const multipole_t *multipole, const real3_t center, const real3_t source_pos,
                      const real3_t source_value, real_t CVL_ARRAY_ARG(cur, restrict),
                      real_t CVL_ARRAY_ARG(nxt, restrict))
{
    const size_t order = multipole->order;
    const size_t scratch = multipole_scratch_size(order);
    const unsigned dim = order + 1;
    const size_t dim2 = (size_t)dim * dim;

    const real3_t rel_pos = real3_sub(source_pos, center);
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
#pragma omp simd
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
    real_t *restrict const coeffs_x = coeffs;
    real_t *restrict const coeffs_y = coeffs + needed_coeffs;
    real_t *restrict const coeffs_z = coeffs + 2 * needed_coeffs;

    const multipole_t this = {
        .order = order,
        .center = center,
        .coeffs_x = coeffs_x,
        .coeffs_y = coeffs_y,
        .coeffs_z = coeffs_z,
    };

    const unsigned dim = order + 1;
    const size_t dim2 = (size_t)dim * dim;

    for (size_t i = 0; i < sources; ++i)
    {
        // cur holds the coefficients of (2 r·pos - pos·pos)^m as a polynomial in r.
        // nxt is used to step from m to m+1.
        memset(cur, 0, scratch * sizeof(real_t));
        memset(nxt, 0, scratch * sizeof(real_t));

        multipole_update(&this, center, sources_coords[i], sources_values[i], cur, nxt);
    }

    // Fill output
    *out = this;

    return true;
}

real3_t multipole_eval(const multipole_t *multipole, const real3_t point)
{
    const real3_t rel_point = real3_sub(point, multipole->center);
    const real_t inv_r = 1.0 / sqrt(rel_point.x * rel_point.x + rel_point.y * rel_point.y + rel_point.z * rel_point.z);
    real_t scale = inv_r * inv_r;
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
        res = real3_add(res, real3_mul1(term, scale));
        scale *= inv_r * inv_r;
    }

    return res;
}
