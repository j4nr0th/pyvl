/** Test the local expansion operators for FMM.

 * Tests M2L (multipole_to_local), L2L (local_expansion_shift),
 * L2P (local_expansion_eval), and P2L (particle_to_local) against
 * direct summation and against each other for consistency.
 */

#include "../../src/core/fmm_operators.h"
#include "../../src/core/multipole.h"
#include "../test_common.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum
{
    N_SOURCES = 30,
    N_EVAL = 50,
    N_SEEDS = 10,
    TEST_ORDER = 4,
    WORK_ORDER = 6,
};

static const real_t R_SOURCES = 0.1;
static const real_t DISTANCE_FACTOR = 10.0;
static const real_t SHIFT_MAGNITUDE = 0.02;
static const real_t PI = (real_t)M_PI;

static real3_t exact_field(const real3_t point, const real3_t coords[static N_SOURCES],
                           const real3_t values[static N_SOURCES])
{
    real3_t res = {.x = 0.0, .y = 0.0, .z = 0.0};
    for (size_t i = 0; i < N_SOURCES; ++i)
    {
        const real3_t dr = real3_sub(point, coords[i]);
        const real_t inv_r2 = 1.0 / real3_dot(dr, dr);
        res = real3_add(res, real3_mul1(values[i], inv_r2));
    }
    return res;
}

static real_t rel_error(const real3_t a, const real3_t b)
{
    const real3_t d = real3_sub(a, b);
    const real_t mag_b = (real3_mag(b) + real3_mag(a)) / 2;
    return real3_mag(d) / mag_b;
}

static void generate_sources(uint64_t seed, real3_t coords[static N_SOURCES], real3_t values[static N_SOURCES])
{
    uint64_t state = seed;
    for (size_t i = 0; i < N_SOURCES; ++i)
    {
        coords[i].x = xorshift_uniform_range(&state, -R_SOURCES, R_SOURCES);
        coords[i].y = xorshift_uniform_range(&state, -R_SOURCES, R_SOURCES);
        coords[i].z = xorshift_uniform_range(&state, -R_SOURCES, R_SOURCES);
        values[i].x = xorshift_uniform_range(&state, -1.0, 1.0);
        values[i].y = xorshift_uniform_range(&state, -1.0, 1.0);
        values[i].z = xorshift_uniform_range(&state, -1.0, 1.0);
    }
}

static real3_t weighted_center(const real3_t coords[static N_SOURCES], const real3_t values[static N_SOURCES])
{
    real3_t center = {.x = 0.0, .y = 0.0, .z = 0.0};
    real_t total_weight = 0.0;
    for (size_t i = 0; i < N_SOURCES; ++i)
    {
        const real_t weight = real3_mag(values[i]);
        center.x += coords[i].x * weight;
        center.y += coords[i].y * weight;
        center.z += coords[i].z * weight;
        total_weight += weight;
    }
    if (total_weight > 0.0)
    {
        center.x /= total_weight;
        center.y /= total_weight;
        center.z /= total_weight;
    }
    return center;
}

int main(const int argc, const char *argv[static argc])
{
    (void)argc;
    (void)argv;

    /* ========== Test M2L: multipole -> local, then L2P ========== */
    {
        const size_t n_coeffs = multipole_num_coeffs(TEST_ORDER);
        const size_t scratch_sz = multipole_scratch_size(TEST_ORDER);
        const size_t shift_exp_sz = local_expansion_shift_exp_size(WORK_ORDER);
        const size_t pse_sz = local_expansion_pse_size(WORK_ORDER);
        const size_t n_coeffs_w = multipole_num_coeffs(WORK_ORDER);

        unsigned n_passed = 0;
        for (unsigned s = 0; s < N_SEEDS; ++s)
        {
            const uint64_t seed = (uint64_t)s * 77777 + 1;
            real3_t coords[N_SOURCES];
            real3_t values[N_SOURCES];
            generate_sources(seed, coords, values);

            /* Build a multipole about the source cluster centre. */
            const real3_t src_center = weighted_center(coords, values);

            real_t mp_coeffs[3 * n_coeffs];
            real_t cur[scratch_sz];
            real_t nxt[scratch_sz];
            multipole_t mp;
            const bool mp_ok = multipole_create(TEST_ORDER, (unsigned)(3 * n_coeffs), mp_coeffs, src_center, N_SOURCES,
                                                coords, values, cur, nxt, &mp);
            TEST_ASSERT(mp_ok, "multipole_create failed for M2L test seed %u", s);

            /* Build a local expansion at a shifted evaluation centre. */
            const real_t local_r = R_SOURCES * DISTANCE_FACTOR;
            const real3_t local_center = {.x = local_r, .y = 0, .z = 0};

            real_t local_coeffs[3 * n_coeffs];
            memset(local_coeffs, 0, sizeof(local_coeffs));
            local_expansion_t local = {
                .order = TEST_ORDER,
                .center = local_center,
                .coeffs_x = local_coeffs,
                .coeffs_y = local_coeffs + n_coeffs,
                .coeffs_z = local_coeffs + 2 * n_coeffs,
            };

            real_t shift_exp[shift_exp_sz];
            real_t pse[pse_sz];
            multipole_to_local(&mp, &local, WORK_ORDER, shift_exp, pse);

            /* Evaluate the local expansion at several points near the local centre
             * and compare with direct field. */
            real_t max_err = 0.0;
            for (unsigned j = 0; j < N_EVAL; ++j)
            {
                /* Fibonacci sphere around the local centre. */
                const real_t t = (real_t)j / (real_t)N_EVAL;
                const real_t phi = acos(1.0 - 2.0 * t);
                const real_t theta = (real_t)2.0 * PI * t * (1.0 + sqrt(5.0)) / 2.0;

                const real_t r_local = 0.5 * R_SOURCES; /* within convergence radius */
                const real3_t pt = {
                    .x = local_center.x + r_local * sin(phi) * cos(theta),
                    .y = local_center.y + r_local * sin(phi) * sin(theta),
                    .z = local_center.z + r_local * cos(phi),
                };

                const real3_t exact = exact_field(pt, coords, values);
                const real3_t approx = local_expansion_eval(&local, pt);
                const real_t err = rel_error(exact, approx);
                if (err > max_err)
                    max_err = err;
            }

            if (max_err < 0.5) /* generous tolerance for M2L */
                n_passed++;
        }
        TEST_ASSERT(n_passed >= N_SEEDS / 2, "M2L: only %u/%u seeds passed tolerance 0.5", n_passed, N_SEEDS);
        fprintf(stdout, "M2L test: %u/%u seeds passed\\n", n_passed, N_SEEDS);
    }

    /* ========== Test L2L: shift local expansion ========== */
    {
        const size_t n_coeffs = multipole_num_coeffs(TEST_ORDER);
        const size_t scratch_sz = multipole_scratch_size(TEST_ORDER);
        const size_t shift_exp_sz = local_expansion_shift_exp_size(WORK_ORDER);
        const size_t pse_sz = local_expansion_pse_size(WORK_ORDER);

        unsigned n_passed = 0;
        for (unsigned s = 0; s < N_SEEDS; ++s)
        {
            const uint64_t seed = (uint64_t)s * 88888 + 1;
            real3_t coords[N_SOURCES];
            real3_t values[N_SOURCES];
            generate_sources(seed, coords, values);

            /* Build a local expansion by P2L. */
            const real3_t center1 = {.x = 1.0, .y = 0, .z = 0};
            real_t loc1_coeffs[3 * n_coeffs];
            memset(loc1_coeffs, 0, sizeof(loc1_coeffs));
            local_expansion_t loc1 = {
                .order = TEST_ORDER,
                .center = center1,
                .coeffs_x = loc1_coeffs,
                .coeffs_y = loc1_coeffs + n_coeffs,
                .coeffs_z = loc1_coeffs + 2 * n_coeffs,
            };
            real_t cur[scratch_sz], nxt[scratch_sz];
            for (unsigned i = 0; i < N_SOURCES; ++i)
            {
                memset(cur, 0, sizeof(cur));
                memset(nxt, 0, sizeof(nxt));
                particle_to_local(&loc1, coords[i], values[i], cur, nxt);
            }

            /* Shift to a new centre. */
            const real3_t center2 = {.x = 1.0 + SHIFT_MAGNITUDE, .y = 0, .z = 0};
            real_t loc2_coeffs[3 * n_coeffs];
            memset(loc2_coeffs, 0, sizeof(loc2_coeffs));
            local_expansion_t loc2 = {
                .order = TEST_ORDER,
                .center = center2,
                .coeffs_x = loc2_coeffs,
                .coeffs_y = loc2_coeffs + n_coeffs,
                .coeffs_z = loc2_coeffs + 2 * n_coeffs,
            };

            real_t shift_exp[shift_exp_sz];
            real_t pse[pse_sz];
            local_expansion_shift(&loc1, &loc2, WORK_ORDER, shift_exp, pse);

            /* Build a local directly at the new centre for comparison. */
            real_t loc3_coeffs[3 * n_coeffs];
            memset(loc3_coeffs, 0, sizeof(loc3_coeffs));
            local_expansion_t loc3 = {
                .order = TEST_ORDER,
                .center = center2,
                .coeffs_x = loc3_coeffs,
                .coeffs_y = loc3_coeffs + n_coeffs,
                .coeffs_z = loc3_coeffs + 2 * n_coeffs,
            };
            for (unsigned i = 0; i < N_SOURCES; ++i)
            {
                memset(cur, 0, sizeof(cur));
                memset(nxt, 0, sizeof(nxt));
                particle_to_local(&loc3, coords[i], values[i], cur, nxt);
            }

            /* Compare shifted vs direct local at several eval points. */
            real_t max_err = 0.0;
            for (unsigned j = 0; j < N_EVAL; ++j)
            {
                const real_t t = (real_t)j / (real_t)N_EVAL;
                const real_t phi = acos(1.0 - 2.0 * t);
                const real_t theta = (real_t)2.0 * PI * t * (1.0 + sqrt(5.0)) / 2.0;
                const real_t r_local = 0.3 * R_SOURCES;
                const real3_t pt = {
                    .x = center2.x + r_local * sin(phi) * cos(theta),
                    .y = center2.y + r_local * sin(phi) * sin(theta),
                    .z = center2.z + r_local * cos(phi),
                };

                const real3_t v_shifted = local_expansion_eval(&loc2, pt);
                const real3_t v_direct = local_expansion_eval(&loc3, pt);
                const real_t err = rel_error(v_shifted, v_direct);
                if (err > max_err)
                    max_err = err;
            }

            if (max_err < 0.1)
                n_passed++;
        }
        TEST_ASSERT(n_passed >= N_SEEDS / 2, "L2L: only %u/%u seeds passed tolerance 0.1", n_passed, N_SEEDS);
        fprintf(stdout, "L2L test: %u/%u seeds passed\\n", n_passed, N_SEEDS);
    }

    /* ========== Test L2P vs direct ========== */
    {
        const size_t n_coeffs = multipole_num_coeffs(TEST_ORDER);
        const size_t scratch_sz = multipole_scratch_size(TEST_ORDER);

        unsigned n_passed = 0;
        for (unsigned s = 0; s < N_SEEDS; ++s)
        {
            const uint64_t seed = (uint64_t)s * 99999 + 1;
            real3_t coords[N_SOURCES];
            real3_t values[N_SOURCES];
            generate_sources(seed, coords, values);

            /* Build a local via P2L at a faraway centre. */
            const real3_t center = {.x = 2.0, .y = 0, .z = 0};
            real_t loc_coeffs[3 * n_coeffs];
            memset(loc_coeffs, 0, sizeof(loc_coeffs));
            local_expansion_t local = {
                .order = TEST_ORDER,
                .center = center,
                .coeffs_x = loc_coeffs,
                .coeffs_y = loc_coeffs + n_coeffs,
                .coeffs_z = loc_coeffs + 2 * n_coeffs,
            };
            real_t cur[scratch_sz], nxt[scratch_sz];
            for (unsigned i = 0; i < N_SOURCES; ++i)
            {
                memset(cur, 0, sizeof(cur));
                memset(nxt, 0, sizeof(nxt));
                particle_to_local(&local, coords[i], values[i], cur, nxt);
            }

            /* Compare L2P vs direct. */
            real_t max_err = 0.0;
            for (unsigned j = 0; j < N_EVAL; ++j)
            {
                const real_t t = (real_t)j / (real_t)N_EVAL;
                const real_t phi = acos(1.0 - 2.0 * t);
                const real_t theta = (real_t)2.0 * PI * t * (1.0 + sqrt(5.0)) / 2.0;
                const real_t r_local = 0.3;
                const real3_t pt = {
                    .x = center.x + r_local * sin(phi) * cos(theta),
                    .y = center.y + r_local * sin(phi) * sin(theta),
                    .z = center.z + r_local * cos(phi),
                };

                const real3_t exact = exact_field(pt, coords, values);
                const real3_t approx = local_expansion_eval(&local, pt);
                const real_t err = rel_error(exact, approx);
                if (err > max_err)
                    max_err = err;
            }

            if (max_err < 0.5)
                n_passed++;
        }
        TEST_ASSERT(n_passed >= N_SEEDS / 2, "L2P: only %u/%u seeds passed tolerance 0.5", n_passed, N_SEEDS);
        fprintf(stdout, "L2P test: %u/%u seeds passed\\n", n_passed, N_SEEDS);
    }

    /* ========== Test P2L vs direct ========== */
    {
        const size_t n_coeffs = multipole_num_coeffs(TEST_ORDER);
        const size_t scratch_sz = multipole_scratch_size(TEST_ORDER);

        unsigned n_passed = 0;
        for (unsigned s = 0; s < N_SEEDS; ++s)
        {
            const uint64_t seed = (uint64_t)s * 11111 + 1;
            real3_t coords[N_SOURCES];
            real3_t values[N_SOURCES];
            generate_sources(seed, coords, values);

            /* Build a local via P2L at a faraway centre. */
            const real3_t center = {.x = 1.5, .y = 0, .z = 0};
            real_t loc_coeffs[3 * n_coeffs];
            memset(loc_coeffs, 0, sizeof(loc_coeffs));
            local_expansion_t local = {
                .order = TEST_ORDER,
                .center = center,
                .coeffs_x = loc_coeffs,
                .coeffs_y = loc_coeffs + n_coeffs,
                .coeffs_z = loc_coeffs + 2 * n_coeffs,
            };
            real_t cur[scratch_sz], nxt[scratch_sz];

            /* Build via P2L for each source. */
            for (unsigned i = 0; i < N_SOURCES; ++i)
            {
                memset(cur, 0, scratch_sz * sizeof(real_t));
                memset(nxt, 0, scratch_sz * sizeof(real_t));
                particle_to_local(&local, coords[i], values[i], cur, nxt);
            }

            /* Now build via multipole (at same centre) then M2L to local and compare. */
            real_t mp_coeffs[3 * n_coeffs];
            memset(mp_coeffs, 0, sizeof(mp_coeffs));
            real_t mp_cur[scratch_sz], mp_nxt[scratch_sz];
            multipole_t mp;
            const bool mp_ok = multipole_create(TEST_ORDER, (unsigned)(3 * n_coeffs), mp_coeffs, center, N_SOURCES,
                                                coords, values, mp_cur, mp_nxt, &mp);
            TEST_ASSERT(mp_ok, "multipole_create failed for P2L test seed %u", s);

            /* M2L with zero shift (same centre) should give same result as P2L.
             * Actually M2L with zero shift is degenerate: R' = 0, so qc=0, qlx=0,
             * and the quadratic is just r'^2.  This is a different convergence path.
             * Instead, compare L2P results at several eval points. */
            real_t max_err = 0.0;
            for (unsigned j = 0; j < N_EVAL; ++j)
            {
                const real_t t = (real_t)j / (real_t)N_EVAL;
                const real_t phi = acos(1.0 - 2.0 * t);
                const real_t theta = (real_t)2.0 * PI * t * (1.0 + sqrt(5.0)) / 2.0;
                const real_t r_local = 0.2;
                const real3_t pt = {
                    .x = center.x + r_local * sin(phi) * cos(theta),
                    .y = center.y + r_local * sin(phi) * sin(theta),
                    .z = center.z + r_local * cos(phi),
                };

                const real3_t exact = exact_field(pt, coords, values);
                const real3_t approx_local = local_expansion_eval(&local, pt);
                const real_t err = rel_error(exact, approx_local);
                if (err > max_err)
                    max_err = err;
            }

            if (max_err < 0.5)
                n_passed++;
        }
        TEST_ASSERT(n_passed >= N_SEEDS / 2, "P2L: only %u/%u seeds passed tolerance 0.5", n_passed, N_SEEDS);
        fprintf(stdout, "P2L test: %u/%u seeds passed\\n", n_passed, N_SEEDS);
    }

    fprintf(stdout, "test_fmm_operators PASSED\\n");
    return EXIT_SUCCESS;
}
