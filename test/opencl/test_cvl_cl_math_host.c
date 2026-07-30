/*
 * test_cvl_cl_math_host.c — Test that .cl.h math/multipole headers
 * compile as C17 and verify correctness via the project's native API.
 *
 * This test includes the .cl.h files to verify C17 compilation, but
 * calls the project's native C API (common.h, multipole.h, octree.h)
 * for actual function invocation since the .cl.h function definitions
 * are guarded by #ifdef __OPENCL_C_VERSION__.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Project headers for actual function calls */
#include "../../src/core/common.h"
#include "../../src/core/octree.h"

/* .cl.h headers — verify they compile in C17 mode */
#include "cvl_cl_math.h.cl"
#include "cvl_cl_multipole.h.cl"

/* multipole.h for coefficient helpers and build_binomial_expansion */
#include "../../src/core/multipole.h"

#include "../test_common.h"

int main(void)
{
    uint64_t rng = 12345;
    (void)rng;

    /* ----------------------------------------------------------------- */
    /*  Test 1: particle_kernel                                          */
    /* ----------------------------------------------------------------- */
    {
        real3_t gamma = {1.0, 2.0, 3.0};
        real3_t r = {4.0, 5.0, 6.0};
        real3_t res = particle_kernel(gamma, r);
        real_t r2 = 4.0 * 4.0 + 5.0 * 5.0 + 6.0 * 6.0; /* 77 */
        real_t inv = 1.0 / r2;
        TEST_ASSERT(fabs(res.x - 1.0 * inv) < 1e-15, "particle_kernel x");
        TEST_ASSERT(fabs(res.y - 2.0 * inv) < 1e-15, "particle_kernel y");
        TEST_ASSERT(fabs(res.z - 3.0 * inv) < 1e-15, "particle_kernel z");
    }

    /* ----------------------------------------------------------------- */
    /*  Test 2: particle_kernel zero distance (should return zero)       */
    /* ----------------------------------------------------------------- */
    {
        real3_t res = particle_kernel((real3_t){1, 1, 1}, (real3_t){0, 0, 0});
        TEST_ASSERT(res.x == 0 && res.y == 0 && res.z == 0, "particle_kernel zero");
    }

    /* ----------------------------------------------------------------- */
    /*  Test 3: multipole_num_coeffs                                     */
    /* ----------------------------------------------------------------- */
    TEST_ASSERT(multipole_num_coeffs(0) == 1, "ncoeffs0");
    TEST_ASSERT(multipole_num_coeffs(1) == 5, "ncoeffs1");
    TEST_ASSERT(multipole_num_coeffs(4) == 70, "ncoeffs4");
    TEST_ASSERT(multipole_num_coeffs(8) == 495, "ncoeffs8");

    /* ----------------------------------------------------------------- */
    /*  Test 4: multipole_coeff_index                                    */
    /* ----------------------------------------------------------------- */
    /* Block m=0: C(0+3,3)=1 entry at index 0.
     * Block m=1: starts at offset C(1+3,4)=C(4,4)=1.
     * Within m=1, entries are ordered by (p,q,r) with r innermost:
     *   (0,0,0)->1, (0,0,1)->2, (0,1,0)->3, (1,0,0)->4, (1,1,0)->5 */
    TEST_ASSERT(multipole_coeff_index(0, 0, 0, 0) == 0, "idx_0_0_0_0");
    TEST_ASSERT(multipole_coeff_index(1, 0, 0, 0) == 1, "idx_1_0_0_0");
    TEST_ASSERT(multipole_coeff_index(1, 0, 0, 1) == 2, "idx_1_0_0_1");
    TEST_ASSERT(multipole_coeff_index(1, 0, 1, 0) == 3, "idx_1_0_1_0");
    TEST_ASSERT(multipole_coeff_index(1, 1, 0, 0) == 4, "idx_1_1_0_0");
    TEST_ASSERT(multipole_coeff_index(1, 1, 1, 0) == 5, "idx_1_1_1_0");

    /* ----------------------------------------------------------------- */
    /*  Test 5: multipole_scratch_size                                   */
    /* ----------------------------------------------------------------- */
    TEST_ASSERT(multipole_scratch_size(0) == 1, "scratch0");
    TEST_ASSERT(multipole_scratch_size(4) == 125, "scratch4");

    /* ----------------------------------------------------------------- */
    /*  Test 6: build_binomial_expansion                                 */
    /* ----------------------------------------------------------------- */
    {
        unsigned wo = 3;
        size_t dim = wo + 1;
        size_t plane = dim * dim;
        /* 3 * plane = 3 * 16 = 48 elements */
        real_t shift_exp[3 * 16];
        build_binomial_expansion(shift_exp, 1.0, 2.0, 3.0, wo, dim, plane);

        /* (x+1)^1 = x + 1 -> coeff of x^1 = 1, coeff of x^0 = 1 */
        TEST_ASSERT(fabs(shift_exp[0 * plane + 1 * dim + 1] - 1.0) < 1e-15, "binom x^1 coeff x");
        TEST_ASSERT(fabs(shift_exp[0 * plane + 1 * dim + 0] - 1.0) < 1e-15, "binom x^1 const");

        /* (x+1)^2 = x^2 + 2x + 1 */
        TEST_ASSERT(fabs(shift_exp[0 * plane + 2 * dim + 2] - 1.0) < 1e-15, "binom x^2 coeff x^2");
        TEST_ASSERT(fabs(shift_exp[0 * plane + 2 * dim + 1] - 2.0) < 1e-15, "binom x^2 coeff x");
        TEST_ASSERT(fabs(shift_exp[0 * plane + 2 * dim + 0] - 1.0) < 1e-15, "binom x^2 const");
    }

    /* ----------------------------------------------------------------- */
    /*  Test 7: morton_3d                                                */
    /* ----------------------------------------------------------------- */
    {
        real3_t root_center = {0, 0, 0};
        real_t root_hs = 1.0;

        real3_t p = {0.1, 0.2, 0.3};
        uint64_t mc = morton_3d(p, root_center, root_hs);
        TEST_ASSERT(mc > 0, "morton_3d positive");

        /* Point at origin maps to ~half of max (in-range code) */
        real3_t pc = {0, 0, 0};
        uint64_t mc_c = morton_3d(pc, root_center, root_hs);
        TEST_ASSERT(mc_c > 0, "morton_3d center positive");
    }

    /* ----------------------------------------------------------------- */
    /*  Test 8: multipole_poly_mul_linear                                */
    /* ----------------------------------------------------------------- */
    {
        const unsigned order = 2;
        const size_t nc = multipole_num_coeffs(order); /* 15 */
        real_t a[256] = {0};
        real_t b[256] = {0};
        /* a = 1.0 (constant polynomial) */
        a[multipole_coeff_index(0, 0, 0, 0)] = 1.0;
        /* multiply by (2x + 3y + 4z + 1) */
        multipole_poly_mul_linear(a, b, 2.0, 3.0, 4.0, 1.0, order);
        /* result should be: 1*1 (const) + 2*x + 3*y + 4*z */
        TEST_ASSERT(fabs(b[multipole_coeff_index(0, 0, 0, 0)] - 1.0) < 1e-15, "poly_mul_linear const");
        TEST_ASSERT(fabs(b[multipole_coeff_index(1, 1, 0, 0)] - 2.0) < 1e-15, "poly_mul_linear x");
        TEST_ASSERT(fabs(b[multipole_coeff_index(1, 0, 1, 0)] - 3.0) < 1e-15, "poly_mul_linear y");
        TEST_ASSERT(fabs(b[multipole_coeff_index(1, 0, 0, 1)] - 4.0) < 1e-15, "poly_mul_linear z");
    }

    /* ----------------------------------------------------------------- */
    /*  Test 9: multipole_add_poly_to_order                              */
    /* ----------------------------------------------------------------- */
    {
        const unsigned order = 2;
        const size_t nc = multipole_num_coeffs(order);
        real_t coeffs[3 * 256] = {0};
        multipole_t mp = {
            .order = order,
            .center = {0, 0, 0},
            .coeffs_x = coeffs,
            .coeffs_y = coeffs + nc,
            .coeffs_z = coeffs + 2 * nc,
        };
        real_t poly[256] = {0};
        /* poly = 1.0 (constant) */
        poly[multipole_coeff_index(0, 0, 0, 0)] = 1.0;
        /* add poly (scale=2.0, cx=1, cy=2, cz=3) */
        multipole_add_poly_to_order(poly, order, 2.0, 1.0, 2.0, 3.0, &mp);
        size_t oidx = multipole_coeff_index(order, 0, 0, 0);
        TEST_ASSERT(fabs(coeffs[oidx] - 2.0) < 1e-15, "add_poly x");
        TEST_ASSERT(fabs(coeffs[nc + oidx] - 4.0) < 1e-15, "add_poly y");
        TEST_ASSERT(fabs(coeffs[2 * nc + oidx] - 6.0) < 1e-15, "add_poly z");
    }

    printf("All host math tests passed.\n");
    return 0;
}
