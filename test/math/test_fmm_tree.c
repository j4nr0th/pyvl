/** Test the FMM tree build and tree-code evaluation.

 * Builds an FMM tree from random sources and compares the tree-code
 * evaluation against the direct O(N²) sum for far-field targets.
 * Tests multiple orders and seeds for statistical coverage.
 */

#include "../../src/core/fmm_tree.h"
#include "../test_common.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

enum
{
    N_SOURCES = 200,
    N_EVAL = 50,
    N_SEEDS = 5,
    TEST_ORDER = 4,
    TEST_N_THREADS = 4,
};

static void generate_sources(uint64_t seed, unsigned n, real3_t coords[static n], real3_t values[static n])
{
    uint64_t state = seed;
    const real_t R = 0.1;
    for (unsigned i = 0; i < n; ++i)
    {
        coords[i].x = xorshift_uniform_range(&state, -R, R);
        coords[i].y = xorshift_uniform_range(&state, -R, R);
        coords[i].z = xorshift_uniform_range(&state, -R, R);
        values[i].x = xorshift_uniform_range(&state, -1.0, 1.0);
        values[i].y = xorshift_uniform_range(&state, -1.0, 1.0);
        values[i].z = xorshift_uniform_range(&state, -1.0, 1.0);
    }
}

static real3_t exact_field(const real3_t point, const real3_t coords[static N_SOURCES],
                           const real3_t values[static N_SOURCES])
{
    real3_t res = {.x = 0, .y = 0, .z = 0};
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
    const real_t denom = (real3_mag(b) + real3_mag(a)) * 0.5;
    if (denom == 0.0)
        return 0.0;
    return real3_mag(d) / denom;
}

int main(const int argc, const char *argv[static argc])
{
    (void)argc;
    (void)argv;

    /* ========== Stage 1: basic build ========== */

    {
        /* Build with default settings. */
        uint64_t seed = 42;
        real3_t coords[N_SOURCES];
        real3_t values[N_SOURCES];
        generate_sources(seed, N_SOURCES, coords, values);

        const fmm_settings_t settings = {
            .order = TEST_ORDER,
            .critical_particle_count = 4,
            .max_depth = 20,
            .work_order = 0,
            .alpha_centroid = 0.0,
        };

        fmm_tree_t tree = {0};
        const bool ok = fmm_tree_build(N_SOURCES, TEST_N_THREADS, coords, values, &settings, &TEST_ALLOCATOR, &tree);
        TEST_ASSERT(ok, "fmm_tree_build failed for basic build");

        TEST_ASSERT(tree.n_nodes > 0, "Expected n_nodes > 0, got %u", tree.n_nodes);
        TEST_ASSERT(tree.n_sources == N_SOURCES, "Expected n_sources == %d, got %u", N_SOURCES, tree.n_sources);
        TEST_ASSERT(tree.n_internal + tree.n_multipole_leaves + tree.n_particle_leaves == tree.n_nodes,
                    "Node count mismatch: %u + %u + %u = %u != %u", tree.n_internal, tree.n_multipole_leaves,
                    tree.n_particle_leaves, tree.n_internal + tree.n_multipole_leaves + tree.n_particle_leaves,
                    tree.n_nodes);
        TEST_ASSERT(tree.max_depth_reached > 0, "Expected max_depth > 0");
        TEST_ASSERT(tree.buffer_size > 0, "Expected buffer_size > 0");

        /* Cleanup. */
        if (tree.buffer)
        {
            TEST_ALLOCATOR.deallocate(TEST_ALLOCATOR.state, tree.buffer);
            tree.buffer = NULL;
        }
    }

    /* ========== Stage 2: eval accuracy at far field ========== */

    {
        const real_t TOLERANCE = 0.5; /* generous for first test */
        unsigned n_passed = 0;

        for (unsigned s = 0; s < N_SEEDS; ++s)
        {
            const uint64_t seed = (uint64_t)s * 12345 + 1;
            real3_t coords[N_SOURCES];
            real3_t values[N_SOURCES];
            generate_sources(seed, N_SOURCES, coords, values);

            const fmm_settings_t settings = {
                .order = TEST_ORDER,
                .critical_particle_count = 4,
                .max_depth = 20,
                .work_order = 0,
                .alpha_centroid = 0.0,
            };

            fmm_tree_t tree = {0};
            const bool ok =
                fmm_tree_build(N_SOURCES, TEST_N_THREADS, coords, values, &settings, &TEST_ALLOCATOR, &tree);
            TEST_ASSERT(ok, "fmm_tree_build failed for seed %u", s);

            real_t max_err = 0.0;
            const real_t R = 1.0;
            const real_t PI = (real_t)M_PI;

            for (unsigned j = 0; j < N_EVAL; ++j)
            {
                const real_t t = (real_t)j / (real_t)N_EVAL;
                const real_t phi = acos(1.0 - 2.0 * t);
                const real_t theta = (real_t)2.0 * PI * t * (1.0 + sqrt(5.0)) / 2.0;

                const real3_t pt = {
                    .x = R * sin(phi) * cos(theta),
                    .y = R * sin(phi) * sin(theta),
                    .z = R * cos(phi),
                };
                const real3_t exact = exact_field(pt, coords, values);
                const real3_t approx = fmm_tree_eval(&tree, coords, values, pt, FMM_EVAL_SETTINGS_DEFAULT);
                const real_t err = rel_error(exact, approx);
                if (err > max_err)
                    max_err = err;
            }

            if (max_err < TOLERANCE)
                n_passed++;

            if (tree.buffer)
            {
                TEST_ALLOCATOR.deallocate(TEST_ALLOCATOR.state, tree.buffer);
                tree.buffer = NULL;
            }
        }

        TEST_ASSERT(n_passed >= N_SEEDS / 2, "Only %u/%u seeds passed the accuracy tolerance %e", n_passed, N_SEEDS,
                    TOLERANCE);
    }

    /* ========== Stage 3: determinism ========== */

    {
        const uint64_t seed = 12345;
        real3_t coords[N_SOURCES];
        real3_t values[N_SOURCES];
        generate_sources(seed, N_SOURCES, coords, values);

        const fmm_settings_t settings = {
            .order = TEST_ORDER,
            .critical_particle_count = 4,
            .max_depth = 20,
            .work_order = 0,
            .alpha_centroid = 0.0,
        };

        fmm_tree_t tree1 = {0}, tree2 = {0};
        const bool ok1 = fmm_tree_build(N_SOURCES, TEST_N_THREADS, coords, values, &settings, &TEST_ALLOCATOR, &tree1);
        const bool ok2 = fmm_tree_build(N_SOURCES, TEST_N_THREADS, coords, values, &settings, &TEST_ALLOCATOR, &tree2);
        TEST_ASSERT(ok1 && ok2, "Build failed for determinism test");

        TEST_ASSERT(tree1.n_nodes == tree2.n_nodes, "Node count mismatch: %u vs %u", tree1.n_nodes, tree2.n_nodes);
        TEST_ASSERT(tree1.n_internal == tree2.n_internal, "Internal count mismatch");
        TEST_ASSERT(tree1.n_multipole_leaves == tree2.n_multipole_leaves, "Multipole leaf count mismatch");

        const real_t R = 1.0;
        const real_t PI = (real_t)M_PI;
        for (unsigned j = 0; j < N_EVAL; ++j)
        {
            const real_t t = (real_t)j / (real_t)N_EVAL;
            const real_t phi = acos(1.0 - 2.0 * t);
            const real_t theta = (real_t)2.0 * PI * t * (1.0 + sqrt(5.0)) / 2.0;
            const real3_t pt = {
                .x = R * sin(phi) * cos(theta),
                .y = R * sin(phi) * sin(theta),
                .z = R * cos(phi),
            };
            const real3_t v1 = fmm_tree_eval(&tree1, coords, values, pt, FMM_EVAL_SETTINGS_DEFAULT);
            const real3_t v2 = fmm_tree_eval(&tree2, coords, values, pt, FMM_EVAL_SETTINGS_DEFAULT);
            const real_t err = rel_error(v1, v2);
            TEST_ASSERT(err < 1e-14, "Determinism violation at target %u: rel error %e", j, err);
        }

        if (tree1.buffer)
        {
            TEST_ALLOCATOR.deallocate(TEST_ALLOCATOR.state, tree1.buffer);
        }
        if (tree2.buffer)
        {
            TEST_ALLOCATOR.deallocate(TEST_ALLOCATOR.state, tree2.buffer);
        }
    }

    /* ========== Stage 4: batched eval ========== */

    {
        const uint64_t seed = 99;
        real3_t coords[N_SOURCES];
        real3_t values[N_SOURCES];
        generate_sources(seed, N_SOURCES, coords, values);

        const fmm_settings_t settings = {
            .order = TEST_ORDER,
            .critical_particle_count = 4,
            .max_depth = 20,
            .work_order = 0,
            .alpha_centroid = 0.0,
        };

        fmm_tree_t tree = {0};
        const bool ok = fmm_tree_build(N_SOURCES, TEST_N_THREADS, coords, values, &settings, &TEST_ALLOCATOR, &tree);
        TEST_ASSERT(ok, "Build failed for batch eval test");

        const unsigned N_PTS = 10;
        real3_t targets[N_PTS];
        real3_t results[N_PTS];
        const real_t R = 1.0;
        const real_t PI = (real_t)M_PI;
        for (unsigned j = 0; j < N_PTS; ++j)
        {
            const real_t t = (real_t)j / (real_t)N_PTS;
            const real_t phi = acos(1.0 - 2.0 * t);
            const real_t theta = (real_t)2.0 * PI * t * (1.0 + sqrt(5.0)) / 2.0;
            targets[j] = (real3_t){
                .x = R * sin(phi) * cos(theta),
                .y = R * sin(phi) * sin(theta),
                .z = R * cos(phi),
            };
        }

        fmm_tree_eval_all(&tree, coords, values, N_PTS, targets, results, FMM_EVAL_SETTINGS_DEFAULT, 1);

        for (unsigned j = 0; j < N_PTS; ++j)
        {
            const real3_t single = fmm_tree_eval(&tree, coords, values, targets[j], FMM_EVAL_SETTINGS_DEFAULT);
            const real_t err = rel_error(results[j], single);
            TEST_ASSERT(err < 1e-15, "Batch vs single mismatch at target %u: rel error %e", j, err);
        }

        if (tree.buffer)
        {
            TEST_ALLOCATOR.deallocate(TEST_ALLOCATOR.state, tree.buffer);
        }
    }

    fprintf(stdout, "test_fmm_tree PASSED\\n");
    return EXIT_SUCCESS;
}
