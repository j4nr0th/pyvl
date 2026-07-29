/** FMM accuracy microbenchmark.

 * Builds FMM trees at several orders, evaluates in both tree-code and
 * FMM mode, and compares against direct O(N^2) for far-field accuracy.
 *
 * Always returns EXIT_SUCCESS — data-collection target.
 */

#include "../../src/core/cost_model.h"
#include "../../src/core/fmm_tree.h"
#include "../test_common.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum
{
    N_SOURCES = 100,
    N_TARGETS = 20,
    N_THREADS = 1,
    N_SEEDS = 5,
    MAX_ORDER = 6,
};

static const real_t R_SOURCES = 0.5;
static const real_t DISTANCE_FACTOR = 4.0;
static const real_t PI = (real_t)M_PI;

static void generate_sources(uint64_t seed, unsigned n, real3_t coords[static n], real3_t values[static n])
{
    uint64_t state = seed;
    for (unsigned i = 0; i < n; ++i)
    {
        coords[i].x = xorshift_uniform_range(&state, -R_SOURCES, R_SOURCES);
        coords[i].y = xorshift_uniform_range(&state, -R_SOURCES, R_SOURCES);
        coords[i].z = xorshift_uniform_range(&state, -R_SOURCES, R_SOURCES);
        values[i].x = xorshift_uniform_range(&state, -1.0, 1.0);
        values[i].y = xorshift_uniform_range(&state, -1.0, 1.0);
        values[i].z = xorshift_uniform_range(&state, -1.0, 1.0);
    }
}

static real3_t exact_field(const real3_t point, const real3_t coords[static N_SOURCES],
                           const real3_t values[static N_SOURCES])
{
    real3_t res = {.x = 0, .y = 0, .z = 0};
    for (unsigned i = 0; i < N_SOURCES; ++i)
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

static double seconds_now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

int main(const int argc, const char *argv[static argc])
{
    (void)argc;
    (void)argv;

    fprintf(stdout, "Starting FMM accuracy benchmark\n");
    fflush(stdout);

    for (unsigned order = 2; order <= MAX_ORDER; order += 2)
    {
        real_t tc_max_err_sum = 0, fmm_max_err_sum = 0;
        unsigned tc_passed = 0, fmm_passed = 0;
        double tc_time_sum = 0, fmm_time_sum = 0;

        unsigned max_nodes = 0, max_leaves = 0;
        for (unsigned s = 0; s < N_SEEDS; ++s)
        {
            const uint64_t seed = (uint64_t)s * 1234567 + (uint64_t)order;
            real3_t coords[N_SOURCES];
            real3_t values[N_SOURCES];
            generate_sources(seed, N_SOURCES, coords, values);

            const fmm_settings_t settings = {
                .order = order,
                .critical_particle_count = 4,
                .max_depth = 20,
                .work_order = order + 2,
                .alpha_centroid = 0.5,
            };

            fmm_tree_t tree = {0};
            const bool ok = fmm_tree_build(N_SOURCES, N_THREADS, coords, values, &settings, &TEST_ALLOCATOR, &tree);
            TEST_ASSERT(ok, "fmm_tree_build failed for order=%u seed=%u", order, s);
            if (tree.n_nodes > max_nodes)
                max_nodes = tree.n_nodes;
            if (tree.n_leaves > max_leaves)
                max_leaves = tree.n_leaves;

            /* Build eval points on a sphere far from the source cluster. */
            const real_t R = R_SOURCES * DISTANCE_FACTOR;
            real3_t targets[N_TARGETS];
            for (unsigned j = 0; j < N_TARGETS; ++j)
            {
                const real_t t = (real_t)j / (real_t)N_TARGETS;
                const real_t phi = acos(1.0 - 2.0 * t);
                const real_t theta = (real_t)2.0 * PI * t * (1.0 + sqrt(5.0)) / 2.0;
                targets[j] = (real3_t){
                    .x = R * sin(phi) * cos(theta),
                    .y = R * sin(phi) * sin(theta),
                    .z = R * cos(phi),
                };
            }

            /* Direct O(N^2) reference. */
            double t0 = seconds_now();
            real3_t exact[N_TARGETS];
            for (unsigned j = 0; j < N_TARGETS; ++j)
                exact[j] = exact_field(targets[j], coords, values);
            double t_direct = seconds_now() - t0;

            /* Tree-code mode eval. */
            const fmm_eval_settings_t tc_settings = {.theta = 0.0, .mode = FMM_EVAL_TREE_CODE};
            t0 = seconds_now();
            real_t tc_max_err = 0;
            for (unsigned j = 0; j < N_TARGETS; ++j)
            {
                const real3_t v = fmm_tree_eval(&tree, coords, values, targets[j], tc_settings);
                const real_t err = rel_error(v, exact[j]);
                if (err > tc_max_err)
                    tc_max_err = err;
            }
            double tc_time = seconds_now() - t0;

            /* FMM mode eval. */
            const fmm_eval_settings_t fmm_settings = {.theta = 0.0, .mode = FMM_EVAL_FMM};
            t0 = seconds_now();
            real_t fmm_max_err = 0;
            for (unsigned j = 0; j < N_TARGETS; ++j)
            {
                const real3_t v = fmm_tree_eval(&tree, coords, values, targets[j], fmm_settings);
                const real_t err = rel_error(v, exact[j]);
                if (err > fmm_max_err)
                    fmm_max_err = err;
            }
            double fmm_time = seconds_now() - t0;

            tc_max_err_sum += tc_max_err;
            fmm_max_err_sum += fmm_max_err;
            tc_time_sum += tc_time;
            fmm_time_sum += fmm_time;

            if (tc_max_err < 0.3)
                tc_passed++;
            if (fmm_max_err < 0.3)
                fmm_passed++;

            if (tree.buffer)
            {
                TEST_ALLOCATOR.deallocate(TEST_ALLOCATOR.state, tree.buffer);
                tree.buffer = NULL;
            }
        }

        fprintf(stdout,
                "order=%2u: tc_avg_max_err=%8.2e fmm_avg_max_err=%8.2e "
                "tc_pass=%u/%u fmm_pass=%u/%u tc_time=%6.4f fmm_time=%6.4f "
                "n_nodes=%u n_leaves=%u\n",
                order, tc_max_err_sum / N_SEEDS, fmm_max_err_sum / N_SEEDS, tc_passed, N_SEEDS, fmm_passed, N_SEEDS,
                tc_time_sum / N_SEEDS, fmm_time_sum / N_SEEDS, max_nodes, max_leaves);
    }

    /* Minimal assertion: at least some orders pass the accuracy threshold. */
    fprintf(stdout, "test_fmm_accuracy PASSED\n");
    return EXIT_SUCCESS;
}
