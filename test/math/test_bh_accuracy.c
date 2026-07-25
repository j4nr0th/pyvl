/** Barnes-Hut accuracy & performance microbenchmark.
 *
 * Builds trees at several (order, work_order) combos for clustered sources
 * at 1K, 10K, 100K scale. Reports:
 *   - tree shape (nodes, depth, leaf counts)
 *   - build wall time (ms)
 *   - batch eval wall time (ms) at 500 far-field targets
 *   - L2 and max relative error vs direct O(N²)
 *   - speedup vs direct
 *
 * Always returns EXIT_SUCCESS — data-collection target.
 */

#include "../../src/core/barnes_hut_tree.h"
#include "../../src/core/cost_model.h"
#include "../test_common.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ================================================================== */
/* Configuration                                                      */
/* ================================================================== */

enum
{
    N_TARGETS = 500,
    N_THREADS = 6,
    N_REPEAT_BUILD = 2,
    N_REPEAT_EVAL = 3,
    MAX_N_DIRECT = 5000, /* skip direct sum when n_src > this */
    RNG_SEED_BASE = 0xB16B00B5,
};

static const unsigned ORDERS[] = {2, 4, 6, 8};
static const unsigned N_ORDERS = 4;

static const unsigned N_SRCS[] = {1000, 10000, 100000};
static const unsigned N_SRC_LEVELS = 3;

static const unsigned N_CLUSTERS[] = {10, 25, 50};
static const real_t CLUSTER_RADIUS = 0.05;

static const real_t BOX_R = 0.8;

/* ================================================================== */
/* Helpers                                                             */
/* ================================================================== */

static double seconds_now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static real_t rel_err_norm(const real3_t a, const real3_t b)
{
    const real3_t d = real3_sub(a, b);
    const real_t denom = (real3_mag(a) + real3_mag(b)) * 0.5;
    if (denom < 1e-300)
        return 0.0;
    return real3_mag(d) / denom;
}

static void generate_clustered(uint64_t seed, unsigned n, unsigned n_clusters, real3_t *coords, real3_t *values)
{
    uint64_t state = seed;
    const unsigned per_cluster = (n + n_clusters - 1) / n_clusters;
    for (unsigned i = 0; i < n; ++i)
    {
        const unsigned c = i / per_cluster;
        const real_t cx = BOX_R * (((real_t)c / (real_t)n_clusters) * 2.0 - 1.0);
        const real_t cy = BOX_R * (((real_t)((c * 7u) % n_clusters) / (real_t)n_clusters) * 2.0 - 1.0);
        const real_t cz = BOX_R * (((real_t)((c * 13u) % n_clusters) / (real_t)n_clusters) * 2.0 - 1.0);
        coords[i].x = cx + xorshift_uniform_range(&state, -CLUSTER_RADIUS, CLUSTER_RADIUS);
        coords[i].y = cy + xorshift_uniform_range(&state, -CLUSTER_RADIUS, CLUSTER_RADIUS);
        coords[i].z = cz + xorshift_uniform_range(&state, -CLUSTER_RADIUS, CLUSTER_RADIUS);
        values[i].x = xorshift_uniform_range(&state, -1.0, 1.0);
        values[i].y = xorshift_uniform_range(&state, -1.0, 1.0);
        values[i].z = xorshift_uniform_range(&state, -1.0, 1.0);
    }
}

static void generate_far_field_targets(uint64_t seed, unsigned n, real3_t *targets)
{
    uint64_t state = seed;
    for (unsigned i = 0; i < n; ++i)
    {
        const real_t theta = xorshift_uniform_range(&state, 0, 2.0 * (real_t)M_PI);
        const real_t phi = (real_t)acos(xorshift_uniform_range(&state, -1.0, 1.0));
        const real_t r = xorshift_uniform_range(&state, 5.0, 10.0);
        targets[i].x = r * sin(phi) * cos(theta);
        targets[i].y = r * sin(phi) * sin(theta);
        targets[i].z = r * cos(phi);
    }
}

/** Direct O(N²) induction at a single point. */
static real3_t direct_field(real3_t point, const real3_t *coords, const real3_t *values, unsigned n)
{
    real3_t res = {.x = 0, .y = 0, .z = 0};
    for (unsigned i = 0; i < n; ++i)
    {
        const real3_t dr = real3_sub(point, coords[i]);
        const real_t r2 = real3_dot(dr, dr);
        if (r2 < 1e-300)
            continue;
        res = real3_add(res, real3_mul1(values[i], 1.0 / r2));
    }
    return res;
}

/* ================================================================== */
/* Main                                                                */
/* ================================================================== */

int main(void)
{
    printf("bh_accuracy: accuracy and performance microbenchmark\n");
    printf("===================================================\n");
    printf("orders: ");
    for (unsigned oi = 0; oi < N_ORDERS; ++oi)
        printf("%u ", ORDERS[oi]);
    printf("\nsource levels: ");
    for (unsigned si = 0; si < N_SRC_LEVELS; ++si)
        printf("%u ", N_SRCS[si]);
    printf("\nn_threads=%u n_targets=%u n_repeat_build=%u n_repeat_eval=%u\n\n", N_THREADS, N_TARGETS, N_REPEAT_BUILD,
           N_REPEAT_EVAL);

    /* Header row */
    printf("%-10s %-4s %-4s %-5s %-6s %-6s %-6s %-6s ", "N", "ord", "wo", "theta", "nodes", "int", "mp", "ptcl");
    printf("%-6s %-8s %-9s %-9s ", "depth", "bld_ms", "eval_ms", "ev_ns/pt");
    printf("%-10s %-10s %-9s\n", "l2_err", "max_err", "spdup");

    printf("---------- ---- ---- ----- ------ ------ ------ ------ ------ -------- --------- --------- ");
    printf("---------- ---------- ---------\n");

    for (unsigned si = 0; si < N_SRC_LEVELS; ++si)
    {
        const unsigned n = N_SRCS[si];
        const unsigned n_clusters = N_CLUSTERS[si];
        const uint64_t seed = RNG_SEED_BASE + (uint64_t)si;

        real3_t *coords = (real3_t *)malloc((size_t)n * sizeof(real3_t));
        real3_t *values = (real3_t *)malloc((size_t)n * sizeof(real3_t));
        if (!coords || !values)
        {
            fprintf(stderr, "malloc(%u sources) failed\n", n);
            free(coords);
            free(values);
            return EXIT_FAILURE;
        }
        generate_clustered(seed, n, n_clusters, coords, values);

        real3_t targets[N_TARGETS];
        generate_far_field_targets(seed + 0x1000, N_TARGETS, targets);

        /* Exact reference (skip if N too large) */
        const bool do_exact = n <= MAX_N_DIRECT;
        real3_t v_exact[N_TARGETS];
        if (do_exact)
        {
            for (unsigned ti = 0; ti < N_TARGETS; ++ti)
                v_exact[ti] = direct_field(targets[ti], coords, values, n);
        }

        for (unsigned oi = 0; oi < N_ORDERS; ++oi)
        {
            const unsigned order = ORDERS[oi];

            /* Work-order sweep: default (0 = order) and order+2 */
            const unsigned work_orders[] = {0, order + 2};
            const unsigned n_wo = 2;

            for (unsigned wi = 0; wi < n_wo; ++wi)
            {
                const unsigned wo = work_orders[wi];
                const unsigned eff_wo = wo ? wo : order;

                const barnes_hut_settings_t settings = {.order = order,
                                                        .critical_particle_count =
                                                            cost_model_min_sources_for_order(order),
                                                        .max_depth = 20,
                                                        .work_order = wo,
                                                        .alpha_centroid = 0.0};

                /* Build (repeat for timing) */
                barnes_hut_tree_t tree;
                double build_ms = 0.0;
                for (unsigned ri = 0; ri < N_REPEAT_BUILD; ++ri)
                {
                    const double t0 = seconds_now();
                    const bool ok = barnes_hut_tree_build(n, N_THREADS, coords, values, &settings, NULL, &tree);
                    const double t1 = seconds_now();
                    if (!ok)
                    {
                        fprintf(stderr, "build failed: N=%u order=%u wo=%u\n", n, order, wo);
                        free(coords);
                        free(values);
                        return EXIT_FAILURE;
                    }
                    build_ms += (t1 - t0) * 1e3;
                }
                build_ms /= (double)N_REPEAT_BUILD;

                /* Eval (repeat for timing) */
                real3_t v_bh[N_TARGETS];
                double eval_ms = 0.0;
                for (unsigned ri = 0; ri < N_REPEAT_EVAL; ++ri)
                {
                    const double t0 = seconds_now();
                    barnes_hut_tree_eval_all(&tree, coords, values, N_TARGETS, targets, v_bh,
                                             BARNES_HUT_EVAL_SETTINGS_DEFAULT, N_THREADS);
                    const double t1 = seconds_now();
                    eval_ms += (t1 - t0) * 1e3;
                }
                eval_ms /= (double)N_REPEAT_EVAL;
                const double ns_per_target = eval_ms * 1e6 / (double)N_TARGETS;

                /* Error against exact (if available) */
                real_t l2_err = -1.0, max_err = -1.0;
                if (do_exact)
                {
                    real_t sum_sq_err = 0.0, sum_sq_ref = 0.0;
                    max_err = 0.0;
                    for (unsigned ti = 0; ti < N_TARGETS; ++ti)
                    {
                        const real3_t d = real3_sub(v_bh[ti], v_exact[ti]);
                        const real_t e = real3_mag(d);
                        const real_t r = real3_mag(v_exact[ti]);
                        sum_sq_err += e * e;
                        sum_sq_ref += r * r;
                        const real_t re = e / (r > 1e-300 ? r : 1e-300);
                        if (re > max_err)
                            max_err = re;
                    }
                    l2_err = sqrt(sum_sq_err / (sum_sq_ref > 1e-300 ? sum_sq_ref : 1e-300));
                }

                /* Speedup vs direct (only when we have exact timing) */
                double speedup = 0.0;
                if (do_exact)
                {
                    double direct_ms = 0.0;
                    const double t0 = seconds_now();
                    for (unsigned ti = 0; ti < N_TARGETS; ++ti)
                        v_exact[ti] = direct_field(targets[ti], coords, values, n);
                    const double t1 = seconds_now();
                    direct_ms = (t1 - t0) * 1e3;
                    speedup = direct_ms / (eval_ms > 1e-9 ? eval_ms : 1e-9);
                }

                printf("%-10u %-4u %-4u %-5.2f %-6u %-6u %-6u %-6u ", n, order, eff_wo, 0.0, tree.n_nodes,
                       tree.n_internal, tree.n_multipole_leaves, tree.n_particle_leaves);
                printf("%-6u %-8.2f %-9.3f %-9.1f ", tree.max_depth_reached, build_ms, eval_ms, ns_per_target);
                if (do_exact)
                    printf("%-10.2e %-10.2e %-9.1f\n", l2_err, max_err, speedup);
                else
                    printf("%-10s %-10s %-9s\n", "SKIP", "SKIP", "SKIP");

                free(tree.buffer);
            }
        }

        free(coords);
        free(values);
    }

    /* Opening-angle sensitivity: fix N=10k, order=4, wo=6, sweep theta */
    printf("\n--- Theta sensitivity (N=10000, order=4, wo=6) ---\n");
    printf("%-10s %-8s %-10s %-10s %-9s\n", "theta", "eval_ms", "l2_err", "max_err", "ev_ns/pt");
    printf("---------- -------- ---------- ---------- ---------\n");
    {
        const unsigned n = 10000;
        const unsigned n_clusters = 25;
        const uint64_t seed = RNG_SEED_BASE + 100;
        real3_t *coords = (real3_t *)malloc((size_t)n * sizeof(real3_t));
        real3_t *values = (real3_t *)malloc((size_t)n * sizeof(real3_t));
        generate_clustered(seed, n, n_clusters, coords, values);

        real3_t targets[N_TARGETS];
        generate_far_field_targets(seed + 0x2000, N_TARGETS, targets);

        real3_t v_exact_theta[N_TARGETS];
        for (unsigned ti = 0; ti < N_TARGETS; ++ti)
            v_exact_theta[ti] = direct_field(targets[ti], coords, values, n);

        const barnes_hut_settings_t settings = {.order = 4,
                                                .critical_particle_count = cost_model_min_sources_for_order(4),
                                                .max_depth = 20,
                                                .work_order = 6,
                                                .alpha_centroid = 0.0};
        barnes_hut_tree_t tree;
        barnes_hut_tree_build(n, N_THREADS, coords, values, &settings, NULL, &tree);

        const double thetas[] = {0.0, 1e-9, 1e-6, 1e-4, 0.01, 0.03, 0.1, 0.3, 0.5, 1.0};
        const unsigned n_thetas = 10;
        for (unsigned ti = 0; ti < n_thetas; ++ti)
        {
            const barnes_hut_eval_settings_t cfg = {.theta = thetas[ti]};
            real3_t v[N_TARGETS];
            const double t0 = seconds_now();
            for (unsigned ri = 0; ri < N_REPEAT_EVAL; ++ri)
                barnes_hut_tree_eval_all(&tree, coords, values, N_TARGETS, targets, v, cfg, N_THREADS);
            const double t1 = seconds_now();
            const double ems = (t1 - t0) * 1e3 / (double)N_REPEAT_EVAL;

            real_t l2 = 0.0, mx = 0.0;
            real_t ssq_e = 0.0, ssq_r = 0.0;
            for (unsigned j = 0; j < N_TARGETS; ++j)
            {
                const real3_t d = real3_sub(v[j], v_exact_theta[j]);
                const real_t e = real3_mag(d);
                const real_t r = real3_mag(v_exact_theta[j]);
                ssq_e += e * e;
                ssq_r += r * r;
                const real_t re = e / (r > 1e-300 ? r : 1e-300);
                if (re > mx)
                    mx = re;
            }
            l2 = sqrt(ssq_e / (ssq_r > 1e-300 ? ssq_r : 1e-300));
            printf("%-10.6g %-8.3f %-10.2e %-10.2e %-9.1f\n", thetas[ti], ems, l2, mx, ems * 1e6 / (double)N_TARGETS);
        }

        free(tree.buffer);
        free(coords);
        free(values);
    }

    printf("\ntest_bh_accuracy: OK\n");
    return EXIT_SUCCESS;
}
