/** Benchmark the Barnes-Hut tree build (count + insert) at several scales.
 *
 * Wall-time the tree-build step for various N and report
 *   - tree shape (n_nodes, depth, multipole/particle leaves)
 *   - per-source build time (ns/source)
 *   - buffer memory footprint (KiB)
 *
 * Designed to be run as a CTest target. Skipped on small-N correctness: a
 * weak sanity check on tree shape is performed.
 */

#include "../../src/core/barnes_hut_tree.h"
#include "../test_common.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum
{
    N_LEVELS = 4
};

static const unsigned N_VALUES[N_LEVELS] = {1000u, 10000u, 100000u, 1000000u};
static const real_t CLUSTER_RADIUS[N_LEVELS] = {0.1, 0.05, 0.02, 0.01};
static const real_t DISTRIBUTION_R = 1.0;

static void generate_clustered(uint64_t seed, unsigned n, real_t cluster_r, real_t box_r, real3_t coords[static n],
                               real3_t values[static n])
{
    uint64_t state = seed;
    /* 10 clusters evenly spread across the box. */
    const unsigned n_clusters = 10;
    for (unsigned i = 0; i < n; ++i)
    {
        const unsigned c = i % n_clusters;
        const real_t cx = box_r * (((real_t)c / (real_t)n_clusters) * 2.0 - 1.0);
        const real_t cy = box_r * (((real_t)((c * 7u) % n_clusters) / (real_t)n_clusters) * 2.0 - 1.0);
        const real_t cz = box_r * (((real_t)((c * 13u) % n_clusters) / (real_t)n_clusters) * 2.0 - 1.0);
        coords[i].x = cx + xorshift_uniform_range(&state, -cluster_r, cluster_r);
        coords[i].y = cy + xorshift_uniform_range(&state, -cluster_r, cluster_r);
        coords[i].z = cz + xorshift_uniform_range(&state, -cluster_r, cluster_r);
        values[i].x = xorshift_uniform_range(&state, -1.0, 1.0);
        values[i].y = xorshift_uniform_range(&state, -1.0, 1.0);
        values[i].z = xorshift_uniform_range(&state, -1.0, 1.0);
    }
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

    const barnes_hut_settings_t settings = {
        .order = 4, .critical_particle_count = 16, .max_depth = 24, .work_order = 0};
    const unsigned n_threads = 1;

    printf("Barnes-Hut build performance\n");
    printf("============================\n");
    printf("order=%u critical=%u max_depth=%u\n", settings.order, settings.critical_particle_count, settings.max_depth);
    printf("%-8s %-14s %-8s %-9s %-12s %-10s %-9s %-10s\n", "N", "shape", "depth", "build_ms", "ns/source",
           "buffer_KiB", "n_threads", "seed");
    printf("-------- -------------- -------- --------- ------------ ---------- --------- "
           "----------\n");

    for (unsigned lvl = 0; lvl < N_LEVELS; ++lvl)
    {
        const unsigned n = N_VALUES[lvl];
        const uint64_t seed = 0xC0FFEEULL + (uint64_t)lvl;

        real3_t *coords = (real3_t *)malloc((size_t)n * sizeof(real3_t));
        real3_t *values = (real3_t *)malloc((size_t)n * sizeof(real3_t));
        if (!coords || !values)
        {
            free(coords);
            free(values);
            fprintf(stderr, "malloc failed for n=%u\n", n);
            return 1;
        }
        generate_clustered(seed, n, CLUSTER_RADIUS[lvl], DISTRIBUTION_R, coords, values);

        const size_t scratch_sz = barnes_hut_scratch_size(n, 1, &settings);
        const size_t required = barnes_hut_buffer_size(n, &settings);
        if (required == 0 || scratch_sz == 0)
        {
            fprintf(stderr, "sizing returned 0 for n=%u\n", n);
            free(coords);
            free(values);
            return 1;
        }

        void *scratch = malloc(scratch_sz);
        void *buffer = malloc(required);
        if (!scratch || !buffer)
        {
            fprintf(stderr, "malloc failed for n=%u (scratch=%zu, buffer=%zu)\n", n, scratch_sz, required);
            free(scratch);
            free(buffer);
            free(coords);
            free(values);
            return 1;
        }

        barnes_hut_tree_t tree;
        const double t0 = seconds_now();
        const bool ok =
            barnes_hut_tree_insert(n, 1, coords, values, &settings, scratch, scratch_sz, NULL, buffer, required, &tree);
        const double t1 = seconds_now();
        if (!ok)
        {
            fprintf(stderr, "insert failed for n=%u\n", n);
            free(scratch);
            free(buffer);
            free(coords);
            free(values);
            return 1;
        }

        const double build_ms = (t1 - t0) * 1e3;
        const double ns_per_source = (t1 - t0) * 1e9 / (double)n;
        const double buffer_kib = (double)required / 1024.0;
        const double scratch_kib = (double)scratch_sz / 1024.0;
        const double total_kib = buffer_kib + scratch_kib;

        printf("%-8u %4u %4u %4u   %-8u %-9.3f %-12.1f %-10.1f %-9u 0x%lx\n", n, tree.n_internal,
               tree.n_multipole_leaves, tree.n_particle_leaves, tree.max_depth_reached, build_ms, ns_per_source,
               total_kib, 1, (unsigned long)seed);

        /* Sanity checks: tree must have at least one multipole or particle leaf
         * and the sum of leaf kinds must equal total nodes. */
        if (tree.n_nodes == 0)
        {
            fprintf(stderr, "FAIL: n_nodes=0 for n=%u\n", n);
            free(scratch);
            free(buffer);
            free(coords);
            free(values);
            return 1;
        }
        if (tree.n_nodes != tree.n_internal + tree.n_multipole_leaves + tree.n_particle_leaves)
        {
            fprintf(stderr, "FAIL: n_nodes invariant broken for n=%u\n", n);
            free(scratch);
            free(buffer);
            free(coords);
            free(values);
            return 1;
        }

        free(scratch);
        free(buffer);
        free(coords);
        free(values);
    }

    printf("\ntest_barnes_hut_perf: OK\n");
    return 0;
}
