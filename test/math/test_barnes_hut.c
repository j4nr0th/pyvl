/** Test the Barnes-Hut tree.
 *
 * Two stages:
 *  1. Sizing/count pass — `barnes_hut_buffer_size` and `barnes_hut_tree_count`
 *     return positive values for valid input and reject invalid input.
 *  2. Insert pass — `barnes_hut_tree_insert` builds a tree and the
 *     root-level multipole agrees with direct 1/r² summation of all sources
 *     at a far-field evaluation point. Determinism across seeds is checked.
 */

#include "../../src/core/barnes_hut_tree.h"
#include "../test_common.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static const real_t PI = (real_t)M_PI;

enum
{
    N_SOURCES = 100,
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

typedef struct
{
    size_t total_alloc_bytes;
    size_t total_alloc_count;
    size_t total_free_bytes;
    size_t total_free_count;
    size_t total_realloc_count;
} myalloc_state_t;

void *my_alloc(void *s, size_t size)
{
    myalloc_state_t *st = (void *)s;
    st->total_alloc_bytes += size;
    st->total_alloc_count += 1;
    return malloc(size);
}
void my_free(void *s, void *p)
{
    if (p == NULL)
        return;
    myalloc_state_t *st = (void *)s;
    st->total_free_count += 1;
    free(p);
}
void *my_realloc(void *s, void *p, size_t new_size)
{
    myalloc_state_t *st = (void *)s;
    st->total_realloc_count += 1;
    return realloc(p, new_size);
}

int main(const int argc, const char *argv[static argc])
{
    (void)argc;
    (void)argv;

    /* ========== Stage 1: sizing/count ========== */

    {
        const barnes_hut_settings_t settings = {.order = TEST_ORDER,
                                                .critical_particle_count = 8,
                                                .max_depth = 20,
                                                .work_order = 0,
                                                .n_threads = TEST_N_THREADS};
        const size_t sz = barnes_hut_buffer_size(0, &settings);
        TEST_ASSERT(sz == 0, "buffer_size for n_sources=0 must be 0, got %zu", sz);
    }

    {
        const barnes_hut_settings_t settings = {
            .order = 0, .critical_particle_count = 8, .max_depth = 20, .work_order = 0, .n_threads = TEST_N_THREADS};
        const size_t sz = barnes_hut_buffer_size(10, &settings);
        TEST_ASSERT(sz == 0, "buffer_size for order=0 must be 0, got %zu", sz);
    }

    {
        const barnes_hut_settings_t settings = {.order = TEST_ORDER,
                                                .critical_particle_count = 8,
                                                .max_depth = 20,
                                                .work_order = 0,
                                                .n_threads = TEST_N_THREADS};
        const size_t sz = barnes_hut_buffer_size(100, &settings);
        TEST_ASSERT(sz > 0, "buffer_size for n=100 must be positive, got %zu", sz);
        printf("buffer_size(n=100, order=4) = %zu bytes\n", sz);
    }

    {
        /* n_threads == 0 must be rejected by buffer_size (and by every other
         * public entry point). */
        const barnes_hut_settings_t bad = {
            .order = TEST_ORDER, .critical_particle_count = 8, .max_depth = 20, .work_order = 0, .n_threads = 0};
        const size_t sz = barnes_hut_buffer_size(100, &bad);
        TEST_ASSERT(sz == 0, "buffer_size for n_threads=0 must be 0, got %zu", sz);

        const size_t scratch = barnes_hut_scratch_size(100, &bad);
        TEST_ASSERT(scratch == 0, "scratch_size for n_threads=0 must be 0, got %zu", scratch);
    }

    {
        const barnes_hut_settings_t settings = {.order = TEST_ORDER,
                                                .critical_particle_count = 8,
                                                .max_depth = 20,
                                                .work_order = 0,
                                                .n_threads = TEST_N_THREADS};
        const size_t scratch_sz = barnes_hut_scratch_size(N_SOURCES, &settings);
        TEST_ASSERT(scratch_sz > 0, "scratch_size for n=100 must be positive, got %zu", scratch_sz);
        printf("scratch_size(n=100, order=4, n_threads=%u) = %zu bytes\n", TEST_N_THREADS, scratch_sz);

        real3_t coords[N_SOURCES];
        real3_t values[N_SOURCES];
        generate_sources(0x12345ULL, N_SOURCES, coords, values);
        void *scratch = malloc(scratch_sz);
        TEST_ASSERT(scratch != NULL, "scratch malloc failed");
        size_t required = 0;
        const bool ok = barnes_hut_tree_count(N_SOURCES, coords, &settings, scratch, scratch_sz, NULL, &required);
        TEST_ASSERT(ok, "count must succeed for valid input");
        TEST_ASSERT(required > 0, "required buffer size must be positive, got %zu", required);
        printf("count(n=100, order=4) = %zu bytes\n", required);
        free(scratch);
    }

    {
        /* count must fail (and reset required) for n_sources=0. */
        const barnes_hut_settings_t settings = {.order = TEST_ORDER,
                                                .critical_particle_count = 8,
                                                .max_depth = 20,
                                                .work_order = 0,
                                                .n_threads = TEST_N_THREADS};
        const size_t scratch_sz = barnes_hut_scratch_size(N_SOURCES, &settings);
        void *scratch = malloc(scratch_sz);
        TEST_ASSERT(scratch != NULL, "scratch malloc failed");
        size_t required = 12345;
        const bool ok = barnes_hut_tree_count(0, NULL, &settings, scratch, scratch_sz, NULL, &required);
        TEST_ASSERT(!ok, "count must fail for n_sources=0");
        TEST_ASSERT(required == 0, "count must reset required to 0 on failure, got %zu", required);
        free(scratch);
    }

    /* Invalid-input error paths for insert. */
    {
        const barnes_hut_settings_t settings = {.order = TEST_ORDER,
                                                .critical_particle_count = 8,
                                                .max_depth = 20,
                                                .work_order = 0,
                                                .n_threads = TEST_N_THREADS};
        real3_t coords[N_SOURCES];
        real3_t values[N_SOURCES];
        generate_sources(0x12345ULL, N_SOURCES, coords, values);
        const size_t scratch_sz = barnes_hut_scratch_size(N_SOURCES, &settings);
        const size_t required = barnes_hut_buffer_size(N_SOURCES, &settings);
        void *scratch = malloc(scratch_sz);
        void *buffer = malloc(required);
        TEST_ASSERT(scratch && buffer, "scratch/buffer malloc failed");
        barnes_hut_tree_t tree;

        TEST_ASSERT(
            !barnes_hut_tree_insert(0, coords, values, &settings, scratch, scratch_sz, NULL, buffer, required, &tree),
            "insert must fail for n_sources=0");
        TEST_ASSERT(!barnes_hut_tree_insert(N_SOURCES, NULL, values, &settings, scratch, scratch_sz, NULL, buffer,
                                            required, &tree),
                    "insert must fail for NULL sources_coords");
        TEST_ASSERT(!barnes_hut_tree_insert(N_SOURCES, coords, NULL, &settings, scratch, scratch_sz, NULL, buffer,
                                            required, &tree),
                    "insert must fail for NULL sources_values");
        TEST_ASSERT(!barnes_hut_tree_insert(N_SOURCES, coords, values, NULL, scratch, scratch_sz, NULL, buffer,
                                            required, &tree),
                    "insert must fail for NULL settings");
        TEST_ASSERT(!barnes_hut_tree_insert(N_SOURCES, coords, values, &settings, NULL, scratch_sz, NULL, buffer,
                                            required, &tree),
                    "insert must fail for NULL scratch_buffer");
        TEST_ASSERT(
            !barnes_hut_tree_insert(N_SOURCES, coords, values, &settings, scratch, 0, NULL, buffer, required, &tree),
            "insert must fail for scratch_size=0");
        TEST_ASSERT(
            !barnes_hut_tree_insert(N_SOURCES, coords, values, &settings, scratch, 16, NULL, buffer, required, &tree),
            "insert must fail for too-small scratch");
        TEST_ASSERT(!barnes_hut_tree_insert(N_SOURCES, coords, values, &settings, scratch, scratch_sz, NULL, NULL,
                                            required, &tree),
                    "insert must fail for NULL buffer");
        TEST_ASSERT(
            !barnes_hut_tree_insert(N_SOURCES, coords, values, &settings, scratch, scratch_sz, NULL, buffer, 0, &tree),
            "insert must fail for buffer_size=0");
        TEST_ASSERT(
            !barnes_hut_tree_insert(N_SOURCES, coords, values, &settings, scratch, scratch_sz, NULL, buffer, 16, &tree),
            "insert must fail for too-small buffer");
        TEST_ASSERT(!barnes_hut_tree_insert(N_SOURCES, coords, values, &settings, scratch, scratch_sz, NULL, buffer,
                                            required, NULL),
                    "insert must fail for NULL out");

        free(scratch);
        free(buffer);
    }

    /* ========== Stage 2: insert pass ========== */

    const barnes_hut_settings_t settings = {.order = TEST_ORDER,
                                            .critical_particle_count = 8,
                                            .max_depth = 20,
                                            .work_order = 0,
                                            .n_threads = TEST_N_THREADS};

    /* Build a tree once and inspect it. */
    {
        /* Generate 10 tight clusters of 10 sources each, distributed
         * across a unit cube. Each cluster is a tight Gaussian cloud
         * around a random centre. The cluster size (10) is just above
         * the order-2 multipole threshold (3), so each cluster should
         * become a multipole leaf. */
        const unsigned N_CLUSTERS = 10;
        const unsigned PER_CLUSTER = N_SOURCES / N_CLUSTERS;
        real3_t coords[N_SOURCES];
        real3_t values[N_SOURCES];
        uint64_t s = 0xABCDEFULL;
        for (unsigned i = 0; i < N_SOURCES; ++i)
        {
            const unsigned c = i / PER_CLUSTER;
            const real_t cx = xorshift_uniform_range(&s, -1.0, 1.0);
            const real_t cy = xorshift_uniform_range(&s, -1.0, 1.0);
            const real_t cz = xorshift_uniform_range(&s, -1.0, 1.0);
            coords[i].x = cx + xorshift_uniform_range(&s, -0.02, 0.02);
            coords[i].y = cy + xorshift_uniform_range(&s, -0.02, 0.02);
            coords[i].z = cz + xorshift_uniform_range(&s, -0.02, 0.02);
            values[i].x = xorshift_uniform_range(&s, -1.0, 1.0);
            values[i].y = xorshift_uniform_range(&s, -1.0, 1.0);
            values[i].z = xorshift_uniform_range(&s, -1.0, 1.0);
            (void)c;
        }

        const barnes_hut_settings_t settings = {
            .order = 4, .critical_particle_count = 8, .max_depth = 20, .work_order = 0, .n_threads = TEST_N_THREADS};

        const size_t scratch_sz = barnes_hut_scratch_size(N_SOURCES, &settings);
        void *scratch = malloc(scratch_sz);
        TEST_ASSERT(scratch != NULL, "scratch malloc failed");
        fprintf(stderr, "MALLOC scratch=%p sz=%zu\n", scratch, scratch_sz);
        fflush(stderr);

        size_t required = 0;
        TEST_ASSERT(barnes_hut_tree_count(N_SOURCES, coords, &settings, scratch, scratch_sz, NULL, &required),
                    "count failed");
        TEST_ASSERT(required > 0, "required buffer size must be positive");

        void *buffer = malloc(required);
        TEST_ASSERT(buffer != NULL, "buffer malloc failed");
        fprintf(stderr, "MALLOC buffer=%p sz=%zu scratch=%p\n", buffer, required, scratch);
        fflush(stderr);

        barnes_hut_tree_t tree;
        const bool ok = barnes_hut_tree_insert(N_SOURCES, coords, values, &settings, scratch, scratch_sz, NULL, buffer,
                                               required, &tree);
        TEST_ASSERT(ok, "insert failed for valid input");
        TEST_ASSERT(tree.n_sources == N_SOURCES, "n_sources mismatch: %u", tree.n_sources);
        TEST_ASSERT(tree.n_nodes >= 1, "tree must have at least the root, got %u", tree.n_nodes);
        TEST_ASSERT(tree.n_nodes == tree.n_internal + tree.n_multipole_leaves + tree.n_particle_leaves,
                    "n_nodes != n_internal + n_multipole + n_particle: %u != %u + %u + %u", tree.n_nodes,
                    tree.n_internal, tree.n_multipole_leaves, tree.n_particle_leaves);
        TEST_ASSERT(tree.max_depth_reached <= settings.max_depth, "max_depth exceeds cap");
        printf("tree: n_nodes=%u (internal=%u, multipole=%u, particle=%u) max_depth=%u\n", tree.n_nodes,
               tree.n_internal, tree.n_multipole_leaves, tree.n_particle_leaves, tree.max_depth_reached);
        TEST_ASSERT(tree.n_multipole_leaves > 0, "expected at least one multipole leaf, got %u",
                    tree.n_multipole_leaves);

        /* Find a multipole leaf to evaluate. */
        uint32_t multipole_leaf = UINT32_MAX;
        for (uint32_t i = 0; i < tree.n_nodes; ++i)
        {
            if (tree.nodes[i].kind == BH_NODE_MULTIPOLE && tree.nodes[i].particle_count >= 5)
            {
                multipole_leaf = i;
                break;
            }
        }
        TEST_ASSERT(multipole_leaf != UINT32_MAX, "need at least one multipole leaf with >= 5 sources");

        /* Build a direct multipole from this leaf's particles for ground truth. */
        const multipole_t *mp = &tree.nodes[multipole_leaf].data.mp;
        const size_t n_coeffs = multipole_num_coeffs(settings.order);
        real_t direct_coeffs[3 * 70]; /* order=4 → 70 coeffs */
        real_t direct_cur[125];
        real_t direct_nxt[125];
        real3_t leaf_coords_static[64];
        real3_t leaf_values_static[64];
        TEST_ASSERT(tree.nodes[multipole_leaf].particle_count <= 64, "leaf too large for static test buffer");
        for (unsigned k = 0; k < tree.nodes[multipole_leaf].particle_count; ++k)
        {
            const unsigned src = tree.particle_order[tree.nodes[multipole_leaf].particle_begin + k];
            leaf_coords_static[k] = coords[src];
            leaf_values_static[k] = values[src];
        }
        multipole_t direct_mp;
        const bool direct_ok = multipole_create(settings.order, 3 * n_coeffs, direct_coeffs, mp->center,
                                                tree.nodes[multipole_leaf].particle_count, leaf_coords_static,
                                                leaf_values_static, direct_cur, direct_nxt, &direct_mp);
        TEST_ASSERT(direct_ok, "direct multipole_create failed");

        /* Far-field fidelity: evaluate the multipole at 10x the source radius. */
        const real_t DISTANCE_FACTOR = 10.0;
        const real_t r = 0.1 * DISTANCE_FACTOR;

        real_t max_err = 0.0;
        for (unsigned j = 0; j < N_EVAL; ++j)
        {
            const real_t t = (real_t)j / (real_t)N_EVAL;
            const real_t phi = acos(1.0 - 2.0 * t);
            const real_t theta = (real_t)2.0 * PI * t * ((real_t)1.0 + sqrt((real_t)5.0)) / (real_t)2.0;
            const real3_t point = {.x = r * sin(phi) * cos(theta), .y = r * sin(phi) * sin(theta), .z = r * cos(phi)};
            const real3_t direct = multipole_eval(&direct_mp, point);
            const real3_t approx = multipole_eval(mp, point);
            const real_t err = rel_error(direct, approx);
            if (err > max_err)
                max_err = err;
        }
        printf("multipole-leaf fidelity: max rel err = %.3e\n", max_err);
        TEST_ASSERT(max_err < 1e-12, "multipole-leaf should match direct construction: %.3e", max_err);

        free(scratch);
        free(buffer);
    }

    /* Determinism: same seed -> identical tree shape. */
    {
        real3_t coords_a[N_SOURCES];
        real3_t values_a[N_SOURCES];
        real3_t coords_b[N_SOURCES];
        real3_t values_b[N_SOURCES];
        generate_sources(0xABCDEFULL, N_SOURCES, coords_a, values_a);
        generate_sources(0xABCDEFULL, N_SOURCES, coords_b, values_b);

        const size_t scratch_sz = barnes_hut_scratch_size(N_SOURCES, &settings);
        void *scratch_a = malloc(scratch_sz);
        void *scratch_b = malloc(scratch_sz);
        TEST_ASSERT(scratch_a && scratch_b, "scratch malloc failed");
        size_t required_a = 0, required_b = 0;
        barnes_hut_tree_count(N_SOURCES, coords_a, &settings, scratch_a, scratch_sz, NULL, &required_a);
        barnes_hut_tree_count(N_SOURCES, coords_b, &settings, scratch_b, scratch_sz, NULL, &required_b);
        TEST_ASSERT(required_a == required_b, "count must be deterministic: %zu vs %zu", required_a, required_b);

        /* Build both trees and compare depth stats. */
        void *buffer_a = malloc(required_a);
        void *buffer_b = malloc(required_b);
        TEST_ASSERT(buffer_a && buffer_b, "malloc failed");
        barnes_hut_tree_t tree_a, tree_b;
        TEST_ASSERT(barnes_hut_tree_insert(N_SOURCES, coords_a, values_a, &settings, scratch_a, scratch_sz, NULL,
                                           buffer_a, required_a, &tree_a),
                    "insert A failed");
        TEST_ASSERT(barnes_hut_tree_insert(N_SOURCES, coords_b, values_b, &settings, scratch_b, scratch_sz, NULL,
                                           buffer_b, required_b, &tree_b),
                    "insert B failed");

        unsigned lo_a = UINT32_MAX, hi_a = 0;
        unsigned lo_b = UINT32_MAX, hi_b = 0;
        barnes_hut_tree_depth_stats(&tree_a, &lo_a, &hi_a);
        barnes_hut_tree_depth_stats(&tree_b, &lo_b, &hi_b);
        TEST_ASSERT(lo_a == lo_b, "depth_stats min mismatch: %u vs %u", lo_a, lo_b);
        TEST_ASSERT(hi_a == hi_b, "depth_stats max mismatch: %u vs %u", hi_a, hi_b);
        TEST_ASSERT(hi_a == tree_a.max_depth_reached, "depth_stats max must equal max_depth_reached: %u vs %u", hi_a,
                    tree_a.max_depth_reached);

        /* NULL tree: must zero out stats without crashing. */
        unsigned zlo = 99, zhi = 99;
        barnes_hut_tree_depth_stats(NULL, &zlo, &zhi);
        TEST_ASSERT(zlo == 0 && zhi == 0, "NULL tree must zero stats: got lo=%u hi=%u", zlo, zhi);

        /* NULL outputs: must not crash. */
        barnes_hut_tree_depth_stats(&tree_a, NULL, NULL);

        free(scratch_a);
        free(scratch_b);
        free(buffer_a);
        free(buffer_b);
    }

    /* Custom allocator: verify the allocator callbacks are still used for
     * residual allocations (topo_to_real, mp_slices — sized from the count
     * pass output rather than from inputs alone). */
    {
        myalloc_state_t state = {0, 0, 0, 0, 0};

        const allocator_t my_allocator = {
            .allocate = my_alloc,
            .deallocate = my_free,
            .reallocate = my_realloc,
            .state = &state,
        };

        real3_t coords[N_SOURCES];
        real3_t values[N_SOURCES];
        generate_sources(0xCAFEBABEULL, N_SOURCES, coords, values);

        const size_t scratch_sz = barnes_hut_scratch_size(N_SOURCES, &settings);
        void *scratch = malloc(scratch_sz);
        TEST_ASSERT(scratch != NULL, "scratch malloc failed");
        const size_t required = barnes_hut_buffer_size(N_SOURCES, &settings);
        void *buffer = malloc(required);
        TEST_ASSERT(buffer != NULL, "buffer malloc failed");
        barnes_hut_tree_t tree;
        TEST_ASSERT(barnes_hut_tree_insert(N_SOURCES, coords, values, &settings, scratch, scratch_sz, &my_allocator,
                                           buffer, required, &tree),
                    "insert with custom allocator failed");
        TEST_ASSERT(state.total_alloc_count > 0, "custom allocator was not used for any allocation");
        TEST_ASSERT(state.total_alloc_count == state.total_free_count,
                    "leak under custom allocator: %zu allocs vs %zu frees", state.total_alloc_count,
                    state.total_free_count);

        /* Reset counters, repeat with the convenience `build` entry point. */
        const size_t alloc_before = state.total_alloc_count;
        const size_t free_before = state.total_free_count;
        TEST_ASSERT(barnes_hut_tree_build(N_SOURCES, coords, values, &settings, scratch, scratch_sz, &my_allocator,
                                          buffer, required, &tree),
                    "build with custom allocator failed");
        TEST_ASSERT(state.total_alloc_count > alloc_before, "build did not exercise the allocator");
        TEST_ASSERT(state.total_alloc_count - alloc_before == state.total_free_count - free_before,
                    "build leaked scratch under custom allocator");

        /* count() routes only through the scratch buffer (no residual allocation
         * is needed for a count-only pass), so the allocator is untouched. */
        const size_t alloc_before_c = state.total_alloc_count;
        size_t required_b = 0;
        TEST_ASSERT(
            barnes_hut_tree_count(N_SOURCES, coords, &settings, scratch, scratch_sz, &my_allocator, &required_b),
            "count with custom allocator failed");
        TEST_ASSERT(state.total_alloc_count == alloc_before_c, "count unexpectedly went through the allocator");

        free(scratch);
        free(buffer);
    }

    printf("test_barnes_hut: OK\n");
    return EXIT_SUCCESS;
}
