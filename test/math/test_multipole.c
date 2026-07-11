/** Test multipole expansion accuracy against direct summation.
 *
 * Generates random source particles with random vector strengths and compares
 * the multipole approximation to the exact component-wise 1/r^2 summation.
 */

#include "../../src/core/multipole.h"
#include "../test_common.h"

#include <math.h>
#include <stdint.h>

enum
{
    N_SOURCES = 10,
    N_EVAL = 100,
    N_SEEDS = 50,
};

static const real_t R_SOURCES = 0.1;
static const real_t DISTANCE_FACTOR = 50.0;
static const real_t PI = (real_t)M_PI;

static real3_t exact_field(const real3_t point, const real3_t sources_coords[static N_SOURCES],
                           const real3_t sources_values[static N_SOURCES])
{
    real3_t res = {.x = 0.0, .y = 0.0, .z = 0.0};
    for (size_t i = 0; i < N_SOURCES; ++i)
    {
        const real3_t dr = real3_sub(point, sources_coords[i]);
        const real_t inv_r2 = 1.0 / real3_dot(dr, dr);
        res = real3_add(res, real3_mul1(sources_values[i], inv_r2));
    }
    return res;
}

static real_t abs_error(const real3_t a, const real3_t b)
{
    const real3_t d = real3_sub(a, b);
    return real3_mag(d);
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

static void evaluate_errors(uint64_t seed, unsigned order, real_t *max_err, real_t *avg_err)
{
    real3_t coords[N_SOURCES];
    real3_t values[N_SOURCES];
    generate_sources(seed, coords, values);

    const size_t n_coeffs = multipole_num_coeffs(order);
    const size_t scratch = multipole_scratch_size(order);
    real_t coeffs[3 * n_coeffs];
    real_t cur[scratch];
    real_t nxt[scratch];
    const real3_t center = {.x = 0.0, .y = 0.0, .z = 0.0};
    multipole_t multipole;
    const bool ok =
        multipole_create(order, 3 * n_coeffs, coeffs, center, N_SOURCES, coords, values, cur, nxt, &multipole);
    TEST_ASSERT(ok, "multipole_create failed for order %u", order);

    const real_t r = R_SOURCES * DISTANCE_FACTOR;
    real_t total_err = 0.0;
    *max_err = 0.0;

    for (size_t j = 0; j < N_EVAL; ++j)
    {
        // Fibonacci sphere distribution for uniform sampling.
        const real_t t = (real_t)j / (real_t)N_EVAL;
        const real_t phi = acos(1.0 - 2.0 * t);
        const real_t theta = (real_t)2.0 * PI * t * ((real_t)1.0 + sqrt((real_t)5.0)) / (real_t)2.0;

        const real3_t point = {
            .x = r * sin(phi) * cos(theta),
            .y = r * sin(phi) * sin(theta),
            .z = r * cos(phi),
        };

        const real3_t exact = exact_field(point, coords, values);
        const real3_t approx = multipole_eval(&multipole, point);
        const real_t err = abs_error(exact, approx);

        if (err > *max_err)
            *max_err = err;
        total_err += err;
    }

    *avg_err = total_err / (real_t)N_EVAL;
}

int main(const int argc, const char *argv[static argc])
{
    (void)argc;
    (void)argv;

    const unsigned orders[] = {0, 1, 2, 3, 4, 5};
    const size_t n_orders = sizeof(orders) / sizeof(orders[0]);
    real_t sum_avg[n_orders];
    for (size_t i = 0; i < n_orders; ++i)
        sum_avg[i] = 0.0;

    const uint64_t base_seed = 0x123456789ABCDEF0ULL;
    const uint64_t seed_stride = 0x9E3779B97F4A7C15ULL;

    for (size_t seed_idx = 0; seed_idx < N_SEEDS; ++seed_idx)
    {
        const uint64_t seed = base_seed + seed_idx * seed_stride;

        real_t max_errors[n_orders];
        real_t avg_errors[n_orders];
        for (size_t i = 0; i < n_orders; ++i)
        {
            evaluate_errors(seed, orders[i], &max_errors[i], &avg_errors[i]);
            sum_avg[i] += avg_errors[i];
        }
    }

    printf("Averaged over %zu seeds, distance factor %.1f\n", N_SEEDS, DISTANCE_FACTOR);
    for (size_t i = 0; i < n_orders; ++i)
    {
        const real_t mean_avg = sum_avg[i] / (real_t)N_SEEDS;
        printf("Order %u: mean avg absolute error = %.6e\n", orders[i], mean_avg);
    }

    for (size_t i = 1; i < n_orders; ++i)
    {
        const real_t prev = sum_avg[i - 1] / (real_t)N_SEEDS;
        const real_t cur = sum_avg[i] / (real_t)N_SEEDS;
        TEST_ASSERT(cur < prev, "Mean avg error did not decrease from order %u (%.6e) to order %u (%.6e)",
                    orders[i - 1], prev, orders[i], cur);
    }

    return EXIT_SUCCESS;
}
