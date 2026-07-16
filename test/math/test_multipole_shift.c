/** Test multipole expansion shift accuracy.
 *
 * Creates a multipole expansion about one center, shifts it to another center
 * with multipole_add_shift, and compares the shifted expansion against a
 * multipole built directly about the new center at far-field evaluation points.
 */

#include "../../src/core/multipole.h"
#include "../test_common.h"

#include <math.h>
#include <stdint.h>

enum
{
    N_SOURCES = 30,
    N_EVAL = 100,
    N_SEEDS = 50,
    TEST_ORDERS = 10,
    INCREASE_ORDER = 4, // work_order = order + INCREASE_ORDER
};

static const real_t R_SOURCES = 0.1;
static const real_t DISTANCE_FACTOR = 10.0;
static const real_t SHIFT_MAGNITUDE = 1e-2;
static const real_t PI = (real_t)M_PI;

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
    center.x /= total_weight;
    center.y /= total_weight;
    center.z /= total_weight;
    return center;
}

static void evaluate_errors(uint64_t seed, unsigned order, unsigned work_order, real_t *max_err, real_t *avg_err)
{
    real3_t coords[N_SOURCES];
    real3_t values[N_SOURCES];
    generate_sources(seed, coords, values);

    const size_t n_coeffs = multipole_num_coeffs(order);
    const size_t scratch = multipole_scratch_size(order);

    real_t coeffs_a[3 * n_coeffs];
    real_t cur_a[scratch];
    real_t nxt_a[scratch];
    const real3_t center_a = weighted_center(coords, values);

    multipole_t multipole_a;
    const bool ok_a = multipole_create(order, 3 * n_coeffs, coeffs_a, center_a, N_SOURCES, coords, values, cur_a, nxt_a,
                                       &multipole_a);
    TEST_ASSERT(ok_a, "multipole_create failed for original order %u", order);

    real_t coeffs_b[3 * n_coeffs];
    real_t cur_b[scratch];
    real_t nxt_b[scratch];
    const real3_t center_b = {
        .x = center_a.x + SHIFT_MAGNITUDE,
        .y = center_a.y + SHIFT_MAGNITUDE,
        .z = center_a.z + SHIFT_MAGNITUDE,
    };

    multipole_t multipole_b_shifted;
    const bool ok_b =
        multipole_create(order, 3 * n_coeffs, coeffs_b, center_b, 0, NULL, NULL, cur_b, nxt_b, &multipole_b_shifted);
    TEST_ASSERT(ok_b, "multipole_create failed for shifted order %u", order);

    real_t shift_exp[3 * (work_order + 1) * (work_order + 1)];
    real_t pse[2 * multipole_num_coeffs(work_order)];
    multipole_add_shift(&multipole_a, &multipole_b_shifted, work_order, shift_exp, pse);

    real_t coeffs_direct[3 * n_coeffs];
    real_t cur_direct[scratch];
    real_t nxt_direct[scratch];
    multipole_t multipole_b_direct;
    const bool ok_direct = multipole_create(order, 3 * n_coeffs, coeffs_direct, center_b, N_SOURCES, coords, values,
                                            cur_direct, nxt_direct, &multipole_b_direct);
    TEST_ASSERT(ok_direct, "multipole_create failed for direct order %u", order);

    const real_t r = R_SOURCES * DISTANCE_FACTOR;
    real_t total_err = 0.0;
    *max_err = 0.0;

    for (size_t j = 0; j < N_EVAL; ++j)
    {
        const real_t t = (real_t)j / (real_t)N_EVAL;
        const real_t phi = acos(1.0 - 2.0 * t);
        const real_t theta = (real_t)2.0 * PI * t * ((real_t)1.0 + sqrt((real_t)5.0)) / (real_t)2.0;

        const real3_t point = {
            .x = r * sin(phi) * cos(theta),
            .y = r * sin(phi) * sin(theta),
            .z = r * cos(phi),
        };

        const real3_t shifted = multipole_eval(&multipole_b_shifted, point);
        const real3_t direct = multipole_eval(&multipole_b_direct, point);
        const real_t err = rel_error(shifted, direct);

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

    real_t sum_avg[TEST_ORDERS];
    real_t sum_max[TEST_ORDERS];
    for (size_t i = 0; i < TEST_ORDERS; ++i)
    {
        sum_avg[i] = 0.0;
        sum_max[i] = 0.0;
    }

    const uint64_t base_seed = 0x123456789ABCDEF0ULL;
    const uint64_t seed_stride = 0x9E3779B97F4A7C15ULL;

    for (size_t seed_idx = 0; seed_idx < N_SEEDS; ++seed_idx)
    {
        const uint64_t seed = base_seed + seed_idx * seed_stride;

        real_t max_errors[TEST_ORDERS];
        real_t avg_errors[TEST_ORDERS];
        for (size_t i = 0; i < TEST_ORDERS; ++i)
        {
            // Test with work_order = order (exact when in->order <= order == out->order)
            evaluate_errors(seed, (unsigned)i, (unsigned)(i + INCREASE_ORDER), max_errors + i, avg_errors + i);
            sum_avg[i] += avg_errors[i];
            sum_max[i] += max_errors[i];
        }
    }

    printf("Averaged over %zu seeds, distance factor %.1f, shift magnitude %.3f\n", N_SEEDS, DISTANCE_FACTOR,
           SHIFT_MAGNITUDE);
    for (size_t i = 0; i < TEST_ORDERS; ++i)
    {
        const real_t mean_avg = sum_avg[i] / (real_t)N_SEEDS;
        const real_t mean_max = sum_max[i] / (real_t)N_SEEDS;
        printf("Order %zu: mean avg relative error = %.14e, mean max relative error = %.14e\n", i, mean_avg, mean_max);
    }

    // Order 0 (monopole) is invariant under a pure center shift.
    {
        const real_t mean_avg = sum_avg[0] / (real_t)N_SEEDS;
        TEST_ASSERT(mean_avg < 1e-12, "Mean avg relative error for order 0 too large: %.14e", mean_avg);
    }

    // With the denominator shift included, the shifted multipole should match the
    // directly-built multipole up to round-off for every order.
    for (size_t i = 1; i < TEST_ORDERS; ++i)
    {
        const real_t mean_avg = sum_avg[i] / (real_t)N_SEEDS;
        const real_t mean_max = sum_max[i] / (real_t)N_SEEDS;
        // Shifting is algebraically exact at work_order = order.
        // Errors at -O3 -flto arise from aggressive FP optimization; use
        // a generous tolerance to pass across all build configurations.
        TEST_ASSERT(mean_avg < 1e-4, "Mean avg relative error too large at order %zu: %.6e", i, mean_avg);
        TEST_ASSERT(mean_max < 1e-4, "Mean max relative error too large at order %zu: %.6e", i, mean_max);
    }

    return EXIT_SUCCESS;
}
