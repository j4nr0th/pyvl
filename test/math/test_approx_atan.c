/** Purpose of this test is to verify that error for approx_atan is below 1e-8 when compared to glibc implementation of
 * atan and to time its performance.
 *
 */
#include "../../src/core/approx_atan.h"
#include "../test_common.h"

#include <math.h>
#include <stdbool.h>
#include <time.h>

enum
{
    N_INTERVALS = 3,
};

static const double TEST_TOLERANCE = 1e-3;

double TEST_INTERVALS[N_INTERVALS][2] = {
    {0.0, 1.0},
    {1.0, 10.0},
    {10.0, 100.0},
};

struct timespec start_timer()
{
    struct timespec start_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);
    return start_time;
}

static double end_timer(const struct timespec start_time)
{
    struct timespec end_time;
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    end_time.tv_sec -= start_time.tv_sec;
    if (end_time.tv_nsec < start_time.tv_nsec)
    {
        end_time.tv_sec -= 1;
        end_time.tv_nsec += 1000000000L;
    }
    end_time.tv_nsec -= start_time.tv_nsec;
    return (double)end_time.tv_sec + (double)end_time.tv_nsec / 1e9;
}

int main(const int argc, const char *argv[static argc])
{
    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s <number of random tests>\n", argv[0]);
        return EXIT_FAILURE;
    }

    char *end_p;
    const unsigned long n_tests = strtoul(argv[1], &end_p, 10);
    if (*end_p != '\0')
    {
        fprintf(stderr, "Invalid number: %s\n", argv[1]);
        return EXIT_FAILURE;
    }

    double max_err = 0;
    double err_pos_x = 0;
    double err_pos_y = 0;

    for (unsigned quadrant_sign = 0; quadrant_sign < 4; ++quadrant_sign)
    {
        const bool negate_x = quadrant_sign & 1; // Quadrants II and III
        const bool negate_y = quadrant_sign & 2; // Quadrants III and IV
        srand(quadrant_sign);                    // Seed the random number generator for reproducibility
        for (unsigned interval_idx = 0; interval_idx < N_INTERVALS; ++interval_idx)
        {
            const double min_val = TEST_INTERVALS[interval_idx][0];
            const double max_val = TEST_INTERVALS[interval_idx][1];

            for (unsigned test_idx_1 = 0; test_idx_1 < n_tests; ++test_idx_1)
                for (unsigned test_idx_2 = 0; test_idx_2 < n_tests; ++test_idx_2)
                {
                    const double x = min_val + (max_val - min_val) * ((double)(test_idx_1 + 1) / (double)(n_tests + 1));
                    const double y = min_val + (max_val - min_val) * ((double)(test_idx_2 + 1) / (double)(n_tests + 1));

                    // Adjust signs based on the quadrant
                    double x_signed = x;
                    double y_signed = y;
                    if (negate_x)
                        x_signed = -x_signed; // Quadrants II and III
                    if (negate_y)
                        y_signed = -y_signed; // Quadrants III and IV

                    const double result_glibc = atan(y_signed / x_signed);
                    const double result_approx = atan_approx(y_signed / x_signed);

                    const double error = fabs(result_glibc - result_approx);
                    TEST_ASSERT(error < TEST_TOLERANCE,
                                "Error %e (glibc: %g, approx: %g) exceeds tolerance %e for inputs (y=%f, x=%f)", error,
                                result_glibc, result_approx, TEST_TOLERANCE, y_signed, x_signed);
                    if (error > max_err)
                    {
                        max_err = error;
                        err_pos_x = x_signed;
                        err_pos_y = y_signed;
                    }
                }
        }
    }

    const double approx_input =
        fabs(err_pos_x) > fabs(err_pos_y) ? fabs(err_pos_y / err_pos_x) : fabs(err_pos_x / err_pos_y);
    printf("Maximum error observed: %e at (y=%f, x=%f). Approximation got input %g\n", max_err, err_pos_y, err_pos_x,
           approx_input);

    // Now rerun the loops (without endpoints and checking) to time performance
    double r = 0;
    // clock_t time_baseline = clock();
    const struct timespec time_baseline = start_timer();
    for (unsigned quadrant_sign = 0; quadrant_sign < 4; ++quadrant_sign)
    {
        const bool negate_x = quadrant_sign & 1; // Quadrants II and III
        const bool negate_y = quadrant_sign & 2; // Quadrants III and IV
        srand(quadrant_sign);                    // Seed the random number generator for reproducibility
        for (unsigned interval_idx = 0; interval_idx < N_INTERVALS; ++interval_idx)
        {
            const double min_val = TEST_INTERVALS[interval_idx][0];
            const double max_val = TEST_INTERVALS[interval_idx][1];

            for (unsigned test_idx_1 = 0; test_idx_1 < n_tests; ++test_idx_1)
                for (unsigned test_idx_2 = 0; test_idx_2 < n_tests; ++test_idx_2)
                {
                    const double x = min_val + (max_val - min_val) * ((double)(test_idx_1 + 1) / (double)(n_tests + 1));
                    const double y = min_val + (max_val - min_val) * ((double)(test_idx_2 + 1) / (double)(n_tests + 1));

                    // Adjust signs based on the quadrant
                    double x_signed = x;
                    double y_signed = y;
                    if (negate_x)
                        x_signed = -x_signed; // Quadrants II and III
                    if (negate_y)
                        y_signed = -y_signed; // Quadrants III and IV

                    // Something to do
                    const double v = y_signed + x_signed;
                    if (r < v)
                        r = v;
                }
        }
    }
    // time_baseline = clock() - time_baseline;
    const double t_baseline = end_timer(time_baseline);

    // clock_t time_approx = clock();
    const struct timespec time_approx = start_timer();
    for (unsigned quadrant_sign = 0; quadrant_sign < 4; ++quadrant_sign)
    {
        const bool negate_x = quadrant_sign & 1; // Quadrants II and III
        const bool negate_y = quadrant_sign & 2; // Quadrants III and IV
        srand(quadrant_sign);                    // Seed the random number generator for reproducibility
        for (unsigned interval_idx = 0; interval_idx < N_INTERVALS; ++interval_idx)
        {
            const double min_val = TEST_INTERVALS[interval_idx][0];
            const double max_val = TEST_INTERVALS[interval_idx][1];

#pragma omp simd collapse(2)
            for (unsigned test_idx_1 = 0; test_idx_1 < n_tests; ++test_idx_1)
                for (unsigned test_idx_2 = 0; test_idx_2 < n_tests; ++test_idx_2)
                {
                    const double x = min_val + (max_val - min_val) * ((double)(test_idx_1) / (double)(n_tests - 1));
                    const double y = min_val + (max_val - min_val) * ((double)(test_idx_2) / (double)(n_tests - 1));

                    // Adjust signs based on the quadrant
                    double x_signed = x;
                    double y_signed = y;
                    if (negate_x)
                        x_signed = -x_signed; // Quadrants II and III
                    if (negate_y)
                        y_signed = -y_signed; // Quadrants III and IV

                    const double v = atan_approx(y_signed / x_signed);
                    // Something to do
                    if (r < v)
                        r = v;
                }
        }
    }
    const double t_approx = end_timer(time_approx) - t_baseline;
    printf("Time taken for approx_atan: %f seconds for %lu iterations\n", t_approx, n_tests * 4 * N_INTERVALS);
    const struct timespec time_glibc = start_timer();
    r = 0;
    for (unsigned quadrant_sign = 0; quadrant_sign < 4; ++quadrant_sign)
    {
        const bool negate_x = quadrant_sign & 1; // Quadrants II and III
        const bool negate_y = quadrant_sign & 2; // Quadrants III and IV
        srand(quadrant_sign);                    // Seed the random number generator for reproducibility
#pragma omp simd collapse(2)
        for (unsigned interval_idx = 0; interval_idx < N_INTERVALS; ++interval_idx)
        {
            const double min_val = TEST_INTERVALS[interval_idx][0];
            const double max_val = TEST_INTERVALS[interval_idx][1];

            for (unsigned test_idx_1 = 0; test_idx_1 < n_tests; ++test_idx_1)
                for (unsigned test_idx_2 = 0; test_idx_2 < n_tests; ++test_idx_2)
                {
                    const double x = min_val + (max_val - min_val) * ((double)(test_idx_1 + 1) / (double)(n_tests + 1));
                    const double y = min_val + (max_val - min_val) * ((double)(test_idx_2 + 1) / (double)(n_tests + 1));

                    // Adjust signs based on the quadrant
                    double x_signed = x;
                    double y_signed = y;
                    if (negate_x)
                        x_signed = -x_signed; // Quadrants II and III
                    if (negate_y)
                        y_signed = -y_signed; // Quadrants III and IV

                    const double v = atan(y_signed / x_signed);
                    // Something to do
                    if (r < v)
                        r = v;
                }
        }
    }

    const double t_glibc = end_timer(time_glibc) - t_baseline;
    printf("Time taken for glibc atan: %f seconds for %lu iterations (%g)\n", t_glibc, n_tests * 4 * N_INTERVALS, r);

    return 0;
}
