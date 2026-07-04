/**
 * Approximation of atan2 function. We can afford some scuff in exchange for SPEED and POWER!
 */

#include "common.h"

/**
 * Compute an approximation of atan2(y, x) using a polynomial approximation.
 *
 * The approximation uses a lookup table of coefficients for different intervals
 * of the input values. The error in the approximation is below 6e-9 when compared
 * to NumPy's implementation of atan2.
 *
 * Tests on my own machine shows the following results for random inputs:
 * - glibc atan2: TBD
 * - approx_atan2: TBD
 *
 * @param y The y-coordinate of the point.
 * @param x The x-coordinate of the point.
 *
 * @return The approximate value of atan2(y, x) in radians.
 */
#pragma omp declare simd
double atan2_approx(double y, double x);

/**
 * Compute an approximation of atan(x) using a polynomial approximation.
 *
 * Tests on my own machine shows the following results for random inputs:
 * - glibc atan: TBD
 * - approx_atan: TBD
 *
 * @param x The input value.
 *
 * @return The approximate value of atan(x) in radians.
 */
#pragma omp declare simd
double atan_approx(double x);
