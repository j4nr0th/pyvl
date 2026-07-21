#ifndef CVL_COST_MODEL_H
#define CVL_COST_MODEL_H

#include "common.h"

#include <limits.h>

/**
 * @brief Binomial coefficient C(n, k) for k <= 4.
 *
 * Multiplicative formula, integer arithmetic only.  Used internally by the
 * cost-model helper functions for n up to ~15 (C(19,4) = 3876 fits easily in
 * size_t; C(19,4) ≈ 3.9e3, C(15+4,4) ≈ 3.9e3, safe up to order ~15).
 */
static inline size_t cost_model_binom(unsigned n, unsigned k)
{
    switch (k)
    {
    case 0:
        return 1;
    case 1:
        return n;
    case 2:
        return (size_t)n * (n - 1) / 2;
    case 3:
        return (size_t)n * (n - 1) * (n - 2) / 6;
    case 4:
        return (size_t)n * (n - 1) * (n - 2) * (n - 3) / 24;
    default:
        return 0;
    }
}

/**
 * @brief Total floating-point operations for one `multipole_eval` call at given order.
 *
 * Counts every mul, add, sub, sqrt, and div in `multipole_eval`.
 * Breakdown per operation type (validated against trace for O=0..6):
 *
 *   mul  = 4*C(O+4,4) + C(O+3,3) + C(O+2,2) + 4*(O+1) + 4
 *   add  = 3*C(O+4,4) + 3*(O+1) + 2
 *   sub  = 3
 *   sqrt + div = 2
 *
 * @param order  Multipole expansion order (0 <= order <= ~12).
 * @return Total FLOP count (preamble + all monomial loops + aggregation).
 */
static inline size_t cost_model_multipole_eval(unsigned order)
{
    const size_t c4 = cost_model_binom(order + 4, 4);
    const size_t c3 = cost_model_binom(order + 3, 3);
    const size_t c2 = cost_model_binom(order + 2, 2);
    const size_t o1 = (size_t)order + 1;

    const size_t mul = 4 * c4 + c3 + c2 + 4 * o1 + 4;
    const size_t add = 3 * c4 + 3 * o1 + 2;
    const size_t sub = 3;
    const size_t sqrt_div = 2;

    return mul + add + sub + sqrt_div;
}

/**
 * @brief Total floating-point operations for a direct-sum over N sources.
 *
 * Each source uses the `particle_kernel` (Γ/|r|²):
 *   3 sub (rel_vec) + 3 mul (dot) + 2 add (dot) + 1 div (1/r²) +
 *   3 mul (Γ * inv_r²) + 3 add (accumulate)
 *   = 15 FLOPs.
 *
 * @param n_sources  Number of source particles.
 * @return Total FLOP count (15 × n_sources).
 */
static inline size_t cost_model_direct_sum(unsigned n_sources)
{
    return 15 * (size_t)n_sources;
}

/**
 * @brief Smallest multipole order whose eval cost is below the direct-sum cost.
 *
 * Returns the minimum @p order such that
 * `cost_model_multipole_eval(order) < cost_model_direct_sum(n_sources)`.
 *
 * If no order ≤ 15 is cheaper (e.g. n_sources = 1), returns `UINT_MAX`
 * to signal that the direct sum is always less expensive.
 *
 * @param n_sources  Number of source particles.
 * @return Minimum order that beats the direct sum, or `UINT_MAX` if none exists.
 */
static inline unsigned cost_model_crossover_order(unsigned n_sources)
{
    const size_t direct = cost_model_direct_sum(n_sources);
    for (unsigned order = 0; order <= 15; ++order)
    {
        if (cost_model_multipole_eval(order) < direct)
            return order;
    }
    return UINT_MAX;
}

/**
 * @brief Minimum number of sources for which multipole at @p order is cheaper
 *        than the direct sum.
 *
 * Inverse of @ref cost_model_crossover_order.  Returns the smallest N such that
 * `cost_model_direct_sum(N) > cost_model_multipole_eval(order)`.
 *
 * @param order  Multipole expansion order.
 * @return Minimum source count N (≥ 1) where multipole beats direct sum.
 */
static inline unsigned cost_model_min_sources_for_order(unsigned order)
{
    const size_t eval_cost = cost_model_multipole_eval(order);
    // Need 15*N > eval_cost → N > eval_cost/15.
    return (unsigned)(eval_cost / 15) + 1;
}

#endif // CVL_COST_MODEL_H
