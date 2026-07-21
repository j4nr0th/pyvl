/**
 * @brief Runtime verification of the FLOP-count cost model.
 *
 * Checks that each `cost_model_*()` function returns the expected value
 * for key inputs (orders 0..6, direct-sum cross-over points).
 */

#include "../../src/core/cost_model.h"
#include "../test_common.h"

int main(void)
{
    // -- multipole_eval ----------------------------------------------------
    TEST_ASSERT(cost_model_multipole_eval(0) == 27, "O=0 → 27 FLOPs (got %zu)", cost_model_multipole_eval(0));
    TEST_ASSERT(cost_model_multipole_eval(1) == 67, "O=1 → 67 FLOPs (got %zu)", cost_model_multipole_eval(1));
    TEST_ASSERT(cost_model_multipole_eval(2) == 153, "O=2 → 153 FLOPs (got %zu)", cost_model_multipole_eval(2));
    TEST_ASSERT(cost_model_multipole_eval(3) == 314, "O=3 → 314 FLOPs (got %zu)", cost_model_multipole_eval(3));
    TEST_ASSERT(cost_model_multipole_eval(4) == 586, "O=4 → 586 FLOPs (got %zu)", cost_model_multipole_eval(4));
    TEST_ASSERT(cost_model_multipole_eval(5) == 1012, "O=5 → 1012 FLOPs (got %zu)", cost_model_multipole_eval(5));
    TEST_ASSERT(cost_model_multipole_eval(6) == 1642, "O=6 → 1642 FLOPs (got %zu)", cost_model_multipole_eval(6));

    // -- direct_sum --------------------------------------------------------
    TEST_ASSERT(cost_model_direct_sum(0) == 0, "0 sources → 0 FLOPs");
    TEST_ASSERT(cost_model_direct_sum(1) == 15, "1 source → 15 FLOPs");
    TEST_ASSERT(cost_model_direct_sum(10) == 150, "10 sources → 150 FLOPs");
    TEST_ASSERT(cost_model_direct_sum(100) == 1500, "100 sources → 1500 FLOPs");

    // -- crossover_order ---------------------------------------------------
    // 1 source: direct=15, O=0 costs 27 → multipole never beats direct sum.
    TEST_ASSERT(cost_model_crossover_order(1) == UINT_MAX, "1 source → direct sum always cheaper (got %u)",
                cost_model_crossover_order(1));

    // 2+ sources: even O=0 (27) beats direct sum (≥30).
    TEST_ASSERT(cost_model_crossover_order(2) == 0, "2 sources → O=0 wins (got %u)", cost_model_crossover_order(2));
    TEST_ASSERT(cost_model_crossover_order(10) == 0, "10 sources → O=0 wins (got %u)", cost_model_crossover_order(10));
    TEST_ASSERT(cost_model_crossover_order(100) == 0, "100 sources → O=0 wins (got %u)",
                cost_model_crossover_order(100));

    // -- min_sources_for_order (inverse of crossover) --------------------
    TEST_ASSERT(cost_model_min_sources_for_order(0) == 2, "O=0 → min 2 sources (got %u)",
                cost_model_min_sources_for_order(0));
    TEST_ASSERT(cost_model_min_sources_for_order(1) == 5, "O=1 → min 5 sources (got %u)",
                cost_model_min_sources_for_order(1));
    TEST_ASSERT(cost_model_min_sources_for_order(2) == 11, "O=2 → min 11 sources (got %u)",
                cost_model_min_sources_for_order(2));
    TEST_ASSERT(cost_model_min_sources_for_order(3) == 21, "O=3 → min 21 sources (got %u)",
                cost_model_min_sources_for_order(3));
    TEST_ASSERT(cost_model_min_sources_for_order(4) == 40, "O=4 → min 40 sources (got %u)",
                cost_model_min_sources_for_order(4));
    TEST_ASSERT(cost_model_min_sources_for_order(5) == 68, "O=5 → min 68 sources (got %u)",
                cost_model_min_sources_for_order(5));
    TEST_ASSERT(cost_model_min_sources_for_order(6) == 110, "O=6 → min 110 sources (got %u)",
                cost_model_min_sources_for_order(6));

    // -- boundary: at min_sources-1 multipole is NOT cheaper, at min_sources it IS --
    for (unsigned o = 0; o <= 10; ++o)
    {
        const unsigned n_min = cost_model_min_sources_for_order(o);
        if (n_min > 1)
        {
            // At n_min-1 direct sum should be ≤ multipole cost.
            TEST_ASSERT(cost_model_direct_sum(n_min - 1) <= cost_model_multipole_eval(o),
                        "O=%u: at %u sources direct sum (%zu) should NOT beat multipole (%zu)", o, n_min - 1,
                        cost_model_direct_sum(n_min - 1), cost_model_multipole_eval(o));
        }
        // At n_min direct sum must be > multipole cost.
        TEST_ASSERT(cost_model_direct_sum(n_min) > cost_model_multipole_eval(o),
                    "O=%u: at %u sources direct sum (%zu) must beat multipole (%zu)", o, n_min,
                    cost_model_direct_sum(n_min), cost_model_multipole_eval(o));
    }

    // -- monotonicity: min_sources increases with order ------------------
    for (unsigned o = 1; o <= 10; ++o)
    {
        TEST_ASSERT(cost_model_min_sources_for_order(o) > cost_model_min_sources_for_order(o - 1),
                    "min_sources must increase with order (O=%u=%u < O-1=%u)", o, cost_model_min_sources_for_order(o),
                    cost_model_min_sources_for_order(o - 1));
    }

    fprintf(stdout, "All cost-model checks passed.\n");
    return 0;
}
