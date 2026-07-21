#pragma once

#include "common.h"

/** @brief Structure representing a multipole expansion. */
typedef struct
{
    unsigned order;
    real3_t center;
    real_t *restrict coeffs_x;
    real_t *restrict coeffs_y;
    real_t *restrict coeffs_z;
} multipole_t;

/**
 * @brief Returns the number of coefficients needed per vector component.
 *
 * The full expansion stores the polynomial (2 r·s - s·s)^m for each order m,
 * so the total count is C(order+4, 4).
 */
size_t multipole_num_coeffs(unsigned order);

/**
 * @brief Returns the size of one temporary scratch buffer needed by multipole_create.
 *
 * The scratch is a flat array of at least (order + 1)^3 elements.
 */
size_t multipole_scratch_size(unsigned order);

/**
 * @brief Updates a multipole expansion with a new source point.
 *
 * This function adds contributions of a new source point to an existing multipole expansion.
 *
 * @param multipole The multipole expansion to update.
 * @param num_coeffs The number of coefficients in the multipole expansion (used only for checking input).
 * @param center The center of the multipole expansion.
 * @param source_pos The position of the new source point.
 * @param source_value The value of the new source point.
 * @param cur Zeroed work buffer of at least multipole_scratch_size(order) elements.
 * @param nxt Zeroed work buffer of at least multipole_scratch_size(order) elements.
 */
void multipole_update(const multipole_t *multipole, const real3_t center, const real3_t source_pos,
                      const real3_t source_value, real_t CVL_ARRAY_ARG(cur, restrict),
                      real_t CVL_ARRAY_ARG(nxt, restrict));

/**
 * @brief Adds a shifted multipole expansion to a new multipole expansion.
 *
 * @param in The multipole expansion which is shifted.
 * @param out The multipole expansion to which the shifted result is added to.
 * @param work_order Internal expansion order for the series.
 * @param shift_exp Work buffer.
 * @param pse Work buffer for polynomial series expansion.
 */
void multipole_add_shift(const multipole_t *in, const multipole_t *out, unsigned work_order,
                         real_t CVL_ARRAY_ARG(shift_exp, restrict), real_t CVL_ARRAY_ARG(pse, restrict));

/**
 * @brief Creates a multipole expansion from a set of source particles.
 *
 * @param order The order of the multipole expansion.
 * @param num_coeffs The number of coefficients in the multipole expansion (used only for checking input).
 * @param coeffs Array to write the coefficients of the multipole expansion to.
 * @param center The center of the multipole expansion.
 * @param sources The number of source points.
 * @param sources_coords The coordinates of the source points.
 * @param sources_values The values of the source points.
 * @param cur Scratch buffer of at least multipole_scratch_size(order) elements.
 * @param nxt Scratch buffer of at least multipole_scratch_size(order) elements.
 * @param out Pointer to the multipole expansion structure to fill.
 * @return True if the multipole expansion was created successfully, false otherwise.
 */
bool multipole_create(unsigned order, unsigned num_coeffs, real_t CVL_ARRAY_ARG(coeffs, restrict num_coeffs),
                      const real3_t center, unsigned sources,
                      const real3_t CVL_ARRAY_ARG(sources_coords, restrict sources),
                      const real3_t CVL_ARRAY_ARG(sources_values, restrict sources),
                      real_t CVL_ARRAY_ARG(cur, restrict num_coeffs), real_t CVL_ARRAY_ARG(nxt, restrict num_coeffs),
                      multipole_t *out);

/**
 * @brief Evaluates the multipole expansion at a given point.
 *
 * @param multipole The multipole expansion to evaluate.
 * @param point The point at which to evaluate the multipole expansion. It should be relative to center of the multipole
 * expansion.
 * @return The vector value of the multipole expansion at the given point.
 */
// #pragma omp declare simd
real3_t multipole_eval(const multipole_t *multipole, const real3_t point);
