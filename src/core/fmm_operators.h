#pragma once

/*
 * Local expansion operators for the Fast Multipole Method.
 *
 * The pyvl multipole library expands the kernel 1/|r-s|^2 as a Cartesian
 * geometric series.  A *local expansion* is the dual of a multipole
 * expansion: instead of expanding about the source cluster centre (valid
 * far from the cluster), it expands about the *target* region centre
 * (valid near the target, far from the source cluster).
 *
 * For a source cluster centred at S and a local expansion centred at R,
 * with r' = r - R (target relative to local centre) and R' = R - S:
 *
 *   1/|r-s|^2 = (1/|R-S|^2) * sum_{m=0}^{order} u^m
 *
 * where  u = (2 R'·r' + r'^2) / |R-S|^2.
 *
 * The series converges when |r'| < |R-S|, i.e. the target lies inside the
 * local cell and the source cluster is well-separated.  For the standard
 * FMM V-list (|R-S| >= 3h, |r'| <= h) we have |u| <= 7/9 < 1.
 *
 * The local expansion stores, for each order m, the polynomial
 * (2 R'·r' + r'^2)^m as a polynomial in r' = (x,y,z), with the same
 * tetrahedral coefficient layout as multipole_t.  Three component arrays
 * (x, y, z) hold the vector-valued coefficients.
 *
 * Operators:
 *  - multipole_to_local   (M2L): convert a multipole at S into a local at R.
 *  - local_expansion_shift (L2L): shift a local expansion to a new centre.
 *  - local_expansion_eval  (L2P): evaluate a local expansion at a point.
 *  - particle_to_local     (P2L): build a local from a single source.
 */

#include "common.h"
#include "multipole.h"

/**
 * @brief Structure representing a local (Taylor) expansion.
 *
 * Dual of @ref multipole_t.  Coefficients use the same tetrahedral layout
 * as the multipole expansion (@ref multipole_coeff_index).
 */
typedef struct
{
    unsigned order;            /**< Expansion order. */
    real3_t center;            /**< Centre of the local expansion. */
    real_t *restrict coeffs_x; /**< x-component coefficients. */
    real_t *restrict coeffs_y; /**< y-component coefficients. */
    real_t *restrict coeffs_z; /**< z-component coefficients. */
} local_expansion_t;

/**
 * @brief Number of coefficients per vector component for a local expansion.
 *
 * Identical to @ref multipole_num_coeffs because the local expansion uses
 * the same tetrahedral polynomial layout.
 *
 * @param order Expansion order.
 * @return Number of coefficients per component.
 */
size_t local_expansion_num_coeffs(unsigned order);

/**
 * @brief Size of one scratch buffer needed by @ref multipole_to_local.
 *
 * The M2L operator uses a double-buffered polynomial scratch of
 * @c 2 * multipole_num_coeffs(work_order) elements, plus a binomial
 * expansion scratch of @c 3 * (work_order+1)^2 elements.  This returns
 * the larger of the two so a single buffer can be aliased.
 *
 * @param work_order Internal expansion order for the series.
 * @return Minimum scratch size in elements.
 */
size_t local_expansion_m2l_scratch_size(unsigned work_order);

/**
 * @brief Size of the binomial-expansion scratch needed by M2L and L2L.
 *
 * @param work_order Internal expansion order.
 * @return Scratch size in elements: @c 3 * (work_order+1)^2.
 */
size_t local_expansion_shift_exp_size(unsigned work_order);

/**
 * @brief Size of the double-buffered polynomial scratch needed by M2L and L2L.
 *
 * @param work_order Internal expansion order.
 * @return Scratch size in elements: @c 2 * multipole_num_coeffs(work_order).
 */
size_t local_expansion_pse_size(unsigned work_order);

/**
 * @brief Convert a multipole expansion into a local expansion (M2L).
 *
 * Given a multipole expansion of a source cluster centred at
 * @c in->center, produce the equivalent local expansion centred at
 * @c out->center and accumulate it into @p out.
 *
 * The shift vector is @f$ \mathbf{R}' = \mathbf{R} - \mathbf{S} @f$
 * (local centre minus source centre).  The denominator factor
 * @f$ 2\mathbf{R}'\cdot\mathbf{r}' + r'^2 @f$ is quadratic in
 * @f$ \mathbf{r}' @f$, so this operator multiplies by a quadratic form
 * at each binomial-series step (unlike @ref multipole_add_shift which
 * uses a linear form).
 *
 * @param in         Source multipole expansion (read-only).
 * @param out        Target local expansion (accumulated, not zeroed).
 * @param work_order Internal expansion order for the series
 *                   (>= max(in->order, out->order)).
 * @param shift_exp  Scratch buffer of @ref local_expansion_shift_exp_size elements.
 * @param pse        Scratch buffer of @ref local_expansion_pse_size elements.
 */
void multipole_to_local(const multipole_t *in, local_expansion_t *out, unsigned work_order,
                        real_t CVL_ARRAY_ARG(shift_exp, restrict), real_t CVL_ARRAY_ARG(pse, restrict));

/**
 * @brief Shift a local expansion to a new centre (L2L).
 *
 * Translates a local expansion from @c in->center to @c out->center,
 * accumulating into @p out.  The denominator factor
 * @f$ 2\mathbf{d}\cdot\mathbf{r}' + d^2 @f$ (where
 * @f$ \mathbf{d} = \mathbf{out.center} - \mathbf{in.center} @f$) is
 * linear in @f$ \mathbf{r}' @f$, so this reuses the linear polynomial
 * multiplier from the multipole library.
 *
 * @param in         Source local expansion (read-only).
 * @param out        Target local expansion (accumulated, not zeroed).
 * @param work_order Internal expansion order for the series
 *                   (>= max(in->order, out->order)).
 * @param shift_exp  Scratch buffer of @ref local_expansion_shift_exp_size elements.
 * @param pse        Scratch buffer of @ref local_expansion_pse_size elements.
 */
void local_expansion_shift(const local_expansion_t *in, local_expansion_t *out, unsigned work_order,
                           real_t CVL_ARRAY_ARG(shift_exp, restrict), real_t CVL_ARRAY_ARG(pse, restrict));

/**
 * @brief Evaluate a local expansion at a point (L2P).
 *
 * Computes @f$ \mathbf{V}(\mathbf{r}) = \sum_{m=0}^{\text{order}}
 * \frac{1}{|\mathbf{R}-\mathbf{S}|^{2(m+1)}} P_m(\mathbf{r}') @f$
 * where @f$ \mathbf{r}' = \mathbf{r} - \mathbf{center} @f$ and
 * @f$ P_m @f$ is the stored polynomial of order @c m.
 *
 * @param local The local expansion to evaluate.
 * @param point The evaluation point (absolute coordinates).
 * @return The vector value of the local expansion at @p point.
 */
real3_t local_expansion_eval(const local_expansion_t *local, const real3_t point);

/**
 * @brief Build a local expansion from a single source particle (P2L).
 *
 * Accumulates the contribution of one source at @p source_pos with
 * strength @p source_value into the local expansion @p out centred at
 * @c out->center.  This is the single-particle limit of M2L.
 *
 * @param out         Target local expansion (accumulated, not zeroed).
 * @param source_pos  Position of the source particle.
 * @param source_value Vector strength of the source particle.
 * @param cur         Scratch buffer of @ref multipole_scratch_size(out->order) elements.
 * @param nxt         Scratch buffer of @ref multipole_scratch_size(out->order) elements.
 */
void particle_to_local(local_expansion_t *out, const real3_t source_pos, const real3_t source_value,
                       real_t CVL_ARRAY_ARG(cur, restrict), real_t CVL_ARRAY_ARG(nxt, restrict));
