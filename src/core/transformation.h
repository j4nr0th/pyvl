//
// Created by jan on 29.11.2024.
//

#ifndef TRANSFORMATION_H
#define TRANSFORMATION_H

#include "common.h"

typedef struct
{
    real3_t angles; // Rotation angles around x, y, and z axis
    real3_t offset; // Offsets by x, y, and z axis
} transformation_t;

/**
 * @brief Merges transformations A(x) = T_A @ x + r_A and B(x) = T_B @ x + r_B into an equivalent transformation
 * C(x) such that C(x) = A(B(x)) = T_A @ (T_B @ x + r_B) + r_A = (T_A @ T_B) @ x + (r_A + T_A @ r_B).
 *
 * This is useful when this operation will be repeatedly applied, as it will only require same cost as only one
 * transformation after the overhead of merging.
 *
 * @param trans_a Transformation matrix of first transformation.
 * @param off_a Offset of the first transformation.
 * @param trans_b Transformation matrix of second transformation.
 * @param off_b Offset of the second transformation.
 * @param p_trans_out Pointer which receives the output transformation matrix.
 * @param p_off_out Pointer which receives the output offset.
 */
static inline void merge_transformations(const real3x3_t trans_a, const real3_t off_a, const real3x3_t trans_b,
                                         const real3_t off_b, real3x3_t *p_trans_out, real3_t *p_off_out)
{
    *p_trans_out = real3x3_matmul(trans_a, trans_b);
    *p_off_out = real3_add(off_a, real3x3_vecmul(trans_a, off_b));
}

/**
 * @brief Merges transformations A^{-1}(x) = T_A^T @ (x - r_A) and B^{-1}(x) = T_B^T @ (x - r_B) into an equivalent
 * transformation C^{-1}(x) such that:
 *
 * C^{-1}(x) = B^{-1}(A^{-1}(x)) = T_B^T @ (T_A^T @ (x - r_A) - r_B) = (T_B^T @ T_A^T) @ x - (T_A^T @ (r_A + T_B^T @
 * r_B)).
 *
 * This is useful when this operation will be repeatedly applied, as it will only require same cost as only one
 * transformation after the overhead of merging.
 *
 * @param trans_a Inverse transformation matrix of first transformation.
 * @param off_a Offset of the first transformation.
 * @param trans_b Inverse transformation matrix of second transformation.
 * @param off_b Offset of the second transformation.
 * @param p_trans_out Pointer which receives the output inverse transformation matrix.
 * @param p_off_out Pointer which receives the output offset.
 */
static inline void merge_transformations_reverse(const real3x3_t trans_a, const real3_t off_a, const real3x3_t trans_b,
                                                 const real3_t off_b, real3x3_t *p_trans_out, real3_t *p_off_out)
{
    *p_trans_out = real3x3_matmul(trans_b, trans_a);
    *p_off_out = real3x3_vecmul(trans_a, real3_add(off_a, real3x3_vecmul(trans_b, off_b)));
}

#endif // TRANSFORMATION_H
