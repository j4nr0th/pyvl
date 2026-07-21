#pragma once
#include "common.h"

/**
 * @brief Reflection plane used for symmetry transformations.
 *
 * Defines a plane by an origin point and a unit normal vector.
 * Positions and vectors can be reflected across this plane to
 * implement symmetry boundary conditions.
 *
 * The normal must be a unit vector (@f$ \|\mathbf{n}\| = 1 @f$).
 */
typedef struct
{
    real3_t origin; /**< A point lying on the plane. */
    real3_t normal; /**< Unit vector orthogonal to the plane. */
} transformation_plane_t;

/**
 * Transforms a position according to the plane, reflecting it across it.
 *
 * @param plane Plane used for transformation.
 * @param point Position to transform.
 * @return Transformed position.
 */
static inline real3_t transformation_plane_transform_position(const transformation_plane_t *plane, const real3_t point)
{
    // Normal distance from the plane
    const real_t dist = real3_dot(plane->normal, real3_sub(point, plane->origin));
    // Reflect across the plane
    return real3_sub(point, real3_mul1(plane->normal, 2 * dist));
}

/**
 * Transforms a direction vector according to the plane, reflecting it across it.
 * The plane's origin is not relevant for this transformation.
 *
 * @param plane Plane to use for the transformation.
 * @param vector Vector to transform.
 * @return Transformed vector.
 */
static inline real3_t transformation_plane_transform_vector(const transformation_plane_t *plane, const real3_t vector)
{
    // Reflect across the plane
    return real3_sub(vector, real3_mul1(plane->normal, 2 * real3_dot(plane->normal, vector)));
}
