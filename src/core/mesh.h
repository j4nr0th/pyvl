//
// Created by jan on 16.11.2024.
//

#ifndef MESH_H
#define MESH_H

#include "common.h"

typedef struct
{
    geo_id_t p1;
    geo_id_t p2;
} line_t;

typedef struct
{
    uint32_t n_lines;
    geo_id_t lines[];
} surface_t;

_Static_assert(sizeof(geo_id_t) == sizeof(uint32_t));
_Static_assert(sizeof(surface_t) == sizeof(uint32_t));

/**
 *  Struct containing either primary or dual mesh.
 *
 *  A primary mesh like this:
 *
 *  Points:
 *  0-----1-----2-----3-----4
 *  |     |     |     |     |
 *  |     |     |     |     |
 *  |     |     |     |     |
 *  5-----6-----7-----8-----9
 *  |     |     |     |     |
 *  |     |     |     |     |
 *  |     |     |     |     |
 *  10----11----12----13----14
 *  |     |     |     |     |
 *  |     |     |     |     |
 *  |     |     |     |     |
 *  15----16----17----18----19
 *
 *  Lines:
 *  +--1--+--2--+--3--+--4--+
 *  |     |     |     |     |
 *  17    18    19    20    21  example:
 *  |     |     |     |     |       - Line 23 would be {6, 11}, since it connects point 6 and 11.
 *  +--5--+--6--+--7--+--8--+
 *  |     |     |     |     |
 *  22    23    24    25    26
 *  |     |     |     |     |
 *  +--9--+-10--+-11--+-12--+
 *  |     |     |     |     |
 *  27    28    29    30    31
 *  |     |     |     |     |
 *  +-13--+-14--+-15--+-16--+
 *
 *
 *  Surfaces:
 *  +-----+-----+-----+-----+
 *  |     |     |     |     |
 *  |  0  |   1 |   2 |   3 |
 *  |     |     |     |     |
 *  +-----+-----+-----+-----+
 *  |     |     |     |     |   example:
 *  |  4  |   5 |   6 |   7 |       - Surface 6 would be {24, 11, REVERSED | 25, REVERSED | 7}, assuming
 *  |     |     |     |     |         line 24 is {7, 12}, line 11 is {12, 13}, line 25 is {8, 13}, and
 *  +-----+-----+-----+-----+         line 7 is {7, 8}. The REVERSED is OR-ed with the IDs of the lines, which
 *  |     |     |     |     |         are oriented opposite to the lines, meaning their end point is where the
 *  |  8  |   9 |  10 |  11 |         previous line ends instead of their start point.
 *  |     |     |     |     |
 *  +-----+-----+-----+-----+
 *
 *  Would have:
 *
 *  line = {0, 1} -> points 0 and 1 form a line
 *  surfaces = {0}, {1, 6, 7, 2}, ...
 *
 *  The dual mesh would instead contain connectivity:
 *
 *  Points corresponds to primal surfaces:
 *
 *     0------1-----2-----3
 *     |      |     |     |
 *     |      |     |     |
 *     |      |     |     |
 *     4------5-----6-----7
 *     |      |     |     |
 *     |      |     |     |
 *     |      |     |     |
 *     8------9----10----11
 *
 *  Lines indicate adjacency between primal surfaces:
 *     +---0--+--1--+--2--+
 *     |      |     |     |
 *     9      10    11    12
 *     |      |     |     |     example:
 *     +---3--+--4--+--5--+         - Line 4 is {5, 6}, since surfaces 5 and 6 are neighbours.
 *     |      |     |     |
 *     13     14    15    16
 *     |      |     |     |
 *     +---6--+--7--+--8--+
 *
 *
 *  Surfaces:
 *     +------+-----+-----+
 *     |      |     |     |
 *     |  0   |  1  |  2  |
 *     |      |     |     |     example:
 *     +------+-----+-----+         - Surface 1 is {10, 4, REVERSED | 11, REVERSED | 1}.
 *     |      |     |     |
 *     |  3   |  4  |  5  |
 *     |      |     |     |
 *     +------+-----+-----+
 *
 *
 * Intended use for dual mesh:
 *      Dual mesh reveals neighbouring elements, since surfaces on primal mesh become
 *      points on dual mesh, a line connecting two of them indicates surfaces are neighbours
 *      The following operations become very simple:
 *
 *      - Find all surfaces containing point i:
 *          On the dual mesh it becomes surface i. All lines which make up this
 *          surface contain indices, which on dual mesh correspond to points,
 *          but on primal they are surfaces, which all contain point i.
 *      - Find which two surfaces are separated by line j:
 *          Line j on primal mesh becomes line j on the dual mesh. The two dual points
 *          which make it up are then surfaces on primal mesh.
 *      - Find all lines containing point k:
 *          On dual mesh, primal point k is dual surface k. From that all primal surfaces
 *          containing this point can be found by just taking all unique dual points which
 *          make up the dual lines bounding this surface. From there lines containing primal
 *          surfaces can be searched to find all lines containing point k (2 per surface,
 *          one unique).
 *
 */
typedef struct
{
    unsigned n_points;
    real3_t *positions; //  for sake of vectorization, this could be split into 3 arrays
    unsigned n_lines;
    line_t *lines;
    unsigned n_surfaces;
    const surface_t **surfaces; // Pointers to surfaces, directly by surfaces themselves.
} mesh_t;

/**
 * @brief Compute displacement from beginning of the line to the end. By setting the `line_id.orientation != 0`, the
 * direction can be reversed.
 *
 * @param mesh Mesh which contains the geometry.
 * @param line_id ID of the line.
 * @return Vector of displacement from beginning of line to the end.
 */
// [[unsequenced]]
real3_t line_direction(const mesh_t *mesh, geo_id_t line_id);

/**
 * @brief Compute center of the surface element.
 *
 * @param mesh Mesh which contains the geometry.
 * @param surface_id ID of the surface.
 * @return Position vector of the surface center.
 */
// [[unsequenced]]
real3_t surface_center(const mesh_t *mesh, geo_id_t surface_id);

/**
 * @brief Compute unit normal vector of the surface. By setting `surface_id.orientation != 0`, the direction of the
 * normal will be flipped.
 *
 * @param mesh Mesh which contains the geometry.
 * @param surface_id ID of the surface.
 * @return Normal vector of the surface with unit length.
 */
// [[unsequenced]]
real3_t surface_normal(const mesh_t *mesh, geo_id_t surface_id);

/**
 * @brief Create dual mesh, which describes connectivity of the primal mesh. On a dual mesh, each position represents a
 * center of a surface of a primal mesh, each line represents adjacency between two primal surfaces, and each surface
 * represents lines connected to points on primal mesh.
 *
 * If surfaces S1 and S2 are separated by line L1 on the primal mesh, the dual mesh will have line L1 connecting dual
 * points S1 and S2 (which correspond to primal surfaces S1 and S2).
 *
 * @param primal Primal mesh, to which the dual mesh is associated.
 * @param allocator Allocator callbacks used to allocate/deallocate memory for the mesh.
 * @return Dual mesh, which contains information about connectivity of primal mesh.
 */
mesh_t *mesh_dual_from_primal(const mesh_t *primal, const allocator_t *allocator);

/**
 * @brief This function allows a mesh to be created by simply specifying element connectivity, which is very common with
 * meshing tools, such as GMSH.
 * 
 * @param n_elements Number of elements given.
 * @param point_counts Array which contains number of points for each element.
 * @param flat_points Points of elements, one after another.
 * @param allocator Allocator used to allocate memory for the mesh.
 * @return Mesh with elements specified, but no positional information.
 */
mesh_t *mesh_from_elements(unsigned n_elements, const unsigned point_counts[static restrict n_elements],
                           const unsigned flat_points[restrict], const allocator_t *allocator);

/**
 * @brief Intended to be used in order to convert the mesh into a format more common with other meshers, by just
 * specifying point counts and connectivity.
 *
 * @param mesh Mesh to convert.
 * @param p_point_counts Pointer which receives a pointer to an array of point counts per element.
 * @param p_flat_points Pointer which receives a pointer to an array of flattened element indices.
 * @param allocator Allocator used to allocate memory for the two arrays.
 * @return Number of elements converted, or zero on failure.
 */
unsigned mesh_to_elements(const mesh_t *mesh, unsigned **p_point_counts, unsigned **p_flat_points,
                          const allocator_t *allocator);

void mesh_free(mesh_t *this, const allocator_t *allocator);

#endif //MESH_H
