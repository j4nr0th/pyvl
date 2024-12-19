//
// Created by jan on 16.11.2024.
//

#ifndef MESH_IO_H
#define MESH_IO_H

#include "common.h"
#include "mesh.h"

/*

Mesh file format specification for version 0:

    [version number]
    [number of points] [number of lines] [number of elements]
    [x of point 1] [y of point 1] [z of point 1]
        ...
    [x of point n] [y of point n] [z of point n]
    [point 1 for line 1] [point 2 for line 2]
        ...
    [point 1 for line m] [point 2 for line m]
    [number of lines in surface 1] [line 1 for surface 1] ... [line l for surface 1]
        ....
    [number of lines in surface k] [line 1 for surface k] ... [line p for surface k]

    All whitespace is ignored, with no difference between new line and space.
*/

/**
 * @brief Convert mesh into a string, which can then be converted back into the original mesh.
 *
 * @param this Mesh to serialize into a UTF-8 string.
 * @param positions Positions of mesh points.
 * @param allocator Allocator to use for allocating/reallocating the string. Can be a stack allocator.
 * @return Null-terminated string which can be converted into a mesh by calling `deserialize_mesh`.
 */
char *serialize_mesh(const mesh_t *this, const real3_t *positions, const allocator_t *allocator);

/**
 * @brief Convert a serialized mesh into a mesh object. If conversion fails, value returned through `current_line`
 * parameter can be used to find where the error with the mesh was.
 *
 * @param p_out Pointer which receives the deserialized mesh.
 * @param p_positions Pointer which receives positions of mesh points.
 * @param str String which will be converted into a mesh.
 * @param allocator Allocator which should be used to allocate memory for the mesh object.
 * @return On failure -1, on success 0.
 */
int deserialize_mesh(mesh_t *p_out, real3_t **p_positions, const char *str, const allocator_t *allocator);

#endif // MESH_IO_H
