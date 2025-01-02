//
// Created by jan on 24.11.2024.
//

#ifndef SURFACEOBJECT_H
#define SURFACEOBJECT_H

#include "core/mesh.h"
#include "module.h"

typedef struct
{
    PyObject_HEAD unsigned n_lines;
    line_t lines[];
} PyVL_SurfaceObject;

CVL_INTERNAL
extern PyTypeObject pyvl_surface_type;

CVL_INTERNAL
PyVL_SurfaceObject *pyvl_surface_from_points(unsigned n_points,
                                             const unsigned CVL_ARRAY_ARG(points, static restrict n_points));

CVL_INTERNAL
PyVL_SurfaceObject *pyvl_surface_from_lines(unsigned n, const line_t CVL_ARRAY_ARG(lines, static restrict n));

CVL_INTERNAL
PyVL_SurfaceObject *pyvl_surface_from_mesh_surface(const mesh_t *msh, geo_id_t id);

#endif // SURFACEOBJECT_H
