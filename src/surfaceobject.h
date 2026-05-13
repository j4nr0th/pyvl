/**
 *  Header with the Python's Surface type.
 */
#pragma once
#include "core/mesh.h"
#include "module.h"

typedef struct
{
    PyObject_HEAD unsigned n_lines;
    line_t lines[];
} PyVL_SurfaceObject;

CVL_INTERNAL
extern PyType_Spec pyvl_surface_typespec;

CVL_INTERNAL
PyVL_SurfaceObject *pyvl_surface_from_points(PyTypeObject *surface_type, unsigned n_points,
                                             const unsigned CVL_ARRAY_ARG(points, static restrict n_points));

CVL_INTERNAL
PyVL_SurfaceObject *pyvl_surface_from_lines(PyTypeObject *surface_type, unsigned n,
                                            const line_t CVL_ARRAY_ARG(lines, static restrict n));

CVL_INTERNAL
PyVL_SurfaceObject *pyvl_surface_from_mesh_surface(PyTypeObject *surface_type, const mesh_t *msh, geo_id_t id);
