//
// Created by jan on 24.11.2024.
//

#ifndef SURFACEOBJECT_H
#define SURFACEOBJECT_H

#include "module.h"
#include "core/mesh.h"

typedef struct
{
    PyObject_HEAD
    unsigned n_lines;
    line_t lines[];
} PyDust_SurfaceObject;

CDUST_INTERNAL
extern PyTypeObject pydust_surface_type;

CDUST_INTERNAL
PyDust_SurfaceObject *pydust_surface_from_points(unsigned n_points, const unsigned points[static restrict n_points]);

CDUST_INTERNAL
PyDust_SurfaceObject *pydust_surface_from_lines(unsigned n, const line_t lines[static restrict n]);

CDUST_INTERNAL
PyDust_SurfaceObject *pydust_surface_from_mesh_surface(const mesh_t *msh, geo_id_t id);

#endif //SURFACEOBJECT_H
