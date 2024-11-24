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


#endif //SURFACEOBJECT_H
