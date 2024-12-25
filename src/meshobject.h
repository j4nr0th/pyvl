//
// Created by jan on 24.11.2024.
//

#ifndef MESHOBJECT_H
#define MESHOBJECT_H

#include "core/mesh.h"
#include "module.h"

typedef struct
{
    PyObject_HEAD mesh_t mesh;
} PyVL_MeshObject;

CVL_INTERNAL
extern PyTypeObject pyvl_mesh_type;

#endif // MESHOBJECT_H
