/**
 *  Header for the Mesh Python type.
 */
#pragma once
#include "core/mesh.h"
#include "module.h"

typedef struct
{
    PyObject_HEAD;
    mesh_t mesh;
} PyVL_MeshObject;

CVL_INTERNAL
extern PyType_Spec pyvl_mesh_typespec;
