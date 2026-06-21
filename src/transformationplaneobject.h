/**
 * Header exposing Python's TransformationPlane object.
 */
#pragma once

#include "module.h"
#include "time_dependent_vector.h"

typedef struct PyVL_TransformationPlane PyVL_TransformationPlane;

typedef struct PyVL_TransformationPlane
{
    PyObject_HEAD;
    pyvl_vec_time_dependent_t origin;
    pyvl_vec_time_dependent_t normal;
} PyVL_TransformationPlane;

CVL_INTERNAL
extern PyType_Spec pyvl_transformation_plane_typespec;

CVL_INTERNAL
bool pyvl_transformation_plane_is_time_invariant(const PyVL_TransformationPlane *plane);

CVL_INTERNAL
bool pyvl_transformation_plane_ensure_time_invariant(const PyVL_TransformationPlane *plane);
