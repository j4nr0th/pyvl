/**
 *  Header exposing Python's ReferenceFrame object.
 */
#pragma once
#include "core/transformation.h"
#include "module.h"
#include "time_dependent_vector.h"

typedef struct PyVL_ReferenceFrame PyVL_ReferenceFrame;

typedef pyvl_vec_time_dependent_t pyvl_rf_time_dependent_t;

typedef struct PyVL_ReferenceFrame
{
    PyObject_HEAD;
    pyvl_rf_time_dependent_t position;
    pyvl_rf_time_dependent_t velocity;
    pyvl_rf_time_dependent_t orientation;
    pyvl_rf_time_dependent_t rotation;
    PyVL_ReferenceFrame *parent;
} PyVL_ReferenceFrame;

CVL_INTERNAL
extern PyType_Spec pyvl_reference_frame_typespec;
