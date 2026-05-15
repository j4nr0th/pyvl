/**
 *  Header exposing Python's ReferenceFrame object.
 */
#pragma once
#include "core/transformation.h"
#include "module.h"

typedef struct PyVL_ReferenceFrame PyVL_ReferenceFrame;

typedef enum
{
    PYVL_RF_CONSTANT,
    PYVL_RF_CALLABLE,
} pyvl_rf_value_type_t;

typedef struct
{
    pyvl_rf_value_type_t type;
    union {
        real3_t constant;
        PyObject *callable;
    } value;
} pyvl_rf_time_dependent_t;

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
