/**
 *  Header exposing Python's ReferenceFrame object.
 */
#pragma once
#include "core/transformation.h"
#include "module.h"

typedef struct PyVL_ReferenceFrame PyVL_ReferenceFrame;

typedef struct PyVL_ReferenceFrame
{
    PyObject_HEAD;
    transformation_t transformation;
    PyVL_ReferenceFrame *parent;
} PyVL_ReferenceFrame;

CVL_INTERNAL
extern PyType_Spec pyvl_reference_frame_typespec;
