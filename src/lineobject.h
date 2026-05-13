/**
 *  Header for the Line Python type.
 */
#pragma once
#include "module.h"

typedef struct
{
    PyObject_HEAD;
    unsigned begin;
    unsigned end;
} PyVL_LineObject;

CVL_INTERNAL
extern PyType_Spec pyvl_line_typespec;

CVL_INTERNAL
PyVL_LineObject *pyvl_line_from_indices(PyTypeObject *line_type, unsigned begin, unsigned end);
