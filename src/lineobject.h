//
// Created by jan on 23.11.2024.
//

#ifndef LINEOBJECT_H
#define LINEOBJECT_H

#include "module.h"

typedef struct
{
    PyObject_HEAD unsigned begin;
    unsigned end;
} PyVL_LineObject;

CVL_INTERNAL
extern PyTypeObject pyvl_line_type;

CVL_INTERNAL
PyVL_LineObject *pyvl_line_from_indices(unsigned begin, unsigned end);

#endif // LINEOBJECT_H
