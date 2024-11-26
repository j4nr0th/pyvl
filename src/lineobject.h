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
} PyDust_LineObject;

CDUST_INTERNAL
extern PyTypeObject pydust_line_type;

CDUST_INTERNAL
PyDust_LineObject *pydust_line_from_indices(unsigned begin, unsigned end);

#endif // LINEOBJECT_H
