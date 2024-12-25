//
// Created by jan on 29.11.2024.
//

#ifndef REFERENCEFRAMEOBJECT_H
#define REFERENCEFRAMEOBJECT_H

#include "core/transformation.h"
#include "module.h"

typedef struct PyVL_ReferenceFrame PyVL_ReferenceFrame;

typedef struct PyVL_ReferenceFrame
{
    PyObject_HEAD transformation_t transformation;
    PyVL_ReferenceFrame *parent;
} PyVL_ReferenceFrame;

CVL_INTERNAL
extern PyTypeObject pyvl_reference_frame_type;

#endif // REFERENCEFRAMEOBJECT_H
