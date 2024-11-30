//
// Created by jan on 29.11.2024.
//

#ifndef REFERENCEFRAMEOBJECT_H
#define REFERENCEFRAMEOBJECT_H

#include "core/transformation.h"
#include "module.h"

typedef struct PyDust_ReferenceFrame PyDust_ReferenceFrame;

typedef struct PyDust_ReferenceFrame
{
    PyObject_HEAD transformation_t transformation;
    PyDust_ReferenceFrame *parent;
} PyDust_ReferenceFrame;

CDUST_INTERNAL
extern PyTypeObject pydust_reference_frame_type;

#endif // REFERENCEFRAMEOBJECT_H
