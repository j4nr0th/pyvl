//
// Created by jan on 23.11.2024.
//

#ifndef GEOIDOBJECT_H
#define GEOIDOBJECT_H

#include "core/common.h"
#include "module.h"

typedef struct
{
    PyObject_HEAD geo_id_t id;
} PyVL_GeoIDObject;

CVL_INTERNAL
extern PyTypeObject pyvl_geoid_type;

CVL_INTERNAL
PyVL_GeoIDObject *pyvl_geoid_from_value(geo_id_t id);

#endif // GEOIDOBJECT_H
