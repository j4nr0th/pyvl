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
} PyDust_GeoIDObject;

CDUST_INTERNAL
extern PyTypeObject pydust_geoid_type;

CDUST_INTERNAL
PyDust_GeoIDObject *pydust_geoid_from_value(geo_id_t id);

#endif // GEOIDOBJECT_H
