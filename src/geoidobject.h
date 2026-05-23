/**
 * Header with definition of the GeoID Python type.
 */
#pragma once

#include "module.h"

typedef struct
{
    PyObject_HEAD;
    geo_id_t id;

} PyVL_GeoIDObject;

/**
 * TypeSpec for the GeoID type.
 */
CVL_INTERNAL
extern PyType_Spec pyvl_geoid_typespec;

CVL_INTERNAL
bool pyvl_geoid_from_pyvalue(const module_state_t *state, PyObject *o, geo_id_t *p_val);

CVL_INTERNAL
PyVL_GeoIDObject *pyvl_geoid_new(const module_state_t *state, geo_id_t id);
