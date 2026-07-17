/**
 * Common module header with global definitions and includes.
 */
#pragma once
#define PY_SSIZE_T_CLEAN

#ifndef NPY_NO_DEPRECATED_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#endif

#ifndef PY_ARRAY_UNIQUE_SYMBOL
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL cvl
#endif

#include "core/common.h"
#include <Python.h>
#include <stdio.h>

#define PYVL_CMODULE_NAME "pyvl.cvl"
#define PYVL_CTYPE_NAME(type) (PYVL_CMODULE_NAME "." #type)

typedef struct
{
    PyTypeObject *geoid_type;
    PyTypeObject *mesh_type;
    PyTypeObject *multipole_type;
    PyTypeObject *rf_type;
    PyTypeObject *transformation_plane_type;
} module_state_t;

CVL_INTERNAL
extern PyModuleDef cvl_module;

/**
 * Try and get the module state from a Python type which inherits from one of the module's types.
 *
 * For PyCMethod we have "defining_class" as an argument, which we can use for the PyType_GetModuleState method.
 * For all other cases we can use get_module_state(Py_TYPE(obj)) as a fallback instead.
 *
 * @param type Type, which is either defined by the module or inherits from a type that is.
 * @return Pointer to the module state or NULL if not successful.
 */
static inline const module_state_t *get_module_state(PyTypeObject *type)
{
    // Go through the type to get the module
    PyObject *const mod = PyType_GetModuleByDef(type, &cvl_module);
    return mod ? PyModule_GetState(mod) : NULL;
}
