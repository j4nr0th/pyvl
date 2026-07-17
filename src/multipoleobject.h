/**
 * Header for the Multipole Python type.
 *
 * Wraps the C multipole_t struct and its operations
 * (multipole_create, multipole_eval, multipole_add_shift, multipole_update)
 * into a CPython heap type with NumPy ufunc-like eval semantics.
 */
#pragma once
#include "core/multipole.h"
#include "module.h"

typedef struct
{
    PyObject_HEAD;
    multipole_t multipole; /**< Embedded C multipole expansion.              */
    double *coeff_buffer;  /**< Owned buffer: 3 * n_coeffs doubles.          */
} PyVL_MultipoleObject;

CVL_INTERNAL
extern PyType_Spec pyvl_multipole_typespec;
