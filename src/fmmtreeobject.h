/**
 * Header for the FMM Tree Python type.
 *
 * Wraps the C fmm_tree_t struct and its operations
 * (build, eval, depth stats) into a CPython heap type.
 */
#pragma once
#include "core/fmm_tree.h"
#include "module.h"

typedef struct
{
    PyObject_HEAD;
    fmm_tree_t tree;          /**< Embedded C tree handle (buffer owned separately). */
    unsigned n_threads;       /**< Default thread count used for build/eval.            */
    PyObject *sources_coords; /**< INCREF'd reference to source coordinates (PyArrayObject). */
    PyObject *sources_values; /**< INCREF'd reference to source values (PyArrayObject).      */
    bool built;               /**< True once build() has populated the tree.            */
} PyVL_FMMTreeObject;

CVL_INTERNAL
extern PyType_Spec pyvl_fmm_tree_typespec;
