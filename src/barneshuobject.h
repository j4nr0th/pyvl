/**
 * Header for the Barnes-Hut Tree Python type.
 *
 * Wraps the C barnes_hut_tree_t struct and its operations
 * (build, eval, depth stats) into a CPython heap type.
 */
#pragma once
#include "core/barnes_hut_tree.h"
#include "module.h"

typedef struct
{
    PyObject_HEAD;
    barnes_hut_tree_t tree;   /**< Embedded C tree handle (buffer owned separately). */
    unsigned n_threads;       /**< Default thread count used for build/eval.            */
    PyObject *sources_coords; /**< INCREF'd reference to source coordinates (PyArrayObject). */
    PyObject *sources_values; /**< INCREF'd reference to source values (PyArrayObject).      */
    bool built;               /**< True once build() has populated the tree.            */
} PyVL_BarnesHutObject;

CVL_INTERNAL
extern PyType_Spec pyvl_bh_tree_typespec;
