/**
 * Header for the OpenCL backend Python type.
 *
 * Wraps the cvl_cl_compute_t kernel registry (ctx/queue/device + compiled
 * programs) and the cvl_cl_tree_t persistent tree handle into a Python
 * backend that owns all device resources and creates trees.
 */
#pragma once
#include "core/cl_compute/cvl_cl_compute.h"
#include "core/cl_compute/cvl_cl_tree.h"
#include "module.h"

/* ------------------------------------------------------------------ */
/*  CLBackend                                                          */
/* ------------------------------------------------------------------ */

typedef struct
{
    PyObject_HEAD;
    /* Owned device resources. */
    cvl_cl_device_t device;       /**< Cached device info. */
    cl_context ctx;               /**< Owned context. */
    cl_command_queue queue;       /**< Owned queue. */
    cvl_cl_compute_t compute;     /**< Owned kernel registry. */
    cvl_cl_precision_t precision; /**< FP32/FP64. */
    unsigned n_threads;           /**< Thread count (unused for OpenCL, kept for API parity). */
    bool closed;                  /**< True once the backend has been closed. */
} PyVL_CLBackendObject;

/* ------------------------------------------------------------------ */
/*  CLTree                                                             */
/* ------------------------------------------------------------------ */

typedef struct
{
    PyObject_HEAD;
    cvl_cl_tree_t tree;            /**< Embedded persistent tree (device buffers owned). */
    PyVL_CLBackendObject *backend; /**< Owning backend (keeps device resources alive). */
    bool built;                    /**< True once the first build has completed. */
    /* Source arrays (INCREF'd) — kept so eval-at-sources can reuse them. */
    PyObject *sources_coords; /**< INCREF'd (PyArrayObject) or NULL. */
    PyObject *sources_values; /**< INCREF'd (PyArrayObject) or NULL. */
    unsigned n_threads;       /**< Default thread count (API parity). */
} PyVL_CLTreeObject;

/* ------------------------------------------------------------------ */
/*  Futures                                                            */
/* ------------------------------------------------------------------ */

typedef struct
{
    PyObject_HEAD;
    PyVL_CLTreeObject *tree;     /**< Owning tree (keeps it alive). */
    cvl_cl_tree_build_job_t job; /**< Embedded C job. */
    PyObject *result;            /**< Result (tree) or NULL until .result(). */
    PyObject *exception;         /**< Exception instance if the job failed, else NULL. */
    bool done;                   /**< True once finished. */
} PyVL_CLTreeBuildObject;

typedef struct
{
    PyObject_HEAD;
    PyVL_CLTreeObject *tree;    /**< Owning tree (keeps it alive). */
    cvl_cl_tree_eval_job_t job; /**< Embedded C job. */
    PyObject *result;           /**< Result (ndarray) or NULL until .result(). */
    PyObject *exception;        /**< Exception instance if the job failed, else NULL. */
    bool done;                  /**< True once finished. */
    /* Host scratch for the readback (allocated lazily). */
    void *host_out;     /**< real3_t scratch for FP64 readback. */
    float *scratch_f32; /**< FP32 conversion scratch. */
    /* Output shape. */
    size_t n_targets;     /**< Flattened target count. */
    int ndim;             /**< Number of dims (includes the trailing 3). */
    Py_ssize_t *dims_arr; /**< Copy of the (..., 3) dims. */
} PyVL_CLTreeEvalObject;

CVL_INTERNAL
extern PyType_Spec pyvl_cl_backend_typespec;
CVL_INTERNAL
extern PyType_Spec pyvl_cl_tree_typespec;
CVL_INTERNAL
extern PyType_Spec pyvl_cl_tree_build_typespec;
CVL_INTERNAL
extern PyType_Spec pyvl_cl_tree_eval_typespec;

/* Module-level create_backend() (declared for methods.c). */
CVL_INTERNAL
PyObject *pyvl_create_backend(PyObject *self, PyObject *args, PyObject *kwargs);
