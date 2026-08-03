#pragma once
/*
 * OpenCL kernel object and typed argument binding.
 *
 * The highlight of this module is @ref cvl_cl_kernel_set_args, which
 * accepts a NULL-terminated array of @ref cvl_cl_karg_t descriptors
 * built with designated initializers - the same pattern used by
 * cpyutl's parse_arguments.
 *
 * Example:
 * @code
 *   cvl_cl_kernel_set_args(kernel,
 *       (cvl_cl_karg_t[]){
 *           {.type = CVL_CL_KARG_BUFFER,       .index = 0, .mem = pos_buf},
 *           {.type = CVL_CL_KARG_BUFFER,       .index = 1, .mem = val_buf},
 *           {.type = CVL_CL_KARG_SCALAR_UINT,  .index = 2, .scalar_uint = n},
 *           {.type = CVL_CL_KARG_SCALAR_DOUBLE,.index = 3, .scalar_double = theta},
 *           {},
 *       });
 * @endcode
 */

#include "cvl_cl_common.h"

#include <CL/cl.h>

/* ------------------------------------------------------------------ */
/* Kernel argument descriptor                                         */
/* ------------------------------------------------------------------ */

typedef enum
{
    CVL_CL_KARG_NONE,          /**< Array terminator. */
    CVL_CL_KARG_BUFFER,        /**< A cl_mem buffer. */
    CVL_CL_KARG_SCALAR_INT,    /**< int scalar. */
    CVL_CL_KARG_SCALAR_UINT,   /**< unsigned int scalar. */
    CVL_CL_KARG_SCALAR_LONG,   /**< long long scalar. */
    CVL_CL_KARG_SCALAR_ULONG,  /**< unsigned long long scalar. */
    CVL_CL_KARG_SCALAR_FLOAT,  /**< float scalar. */
    CVL_CL_KARG_SCALAR_DOUBLE, /**< double scalar. */
    CVL_CL_KARG_LOCAL,         /**< __local memory buffer (size in bytes). */
} cvl_cl_karg_type_t;

typedef struct cvl_cl_karg_t
{
    cvl_cl_karg_type_t type;
    unsigned index; /**< Kernel argument index (0-based). */
    union {
        cl_mem mem; /**< For CVL_CL_KARG_BUFFER - a cl_mem handle. */
        int scalar_int;
        unsigned scalar_uint;
        long long scalar_long;
        unsigned long long scalar_ulong;
        float scalar_float;
        double scalar_double;
        size_t local_size; /**< Bytes for CVL_CL_KARG_LOCAL. */
    };
} cvl_cl_karg_t;

/* ------------------------------------------------------------------ */
/* Kernel lifecycle                                                   */
/* ------------------------------------------------------------------ */

/**
 * @brief Create a kernel object from a program.
 *
 * @param program Program the kernel belongs to (must outlive the kernel).
 * @param name    Kernel function name (null-terminated).
 * @param out     Filled with the new cl_kernel on success (NULL on failure).
 * @return CVL_CL_SUCCESS or error.
 */
cvl_cl_status_t cvl_cl_kernel_create(cl_program program, const char *name, cl_kernel *out);

/**
 * @brief Set kernel arguments from a typed descriptor array.
 *
 * Walks the NULL-terminated @p kargs array and calls clSetKernelArg
 * for each entry.  Returns the first error encountered (arguments
 * before the error are already set).
 *
 * @param kernel Kernel to set arguments on.
 * @param kargs  NULL-terminated array of argument descriptors.
 * @return CVL_CL_SUCCESS or CVL_CL_ERR_KERNEL_ARG on failure.
 */
cvl_cl_status_t cvl_cl_kernel_set_args(cl_kernel kernel, const cvl_cl_karg_t kargs[]);

/**
 * @brief Destroy a kernel.
 *
 * @param kernel Kernel to destroy (may be NULL).
 */
void cvl_cl_kernel_destroy(cl_kernel *kernel);
