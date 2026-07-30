#pragma once
/*
 * OpenCL kernel object and typed argument binding.
 *
 * The highlight of this module is @ref cvl_cl_kernel_set_args, which
 * accepts a NULL-terminated array of @ref cvl_cl_karg_t descriptors
 * built with designated initializers — the same pattern used by
 * cpyutl's parse_arguments and cpyutl_output_create.
 *
 * Example:
 * @code
 *   cvl_cl_kernel_set_args(kernel,
 *       (cvl_cl_karg_t[]){
 *           {.type = CVL_CL_KARG_BUFFER,  .index = 0, .mem = cvl_cl_buffer_mem(&pos_buf)},
 *           {.type = CVL_CL_KARG_BUFFER,  .index = 1, .mem = cvl_cl_buffer_mem(&val_buf)},
 *           {.type = CVL_CL_KARG_SCALAR_UINT, .index = 2, .scalar_uint = n},
 *           {.type = CVL_CL_KARG_SCALAR_DOUBLE, .index = 3, .scalar_double = theta},
 *           {},
 *       });
 * @endcode
 */

#include "cvl_cl_common.h"
#include "cvl_cl_program.h"

#include <CL/cl.h>

/* ------------------------------------------------------------------ */
/* Kernel argument descriptor                                         */
/* ------------------------------------------------------------------ */

typedef enum
{
    CVL_CL_KARG_NONE,          /**< Array terminator. */
    CVL_CL_KARG_BUFFER,        /**< A cl_mem buffer (via @ref cvl_cl_buffer_t*). */
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
        cl_mem mem; /**< For CVL_CL_KARG_BUFFER — a cl_mem handle. */
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
/* Kernel handle                                                      */
/* ------------------------------------------------------------------ */

struct cvl_cl_kernel_t
{
    cl_kernel kernel;
    const cvl_cl_program_t *program; /**< Borrowed reference. */
    size_t preferred_wg_multiple;    /**< Cached from CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE. */
};

/**
 * @brief Create a kernel object from a program.
 *
 * The kernel name is looked up in the compiled program.  The
 * preferred work-group size multiple is queried and cached.
 *
 * @param program Program (must outlive the kernel).
 * @param name    Kernel function name (null-terminated).
 * @param out     Filled with the new kernel on success.
 * @return CVL_CL_SUCCESS or error.
 */
cvl_cl_status_t cvl_cl_kernel_create(const cvl_cl_program_t *program, const char *name, cvl_cl_kernel_t *out);

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
cvl_cl_status_t cvl_cl_kernel_set_args(cvl_cl_kernel_t *kernel, const cvl_cl_karg_t kargs[]);

/**
 * @brief Destroy a kernel.
 *
 * @param kernel Kernel to destroy (may be NULL).
 */
void cvl_cl_kernel_destroy(cvl_cl_kernel_t *kernel);

/* ------------------------------------------------------------------ */
/* Accessors                                                          */
/* ------------------------------------------------------------------ */

/** @brief Return the raw cl_kernel. */
static inline cl_kernel cvl_cl_kernel_kernel(const cvl_cl_kernel_t *kernel)
{
    return kernel ? kernel->kernel : NULL;
}

/** @brief Return the program this kernel belongs to. */
static inline const cvl_cl_program_t *cvl_cl_kernel_program(const cvl_cl_kernel_t *kernel)
{
    return kernel ? kernel->program : NULL;
}

/** @brief Return the cached preferred work-group size multiple (or 0 if not yet queried). */
static inline size_t cvl_cl_kernel_preferred_wg_multiple(const cvl_cl_kernel_t *kernel)
{
    return kernel ? kernel->preferred_wg_multiple : 0;
}
