#pragma once
/*
 * OpenCL program compilation.
 *
 * A program is compiled from a single source string with optional
 * build options and a precision selector.  On build failure the build
 * log is copied into a caller-provided buffer (truncated if too
 * large) - the wrapper performs no heap allocation.
 *
 * The descriptor has no "source type" tag: a NULL source string is
 * simply invalid (enforced with assert).
 */

#include "../common.h"
#include "cvl_cl_common.h"

#include <CL/cl.h>

/* ------------------------------------------------------------------ */
/* Program descriptor                                                 */
/* ------------------------------------------------------------------ */

typedef struct
{
    const char *source_string;    /**< OpenCL C source code (must be non-NULL). */
    const char *build_options;    /**< Compiler options (e.g. "-cl-fast-relaxed-math"), or NULL. */
    cvl_cl_precision_t precision; /**< FP32 or FP64 (default).  Controls real_t typedef in kernel. */
} cvl_cl_program_desc_t;

/* ------------------------------------------------------------------ */
/* Program handle                                                     */
/* ------------------------------------------------------------------ */

/**
 * @brief Create and build a program from a descriptor.
 *
 * The program is built synchronously during this call.
 *
 * On build failure (CVL_CL_ERR_PROGRAM_BUILD) the device build log is
 * copied into @p log (NUL-terminated, truncated to @p log_capacity - 1).
 * Pass log_capacity == 0 to skip log capture.
 *
 * @param ctx          Context the program belongs to.
 * @param device       Device to build for.
 * @param desc         Program descriptor (source + options).
 * @param log          Caller-provided buffer for the build log (may be NULL).
 * @param log_capacity Capacity of @p log in bytes.
 * @param out          Filled with the new cl_program on success (NULL on failure).
 * @return CVL_CL_SUCCESS, CVL_CL_ERR_PROGRAM_BUILD if compilation failed, or other error.
 */
cvl_cl_status_t cvl_cl_program_create(cl_context ctx, cl_device_id device, const cvl_cl_program_desc_t *desc, char *log,
                                      size_t log_capacity, cl_program *out);

/**
 * @brief Destroy a program.
 *
 * @param program Program to destroy (may be NULL).
 */
void cvl_cl_program_destroy(cl_program *program);
