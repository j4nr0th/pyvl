#pragma once
/*
 * OpenCL program compilation.
 *
 * Supports creating a program from a single source string.  On build
 * failure, the build log is captured and can be retrieved via
 * @ref cvl_cl_program_build_log.  The log pointer is valid until the
 * program is destroyed or rebuilt.
 */

#include "../common.h"
#include "cvl_cl_common.h"
#include "cvl_cl_ctx.h"

#include <CL/cl.h>

/* ------------------------------------------------------------------ */
/* Program descriptor                                                 */
/* ------------------------------------------------------------------ */

typedef enum
{
    CVL_CL_PROGRAM_SOURCE_NONE,   /**< Terminator (not used). */
    CVL_CL_PROGRAM_SOURCE_STRING, /**< Program from a single source string. */
} cvl_cl_program_source_type_t;

typedef struct
{
    cvl_cl_program_source_type_t source_type;
    const char *source_string;    /**< OpenCL C source code. */
    const char *build_options;    /**< Compiler options (e.g. "-cl-fast-relaxed-math"), or NULL. */
    cvl_cl_precision_t precision; /**< FP32 or FP64 (default).  Controls real_t typedef in kernel. */
} cvl_cl_program_desc_t;

/* ------------------------------------------------------------------ */
/* Program handle                                                     */
/* ------------------------------------------------------------------ */

struct cvl_cl_program_t
{
    cl_program program;
    const cvl_cl_ctx_t *ctx;      /**< Borrowed reference. */
    const allocator_t *allocator; /**< Allocator for build log (NULL = default). */
    char *build_log;              /**< Captured build log (NULL if build succeeded or no build attempted). */
};

/**
 * @brief Create and build a program from a descriptor.
 *
 * The program is built synchronously during this call.
 *
 * @param ctx      Context (must outlive the program).
 * @param desc     Program descriptor (source + options).
 * @param device   Device to build for.
 * @param out      Filled with the new program on success.
 * @param allocator  Allocator for the build log (NULL = default).
 * @return CVL_CL_SUCCESS or error.  CVL_CL_ERR_PROGRAM_BUILD if compilation
 *         failed (log available via @ref cvl_cl_program_build_log).
 */
cvl_cl_status_t cvl_cl_program_create(const cvl_cl_ctx_t *ctx, const cvl_cl_program_desc_t *desc, cl_device_id device,
                                      cvl_cl_program_t *out, const allocator_t *allocator);

/**
 * @brief Return the captured build log, or NULL if no log.
 *
 * The returned pointer is valid until the program is destroyed.
 */
const char *cvl_cl_program_build_log(const cvl_cl_program_t *program);

/**
 * @brief Destroy a program, freeing the build log.
 *
 * @param program Program to destroy (may be NULL).
 */
void cvl_cl_program_destroy(cvl_cl_program_t *program);

/** @brief Return the raw cl_program. */
static inline cl_program cvl_cl_program_program(const cvl_cl_program_t *program)
{
    return program ? program->program : NULL;
}

/** @brief Return the context this program belongs to. */
static inline const cvl_cl_ctx_t *cvl_cl_program_ctx(const cvl_cl_program_t *program)
{
    return program ? program->ctx : NULL;
}
