#pragma once
/*
 * Shared types, status codes, and utility macros for the pyvl OpenCL wrapper.
 *
 * This is the single foundational header for all cvl_cl_* modules.
 * It provides:
 *   - cvl_cl_status_t  — typed error codes (cpyutl-style, never exit())
 *   - cvl_cl_status_str() — human-readable error description
 *   - CVL_CL_CHECK  — macro for cl_int → cvl_cl_status_t + goto
 *   - cvl_cl_status_from_cl_int() — map OpenCL error codes
 *   - CVL_CL_ASSERT  — guarded assertion (same pattern as cpyutl)
 *   - Forward declarations of all opaque types
 *   - Cross-compilation macros (__global / restrict) for shared
 *     C/OpenCL-C headers (used in Phase 2)
 */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* ------------------------------------------------------------------ */
/* Cross-compilation macros                                            */
/*                                                                     */
/* These let .cl.h files compile identically in C17 and OpenCL C.      */
/* Phase 1 wrappers always use the host side (empty defines).          */
/* ------------------------------------------------------------------ */

#ifndef CVL_CL_GLOBAL
#define CVL_CL_GLOBAL
#endif

#ifndef CVL_CL_LOCAL
#define CVL_CL_LOCAL
#endif

#ifndef CVL_CL_CONSTANT
#define CVL_CL_CONSTANT
#endif

#ifndef CVL_CL_RESTRICT
#ifdef __GNUC__
#define CVL_CL_RESTRICT __restrict__
#else
#define CVL_CL_RESTRICT restrict
#endif
#endif

/* ------------------------------------------------------------------ */
/* Status codes                                                        */
/* ------------------------------------------------------------------ */

typedef enum
{
    CVL_CL_SUCCESS = 0,

    /* Platform / device errors. */
    CVL_CL_ERR_PLATFORM,         /**< clGetPlatformIDs failed. */
    CVL_CL_ERR_DEVICE_NOT_FOUND, /**< No matching device found (clGetDeviceIDs returned 0). */
    CVL_CL_ERR_DEVICE,           /**< Generic device error. */
    CVL_CL_ERR_INVALID_SELECTOR, /**< Device selection descriptor was invalid. */

    /* Context / queue errors. */
    CVL_CL_ERR_CONTEXT, /**< clCreateContext failed. */
    CVL_CL_ERR_QUEUE,   /**< clCreateCommandQueue failed. */

    /* Program / kernel errors. */
    CVL_CL_ERR_PROGRAM,       /**< clCreateProgramWithSource failed. */
    CVL_CL_ERR_PROGRAM_BUILD, /**< clBuildProgram failed (log available via cvl_cl_program_build_log). */
    CVL_CL_ERR_KERNEL,        /**< clCreateKernel failed. */
    CVL_CL_ERR_KERNEL_ARG,    /**< clSetKernelArg failed. */

    /* Buffer errors. */
    CVL_CL_ERR_BUFFER,      /**< clCreateBuffer failed. */
    CVL_CL_ERR_BUFFER_SIZE, /**< Requested buffer size exceeds limits. */
    CVL_CL_ERR_BUFFER_MAP,  /**< clEnqueueMapBuffer / Unmap failed. */

    /* Command / event errors. */
    CVL_CL_ERR_NDRANGE,    /**< clEnqueueNDRangeKernel failed. */
    CVL_CL_ERR_READ_WRITE, /**< clEnqueueReadBuffer / WriteBuffer failed. */
    CVL_CL_ERR_COPY,       /**< clEnqueueCopyBuffer failed. */
    CVL_CL_ERR_EVENT,      /**< Event operation failed. */
    CVL_CL_ERR_FINISH,     /**< clFinish / clFlush failed. */
    CVL_CL_ERR_BARRIER,    /**< clEnqueueBarrierWithWaitList failed. */

    /* Host-side errors. */
    CVL_CL_ERR_MEMORY,        /**< Host allocation (malloc) failed. */
    CVL_CL_ERR_INVALID_PARAM, /**< NULL pointer, bad size, etc. */
    CVL_CL_ERR_NOT_FOUND,     /**< Entity not found (e.g. kernel name, platform substring). */
    CVL_CL_ERR_INTERNAL,      /**< Unexpected / unknown error. */
} cvl_cl_status_t;

/**
 * @brief Return a human-readable string for a status code.
 * @param status Status code.
 * @return Pointer to a static string (do not free).
 */
const char *cvl_cl_status_str(cvl_cl_status_t status);

/* ------------------------------------------------------------------ */
/* OpenCL error → cvl_cl_status_t mapping                             */
/* ------------------------------------------------------------------ */

/**
 * @brief Map a raw OpenCL cl_int error to cvl_cl_status_t.
 *
 * Covers the full CL error space (CL_SUCCESS … CL_INVALID_OPERATION
 * and extension codes) by grouping related errors.  Unknown codes
 * map to CVL_CL_ERR_INTERNAL.
 *
 * @param err Raw cl_int from an OpenCL API call.
 * @return Canonical cvl_cl_status_t.
 */
cvl_cl_status_t cvl_cl_status_from_cl_int(int err);

/* ------------------------------------------------------------------ */
/* CVL_CL_CHECK macro                                                  */
/*                                                                     */
/* Wraps a cl_int-returning OpenCL call.  On failure converts the      */
/* error to cvl_cl_status_t and jumps to a label (typically cleanup).  */
/*                                                                     */
/* Usage:                                                              */
/*   cvl_cl_status_t status = CVL_CL_SUCCESS;                          */
/*   CVL_CL_CHECK(clSetKernelArg(...), cleanup);                       */
/*   ...                                                               */
/*   cleanup:                                                          */
/*     return status;                                                  */
/* ------------------------------------------------------------------ */

#define CVL_CL_CHECK(stmt, label)                                                                                      \
    do                                                                                                                 \
    {                                                                                                                  \
        const int _cvl_cl_err_ = (stmt);                                                                               \
        if (_cvl_cl_err_ != 0)                                                                                         \
        {                                                                                                              \
            status = cvl_cl_status_from_cl_int(_cvl_cl_err_);                                                          \
            goto label;                                                                                                \
        }                                                                                                              \
    } while (0)

/* ------------------------------------------------------------------ */
/* CVL_CL_CHECK_RAW — direct cl_int check without macro capture        */
/*                                                                     */
/* Use this when stmt is a compound expression or when you need        */
/* the raw cl_int after the check (e.g., clBuildProgram which sets     */
/* CL_BUILD_PROGRAM_FAILURE but returns CL_SUCCESS).                   */
/* ------------------------------------------------------------------ */

#define CVL_CL_CHECK_RAW(cl_err, label)                                                                                \
    do                                                                                                                 \
    {                                                                                                                  \
        if ((cl_err) != 0)                                                                                             \
        {                                                                                                              \
            status = cvl_cl_status_from_cl_int(cl_err);                                                                \
            goto label;                                                                                                \
        }                                                                                                              \
    } while (0)

/* ------------------------------------------------------------------ */
/* Guarded assertions (cpyutl-style)                                   */
/* ------------------------------------------------------------------ */

#ifdef CVL_CL_ENABLE_ASSERTS
#include <stdio.h>
#include <stdlib.h>

#define CVL_CL_ASSERT(cond, fmt, ...)                                                                                  \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(cond))                                                                                                   \
        {                                                                                                              \
            fprintf(stderr, "%s:%d (%s): Assertion \"" #cond "\" failed: " fmt "\n", __FILE__, __LINE__, __func__,     \
                    ##__VA_ARGS__);                                                                                    \
            abort();                                                                                                   \
        }                                                                                                              \
    } while (0)

#else
#define CVL_CL_ASSERT(cond, fmt, ...) ((void)0)
#endif

/* ------------------------------------------------------------------ */
/* Opaque type forward declarations                                    */
/* ------------------------------------------------------------------ */

typedef struct cvl_cl_device_t cvl_cl_device_t;
typedef struct cvl_cl_ctx_t cvl_cl_ctx_t;
typedef struct cvl_cl_queue_t cvl_cl_queue_t;
typedef struct cvl_cl_program_t cvl_cl_program_t;
typedef struct cvl_cl_kernel_t cvl_cl_kernel_t;
typedef struct cvl_cl_buffer_t cvl_cl_buffer_t;
typedef struct cvl_cl_event_t cvl_cl_event_t;
