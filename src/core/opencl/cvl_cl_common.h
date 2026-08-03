#pragma once
/*
 * Shared types, status codes, and utility macros for the pyvl OpenCL wrapper.
 *
 * This is the single foundational header for all cvl_cl_* modules.
 * It provides:
 *   - cvl_cl_status_t  - typed error codes (cpyutl-style, never exit())
 *   - cvl_cl_status_str() - human-readable error description
 *   - cvl_cl_status_from_cl_int() - map OpenCL error codes
 *   - Cross-compilation macros (__global / restrict) for shared
 *     C/OpenCL-C headers
 *
 * This is an internal module: callers are expected to pass valid
 * pointers.  Contract violations are enforced with assert() rather
 * than NULL checks that cost a branch on every call.
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
/* Constants                                                          */
/* ------------------------------------------------------------------ */

/** @brief Maximum number of events in an OpenCL wait list. */
enum
{
    CL_MAX_WAIT_EVENTS = 16
};

/* ------------------------------------------------------------------ */
/* Status codes                                                        */
/* ------------------------------------------------------------------ */

typedef enum
{
    CVL_CL_SUCCESS = 0,

    /* Platform / device errors. */
    CVL_CL_ERR_PLATFORM,           /**< clGetPlatformIDs failed. */
    CVL_CL_ERR_DEVICE_NOT_FOUND,   /**< No matching device found (clGetDeviceIDs returned 0). */
    CVL_CL_ERR_DEVICE,             /**< Generic device error. */
    CVL_CL_ERR_INVALID_SELECTOR,   /**< Device selection descriptor was invalid. */
    CVL_CL_ERR_UNSUPPORTED_DEVICE, /**< Operation refused on the selected device (e.g. original kernels on the
                                        Intel NEO CPU backend - see intel-neo-cpu-bug.md). */

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

/* ------------------------------------------------------------------ */
/* Precision selection                                                 */
/* ------------------------------------------------------------------ */

/**
 * @brief Floating-point precision used when compiling kernel programs.
 *
 * Kernels compiled with FP32 use `float` as the @c real_t type; FP64
 * uses `double`.  FP64 is the default (matching the host-side real_t).
 *
 * When FP64 is selected, the @c cl_khr_fp64 extension pragma is
 * automatically injected into the kernel source.
 *
 * Devices that lack @c cl_khr_fp64 support will fail to build programs
 * in FP64 mode (build log will indicate the missing extension).
 */
typedef enum
{
    CVL_CL_PRECISION_DEFAULT = 0, /**< FP64 (double) - matches host real_t. */
    CVL_CL_PRECISION_FP32,        /**< 32-bit float. */
    CVL_CL_PRECISION_FP64,        /**< 64-bit double (same as DEFAULT). */
} cvl_cl_precision_t;

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
/* Status → string                                                    */
/* ------------------------------------------------------------------ */
