#pragma once
/*
 * High-level compute backend for OpenCL-accelerated tree methods.
 *
 * Owns a compiled program (from a multi-kernel source string), a
 * registry of named kernel handles, and cached device capabilities
 * (precision mode, unified memory flag, work-group limits).
 *
 * The context, queue, and device are borrowed - they must outlive
 * the compute backend.
 *
 * Typical usage:
 * @code
 *   cvl_cl_compute_t comp;
 *   cvl_cl_compute_init(&comp, &ctx, &queue, &device,
 *       CVL_CL_PRECISION_FP64, multi_kernel_source,
 *       (const char*[]){"direct_sum", "bh_eval"}, 2);
 *
 *   cvl_cl_kernel_t *k = cvl_cl_compute_kernel(&comp, "direct_sum");
 *   cvl_cl_kernel_set_args(k, ...);
 *   ...
 *   cvl_cl_compute_destroy(&comp);
 * @endcode
 */

#include "cvl_cl_common.h"
#include "cvl_cl_ctx.h"
#include "cvl_cl_device.h"
#include "cvl_cl_future.h"
#include "cvl_cl_kernel.h"
#include "cvl_cl_staging_buffer.h"

#include <CL/cl.h>

/* ------------------------------------------------------------------ */
/* Compute backend handle                                             */
/* ------------------------------------------------------------------ */

/** @brief Maximum number of kernels in a single program. */
enum
{
    CVL_CL_COMPUTE_MAX_KERNELS = 16
};

typedef struct
{
    /* Borrowed references (caller keeps these alive). */
    const cvl_cl_ctx_t *ctx;
    cvl_cl_queue_t *queue;
    const cvl_cl_device_t *device;

    /* Owned resources. */
    cvl_cl_program_t program;
    cvl_cl_kernel_t kernels[CVL_CL_COMPUTE_MAX_KERNELS];
    const char *kernel_names[CVL_CL_COMPUTE_MAX_KERNELS]; /**< Names (borrowed - caller keeps strings alive). */
    unsigned n_kernels;

    /* Cached device capabilities. */
    cvl_cl_precision_t precision;
    bool unified_memory;
    size_t max_work_group_size;

    /* Valid flag. */
    bool initialized;
} cvl_cl_compute_t;

/**
 * @brief Initialise the compute backend.
 *
 * Compiles the @p kernel_source into a program and extracts the
 * named kernels into the kernel registry.
 *
 * Device capabilities are queried once and cached.
 *
 * @param comp           Uninitialised compute handle.
 * @param ctx            Context (borrowed - must outlive comp).
 * @param queue          Queue (borrowed - must outlive comp).
 * @param device         Device handle (borrowed - must outlive comp).
 * @param precision      FP32 or FP64.
 * @param kernel_source  Multi-kernel OpenCL C source string.
 * @param kernel_names   Array of kernel function names to extract.
 * @param n_kernels      Number of entries in @p kernel_names.
 * @return CVL_CL_SUCCESS or error.
 */
cvl_cl_status_t cvl_cl_compute_init(cvl_cl_compute_t *comp, const cvl_cl_ctx_t *ctx, cvl_cl_queue_t *queue,
                                    const cvl_cl_device_t *device, cvl_cl_precision_t precision,
                                    const char *kernel_source, const char *kernel_names[], unsigned n_kernels);

/**
 * @brief Look up a kernel by name.
 *
 * @param comp  Compute backend.
 * @param name  Kernel function name.
 * @return Pointer to the kernel handle, or NULL if not found.
 */
cvl_cl_kernel_t *cvl_cl_compute_kernel(cvl_cl_compute_t *comp, const char *name);

/* ------------------------------------------------------------------ */
/* Accessors                                                          */
/* ------------------------------------------------------------------ */

/** @brief Whether the device has unified host/device memory. */
static inline bool cvl_cl_compute_unified_memory(const cvl_cl_compute_t *comp)
{
    return comp ? comp->unified_memory : false;
}

/** @brief Return the backend's precision. */
static inline cvl_cl_precision_t cvl_cl_compute_precision(const cvl_cl_compute_t *comp)
{
    return comp ? comp->precision : CVL_CL_PRECISION_DEFAULT;
}

/** @brief Return the queue. */
static inline cvl_cl_queue_t *cvl_cl_compute_queue(cvl_cl_compute_t *comp)
{
    return comp ? comp->queue : NULL;
}

/** @brief Return the context. */
static inline const cvl_cl_ctx_t *cvl_cl_compute_ctx(const cvl_cl_compute_t *comp)
{
    return comp ? comp->ctx : NULL;
}

/** @brief Return the device handle. */
static inline const cvl_cl_device_t *cvl_cl_compute_device(const cvl_cl_compute_t *comp)
{
    return comp ? comp->device : NULL;
}

/** @brief Return the max work-group size. */
static inline size_t cvl_cl_compute_max_work_group_size(const cvl_cl_compute_t *comp)
{
    return comp ? comp->max_work_group_size : 0;
}

/** @brief Whether the backend is initialised. */
static inline bool cvl_cl_compute_initialized(const cvl_cl_compute_t *comp)
{
    return comp && comp->initialized;
}

/**
 * @brief Destroy the compute backend.
 *
 * Releases program and all registered kernels.
 * Does NOT release the borrowed ctx, queue, or device.
 */
void cvl_cl_compute_destroy(cvl_cl_compute_t *comp);
