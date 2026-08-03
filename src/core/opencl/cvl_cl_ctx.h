#pragma once
/*
 * OpenCL context and command-queue management.
 *
 * Thin helpers around the raw OpenCL types: cl_context and
 * cl_command_queue are created/destroyed here; everything else in the
 * wrapper layer operates on the raw handles directly.
 *
 * Lifetime: device → context → queues → (use) → destroy queues → destroy context.
 */

#include "cvl_cl_common.h"
#include "cvl_cl_device.h"

#include <CL/cl.h>

/* ------------------------------------------------------------------ */
/* Context                                                            */
/* ------------------------------------------------------------------ */

/**
 * @brief Create an OpenCL context for a single device.
 *
 * @param device  Device handle (must outlive the context).
 * @param out_ctx Filled with the new cl_context on success (NULL on failure).
 * @return CVL_CL_SUCCESS or error.
 */
cvl_cl_status_t cvl_cl_ctx_create(const cvl_cl_device_t *device, cl_context *out_ctx);

/**
 * @brief Destroy a context.
 *
 * @param ctx Context to destroy (may be NULL).
 */
void cvl_cl_ctx_destroy(cl_context *ctx);

/* ------------------------------------------------------------------ */
/* Command Queue                                                      */
/* ------------------------------------------------------------------ */

typedef struct
{
    bool out_of_order; /**< Enable out-of-order execution. */
    bool profiling;    /**< Enable CL_QUEUE_PROFILING_ENABLE. */
} cvl_cl_queue_props_t;

/**
 * @brief Create a command queue.
 *
 * Uses clCreateCommandQueueWithProperties when available (OpenCL 2.0+),
 * falling back to the 1.2 clCreateCommandQueue.
 *
 * @param ctx        Context the queue belongs to.
 * @param device_id  Device the queue targets.
 * @param props      Queue properties (NULL = default in-order, no profiling).
 * @param out_q      Filled with the new cl_command_queue on success (NULL on failure).
 * @return CVL_CL_SUCCESS or error.
 */
cvl_cl_status_t cvl_cl_queue_create(cl_context ctx, cl_device_id device_id, const cvl_cl_queue_props_t *props,
                                    cl_command_queue *out_q);

/**
 * @brief Destroy a command queue.
 *
 * @param q Queue to destroy (may be NULL).
 */
void cvl_cl_queue_destroy(cl_command_queue *q);
