#pragma once
/*
 * OpenCL context and command-queue management.
 *
 * A @ref cvl_cl_ctx_t wraps a cl_context and holds an array of queues.
 * Each @ref cvl_cl_queue_t wraps a single in-order cl_command_queue
 * by default (out-of-order available via properties).
 *
 * Lifetime: device → context → queues → (use) → destroy queues → destroy context.
 */

#include "cvl_cl_common.h"
#include "cvl_cl_device.h"

#include <CL/cl.h>

/* ------------------------------------------------------------------ */
/* Context                                                            */
/* ------------------------------------------------------------------ */

struct cvl_cl_ctx_t
{
    cl_context context;
    const cvl_cl_device_t *device; /**< Borrowed reference — caller keeps device alive. */
};

/**
 * @brief Create an OpenCL context for a single device.
 *
 * @param device  Device handle (must outlive the context).
 * @param out_ctx Filled with the new context on success.
 * @return CVL_CL_SUCCESS or error.
 */
cvl_cl_status_t cvl_cl_ctx_create(const cvl_cl_device_t *device, cvl_cl_ctx_t *out_ctx);

/**
 * @brief Destroy a context.
 *
 * @param ctx Context to destroy (may be NULL).
 */
void cvl_cl_ctx_destroy(cvl_cl_ctx_t *ctx);

/** @brief Return the raw cl_context. */
static inline cl_context cvl_cl_ctx_context(const cvl_cl_ctx_t *ctx)
{
    return ctx ? ctx->context : NULL;
}

/** @brief Return the device associated with this context. */
static inline const cvl_cl_device_t *cvl_cl_ctx_device(const cvl_cl_ctx_t *ctx)
{
    return ctx ? ctx->device : NULL;
}

/* ------------------------------------------------------------------ */
/* Command Queue                                                      */
/* ------------------------------------------------------------------ */

typedef struct
{
    bool out_of_order; /**< Enable out-of-order execution (requires CL 2.0+ or cl_khr_command_buffer). */
    bool profiling;    /**< Enable CL_QUEUE_PROFILING_ENABLE. */
} cvl_cl_queue_props_t;

struct cvl_cl_queue_t
{
    cl_command_queue queue;
    const cvl_cl_ctx_t *ctx; /**< Borrowed reference — caller keeps ctx alive. */
};

/**
 * @brief Create a command queue.
 *
 * @param ctx     Context (must outlive the queue).
 * @param props   Queue properties (pass NULL for default in-order, no profiling).
 * @param out_q   Filled with the new queue on success.
 * @return CVL_CL_SUCCESS or error.
 */
cvl_cl_status_t cvl_cl_queue_create(const cvl_cl_ctx_t *ctx, const cvl_cl_queue_props_t *props, cvl_cl_queue_t *out_q);

/**
 * @brief Destroy a command queue.
 *
 * @param q Queue to destroy (may be NULL).
 */
void cvl_cl_queue_destroy(cvl_cl_queue_t *q);

/** @brief Return the raw cl_command_queue. */
static inline cl_command_queue cvl_cl_queue_queue(const cvl_cl_queue_t *q)
{
    return q ? q->queue : NULL;
}

/** @brief Return the context this queue belongs to. */
static inline const cvl_cl_ctx_t *cvl_cl_queue_ctx(const cvl_cl_queue_t *q)
{
    return q ? q->ctx : NULL;
}
