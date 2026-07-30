#pragma once
/*
 * Capacity-tracked OpenCL device buffer.
 *
 * The buffer grows on demand (never shrinks).  When reserve is called
 * with a larger capacity, a new cl_mem is created, old content is
 * copied via clEnqueueCopyBuffer (if the queue is provided), and the
 * old buffer is released.
 *
 * This avoids repeated re-allocation when buffer sizes fluctuate
 * between kernel invocations.
 */

#include "cvl_cl_common.h"
#include "cvl_cl_ctx.h"

#include <CL/cl.h>

/* ------------------------------------------------------------------ */
/* Buffer descriptor                                                  */
/* ------------------------------------------------------------------ */

typedef enum
{
    CVL_CL_BUF_READ_ONLY,  /**< CL_MEM_READ_ONLY. */
    CVL_CL_BUF_WRITE_ONLY, /**< CL_MEM_WRITE_ONLY. */
    CVL_CL_BUF_READ_WRITE, /**< CL_MEM_READ_WRITE. */
} cvl_cl_buffer_access_t;

/**
 * @brief Descriptor for creating a new buffer.
 *
 * If @p host_ptr is non-NULL and @p use_host_ptr is false, the data
 * is copied via CL_MEM_COPY_HOST_PTR (the default for initial data).
 * If @p use_host_ptr is true, CL_MEM_USE_HOST_PTR is passed instead
 * (the host memory must remain valid for the buffer's lifetime).
 */
typedef struct
{
    cvl_cl_buffer_access_t access;
    size_t size_bytes;    /**< Logical size in bytes. */
    const void *host_ptr; /**< Optional initial data (or NULL for uninitialised). */
    bool use_host_ptr;    /**< If true, use CL_MEM_USE_HOST_PTR.  Default (false) uses CL_MEM_COPY_HOST_PTR. */
} cvl_cl_buffer_desc_t;

/* ------------------------------------------------------------------ */
/* Buffer handle                                                      */
/* ------------------------------------------------------------------ */

struct cvl_cl_buffer_t
{
    cl_mem mem;
    size_t capacity; /**< Allocated bytes (>= size). */
    size_t size;     /**< Logical bytes in use. */
    cvl_cl_buffer_access_t access;
};

/**
 * @brief Create a device buffer.
 *
 * @param ctx   Context (must outlive the buffer).
 * @param desc  Buffer descriptor.
 * @param out   Filled with the new buffer on success.
 * @return CVL_CL_SUCCESS or error.
 */
cvl_cl_status_t cvl_cl_buffer_create(const cvl_cl_ctx_t *ctx, const cvl_cl_buffer_desc_t *desc, cvl_cl_buffer_t *out);

/**
 * @brief Grow (or keep) the buffer to at least @p new_capacity bytes.
 *
 * If @p new_capacity > current capacity, allocates a new cl_mem,
 * copies old contents via clEnqueueCopyBuffer, releases the old
 * buffer.
 *
 * @param buf          Buffer to resize.
 * @param ctx          Context (for creating the new buffer).
 * @param queue        Queue for the copy operation (may be NULL if capacity <= current capacity).
 * @param new_capacity Minimum capacity in bytes.
 * @return CVL_CL_SUCCESS or error.
 */
cvl_cl_status_t cvl_cl_buffer_reserve(cvl_cl_buffer_t *buf, const cvl_cl_ctx_t *ctx, cvl_cl_queue_t *queue,
                                      size_t new_capacity);

/**
 * @brief Destroy a buffer.
 *
 * @param buf Buffer to destroy (may be NULL).
 */
void cvl_cl_buffer_destroy(cvl_cl_buffer_t *buf);

/* ------------------------------------------------------------------ */
/* Accessors                                                          */
/* ------------------------------------------------------------------ */

/** @brief Return the raw cl_mem handle. */
static inline cl_mem cvl_cl_buffer_mem(const cvl_cl_buffer_t *buf)
{
    return buf ? buf->mem : NULL;
}

/** @brief Return the allocated capacity in bytes. */
static inline size_t cvl_cl_buffer_capacity(const cvl_cl_buffer_t *buf)
{
    return buf ? buf->capacity : 0;
}

/** @brief Return the logical size in bytes. */
static inline size_t cvl_cl_buffer_size(const cvl_cl_buffer_t *buf)
{
    return buf ? buf->size : 0;
}

/** @brief Return the access mode. */
static inline cvl_cl_buffer_access_t cvl_cl_buffer_access(const cvl_cl_buffer_t *buf)
{
    return buf ? buf->access : CVL_CL_BUF_READ_WRITE;
}
