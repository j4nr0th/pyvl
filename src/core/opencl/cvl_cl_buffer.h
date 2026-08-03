#pragma once
/*
 * Capacity-tracked OpenCL device buffer.
 *
 * The buffer grows on demand (never shrinks).  When reserve is called
 * with a larger capacity, a new cl_mem is created, old content is
 * copied via clEnqueueCopyBuffer, and the old buffer is released.
 *
 * This avoids repeated re-allocation when buffer sizes fluctuate
 * between kernel invocations.
 */

#include "cvl_cl_common.h"

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

/** @brief Map a buffer access mode to its cl_mem_flags bitmask. */
static inline cl_mem_flags cvl_cl_buffer_access_to_flags(cvl_cl_buffer_access_t access)
{
    switch (access)
    {
    case CVL_CL_BUF_READ_ONLY:
        return CL_MEM_READ_ONLY;
    case CVL_CL_BUF_WRITE_ONLY:
        return CL_MEM_WRITE_ONLY;
    case CVL_CL_BUF_READ_WRITE:
    default:
        return CL_MEM_READ_WRITE;
    }
}

typedef struct cvl_cl_buffer_t cvl_cl_buffer_t;

/**
 * @brief Create a device buffer.
 *
 * @param ctx   Context (must outlive the buffer).
 * @param desc  Buffer descriptor.
 * @param out   Filled with the new buffer on success.
 * @return CVL_CL_SUCCESS or error.
 */
cvl_cl_status_t cvl_cl_buffer_create(cl_context ctx, const cvl_cl_buffer_desc_t *desc, cvl_cl_buffer_t *out);

/**
 * @brief Grow (or keep) the buffer to at least @p new_capacity bytes.
 *
 * If @p new_capacity > current capacity, allocates a new cl_mem,
 * copies old contents via clEnqueueCopyBuffer on @p queue, releases
 * the old buffer.  If the buffer is empty (no old contents) the copy
 * is skipped and @p queue may be NULL.
 *
 * The copy is enqueued non-blocking (async).  If @p out_event is
 * non-NULL it receives an owned cl_event for the copy (caller must
 * release it with clReleaseEvent) so the growth can be synchronised
 * asynchronously; pass NULL to ignore.  The event is only produced
 * when a copy was actually enqueued (i.e. the buffer had content).
 *
 * For dependency-tracked growth that participates in a command chain
 * use cvl_cl_chain_grow_buffer (cvl_cl_chain.h) instead.
 *
 * @param buf          Buffer to resize.
 * @param ctx          Context (for creating the new buffer).
 * @param queue        Queue for the copy operation (may be NULL only when growing an empty buffer).
 * @param new_capacity Minimum capacity in bytes.
 * @param out_event    Optional owned event for the copy (may be NULL).
 * @return CVL_CL_SUCCESS or error.
 */
cvl_cl_status_t cvl_cl_buffer_reserve(cvl_cl_buffer_t *buf, cl_context ctx, cl_command_queue queue, size_t new_capacity,
                                      cl_event *out_event);

/**
 * @brief Destroy a buffer.
 *
 * @param buf Buffer to destroy (may be NULL).
 */
void cvl_cl_buffer_destroy(cvl_cl_buffer_t *buf);
