#include "cvl_cl_buffer.h"

#include <stdlib.h>
#include <string.h>

/* Map buffer access to cl_mem_flags. */
static cl_mem_flags access_to_flags(cvl_cl_buffer_access_t access)
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

cvl_cl_status_t cvl_cl_buffer_create(const cvl_cl_ctx_t *ctx, const cvl_cl_buffer_desc_t *desc, cvl_cl_buffer_t *out)
{
    if (!ctx || !desc || !out || !ctx->context)
        return CVL_CL_ERR_INVALID_PARAM;

    out->mem = NULL;
    out->capacity = 0;
    out->size = 0;
    out->access = desc->access;

    if (desc->size_bytes == 0)
    {
        /* Zero-size buffer: still valid (size=0, capacity=0, mem=NULL). */
        return CVL_CL_SUCCESS;
    }

    cl_mem_flags flags = access_to_flags(desc->access);
    if (desc->host_ptr)
    {
        if (desc->use_host_ptr)
            flags |= CL_MEM_USE_HOST_PTR;
        else
            flags |= CL_MEM_COPY_HOST_PTR;
    }

    cl_int err;
    cl_mem mem = clCreateBuffer(ctx->context, flags, desc->size_bytes, (void *)desc->host_ptr, &err);
    if (err != CL_SUCCESS)
        return cvl_cl_status_from_cl_int(err);

    out->mem = mem;
    out->capacity = desc->size_bytes;
    out->size = desc->size_bytes;
    return CVL_CL_SUCCESS;
}

cvl_cl_status_t cvl_cl_buffer_reserve(cvl_cl_buffer_t *buf, const cvl_cl_ctx_t *ctx, cvl_cl_queue_t *queue,
                                      size_t new_capacity)
{
    if (!buf || !ctx)
        return CVL_CL_ERR_INVALID_PARAM;

    if (new_capacity <= buf->capacity)
        return CVL_CL_SUCCESS;

    cl_mem_flags flags = access_to_flags(buf->access);
    cl_int err;
    cl_mem new_mem = clCreateBuffer(ctx->context, flags, new_capacity, NULL, &err);
    if (err != CL_SUCCESS)
        return cvl_cl_status_from_cl_int(err);

    /* Copy old contents if present. */
    if (buf->mem != NULL && buf->size > 0 && queue != NULL)
    {
        err = clEnqueueCopyBuffer(queue->queue, buf->mem, new_mem, 0, 0, buf->size, 0, NULL, NULL);
        if (err != CL_SUCCESS)
        {
            clReleaseMemObject(new_mem);
            return cvl_cl_status_from_cl_int(err);
        }
    }

    /* Release old buffer. */
    if (buf->mem != NULL)
        clReleaseMemObject(buf->mem);

    buf->mem = new_mem;
    buf->capacity = new_capacity;
    /* size stays unchanged (the logical data size doesn't grow from reserve alone). */
    return CVL_CL_SUCCESS;
}

void cvl_cl_buffer_destroy(cvl_cl_buffer_t *buf)
{
    if (!buf)
        return;
    if (buf->mem)
    {
        clReleaseMemObject(buf->mem);
        buf->mem = NULL;
    }
    buf->capacity = 0;
    buf->size = 0;
}
