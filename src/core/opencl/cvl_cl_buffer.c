#include "cvl_cl_buffer.h"

#include <assert.h>
#include <string.h>

cvl_cl_status_t cvl_cl_buffer_create(cl_context ctx, const cvl_cl_buffer_desc_t *desc, cvl_cl_buffer_t *out)
{
    assert(desc && out);

    out->mem = NULL;
    out->capacity = 0;
    out->size = 0;
    out->access = desc->access;

    if (desc->size_bytes == 0)
    {
        /* Zero-size buffer: valid (size=0, capacity=0, mem=NULL). */
        return CVL_CL_SUCCESS;
    }

    cl_mem_flags flags = cvl_cl_buffer_access_to_flags(desc->access);
    if (desc->host_ptr)
    {
        if (desc->use_host_ptr)
            flags |= CL_MEM_USE_HOST_PTR;
        else
            flags |= CL_MEM_COPY_HOST_PTR;
    }

    cl_int err;
    cl_mem mem = clCreateBuffer(ctx, flags, desc->size_bytes, (void *)desc->host_ptr, &err);
    if (err != CL_SUCCESS)
    {
        memset(out, 0, sizeof(*out));
        return cvl_cl_status_from_cl_int(err);
    }

    out->mem = mem;
    out->capacity = desc->size_bytes;
    out->size = desc->size_bytes;
    return CVL_CL_SUCCESS;
}

cvl_cl_status_t cvl_cl_buffer_reserve(cvl_cl_buffer_t *buf, cl_context ctx, cl_command_queue queue, size_t new_capacity,
                                      cl_event *out_event)
{
    assert(buf);
    if (out_event)
        *out_event = NULL;

    if (new_capacity <= buf->capacity)
        return CVL_CL_SUCCESS;

    cl_mem_flags flags = cvl_cl_buffer_access_to_flags(buf->access);
    cl_int err;
    cl_mem new_mem = clCreateBuffer(ctx, flags, new_capacity, NULL, &err);
    if (err != CL_SUCCESS)
        return cvl_cl_status_from_cl_int(err);

    /* Copy old contents if present.  The old cl_mem is retained by the
     * enqueued copy command, so releasing our reference right after the
     * enqueue is safe. */
    if (buf->mem != NULL && buf->size > 0)
    {
        assert(queue != NULL);
        cl_event copy_event = NULL;
        err = clEnqueueCopyBuffer(queue, buf->mem, new_mem, 0, 0, buf->size, 0, NULL, out_event ? &copy_event : NULL);
        if (err != CL_SUCCESS)
        {
            clReleaseMemObject(new_mem);
            return cvl_cl_status_from_cl_int(err);
        }
        if (out_event)
            *out_event = copy_event;
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
    assert(buf);
    if (buf->mem != NULL)
        clReleaseMemObject(buf->mem);
    memset(buf, 0, sizeof(*buf));
}
