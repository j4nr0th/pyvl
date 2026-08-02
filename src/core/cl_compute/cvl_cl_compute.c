#include "cvl_cl_compute.h"

#include <string.h> /* strcmp */

#include <stdlib.h>
#include <string.h>

cvl_cl_status_t cvl_cl_compute_init(cvl_cl_compute_t *comp, const cvl_cl_ctx_t *ctx, cvl_cl_queue_t *queue,
                                    const cvl_cl_device_t *device, cvl_cl_precision_t precision,
                                    const char *kernel_source, const char *kernel_names[], unsigned n_kernels)
{
    if (!comp || !ctx || !queue || !device || !kernel_source || (!kernel_names && n_kernels > 0))
        return CVL_CL_ERR_INVALID_PARAM;
    if (n_kernels > CVL_CL_COMPUTE_MAX_KERNELS)
        return CVL_CL_ERR_INVALID_PARAM;

    memset(comp, 0, sizeof(*comp));
    comp->ctx = ctx;
    comp->queue = queue;
    comp->device = device;
    comp->precision = precision;
    comp->n_kernels = n_kernels;

    /* Cache device info. */
    {
        const cvl_cl_device_info_t *info = cvl_cl_device_info(device);
        if (info)
        {
            comp->max_work_group_size = info->max_work_group_size;
        }
    }

    /* Query unified memory. */
    {
        cl_bool unified = CL_FALSE;
        cl_int err =
            clGetDeviceInfo(cvl_cl_device_id(device), CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof(unified), &unified, NULL);
        comp->unified_memory = (err == CL_SUCCESS && unified);
    }

    /* Compile the program. */
    cvl_cl_program_desc_t desc = {
        .source_type = CVL_CL_PROGRAM_SOURCE_STRING,
        .source_string = kernel_source,
        .precision = precision,
    };
    cvl_cl_status_t st = cvl_cl_program_create(ctx, &desc, cvl_cl_device_id(device), &comp->program, NULL);
    if (st != CVL_CL_SUCCESS)
        return st;

    /* Extract kernels. */
    for (unsigned i = 0; i < n_kernels; ++i)
    {
        comp->kernel_names[i] = kernel_names[i];
        st = cvl_cl_kernel_create(&comp->program, kernel_names[i], &comp->kernels[i]);
        if (st != CVL_CL_SUCCESS)
        {
            cvl_cl_compute_destroy(comp);
            return st;
        }
    }

    comp->initialized = true;
    return CVL_CL_SUCCESS;
}

cvl_cl_kernel_t *cvl_cl_compute_kernel(cvl_cl_compute_t *comp, const char *name)
{
    if (!comp || !name || !comp->initialized)
        return NULL;

    for (unsigned i = 0; i < comp->n_kernels; ++i)
    {
        if (strcmp(comp->kernel_names[i], name) == 0)
            return &comp->kernels[i];
    }

    return NULL;
}

void cvl_cl_compute_destroy(cvl_cl_compute_t *comp)
{
    if (!comp)
        return;

    /* Destroy kernels in reverse order, then program. */
    for (unsigned i = comp->n_kernels; i > 0; --i)
        cvl_cl_kernel_destroy(&comp->kernels[i - 1]);

    cvl_cl_program_destroy(&comp->program);

    memset(comp, 0, sizeof(*comp));
}
