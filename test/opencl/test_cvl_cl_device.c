#ifdef CVL_OPENCL

#include "../test_common.h"
#include "cvl_cl.h"

int main(void)
{
    cvl_cl_status_t status = CVL_CL_SUCCESS;
    cvl_cl_device_t gpu_dev = {0};
    cvl_cl_device_t cpu_dev = {0};
    unsigned count = 0;

    /* ---- GPU discovery ---- */
    status = cvl_cl_device_discover(
        (cvl_cl_device_sel_t[]){
            {.type = CVL_CL_DEVICE_SEL_TYPE, .device_type = CL_DEVICE_TYPE_GPU},
            {},
        },
        1, &count, &gpu_dev, NULL);

    if (status == CVL_CL_SUCCESS && count > 0)
    {
        TEST_ASSERT(gpu_dev.info.name != NULL, "GPU device name is NULL");
        TEST_ASSERT(gpu_dev.info.vendor != NULL, "GPU device vendor is NULL");
        TEST_ASSERT(gpu_dev.info.max_work_group_size > 0, "GPU max_work_group_size should be > 0, got %zu",
                    gpu_dev.info.max_work_group_size);
        TEST_ASSERT(gpu_dev.info.available, "GPU device should be available");
    }
    /* else: no GPU on this system – skip GPU checks */

    /* ---- CPU discovery ---- */
    count = 0;
    status = cvl_cl_device_discover(
        (cvl_cl_device_sel_t[]){
            {.type = CVL_CL_DEVICE_SEL_TYPE, .device_type = CL_DEVICE_TYPE_CPU},
            {},
        },
        1, &count, &cpu_dev, NULL);

    if (status == CVL_CL_SUCCESS && count > 0)
    {
        TEST_ASSERT(cpu_dev.info.name != NULL, "CPU device name is NULL");
        TEST_ASSERT(cpu_dev.info.vendor != NULL, "CPU device vendor is NULL");
        TEST_ASSERT(cpu_dev.info.max_work_group_size > 0, "CPU max_work_group_size should be > 0, got %zu",
                    cpu_dev.info.max_work_group_size);
        TEST_ASSERT(cpu_dev.info.available, "CPU device should be available");
    }
    /* else: no CPU on this system – skip CPU checks */

    /* ---- NULL destroy (must not crash) ---- */
    cvl_cl_device_destroy(NULL);

    /* ---- Destroy valid handles ---- */
    cvl_cl_device_destroy(&gpu_dev);
    cvl_cl_device_destroy(&cpu_dev);

    return 0;
}

#else

#include <stdio.h>
int main(void)
{
    printf("OpenCL not available – skipping test.\n");
    return 0;
}

#endif
