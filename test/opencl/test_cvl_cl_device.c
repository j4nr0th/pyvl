#ifdef CVL_OPENCL

#include "../test_common.h"
#include "cvl_cl.h"
#include "cvl_cl_test_common.h"

int main(void)
{
    cvl_cl_status_t status = CVL_CL_SUCCESS;
    cvl_cl_device_t gpu_dev = {0};
    cvl_cl_device_t cpu_dev = {0};
    unsigned count = 0;

    /* ---- GPU discovery ---- */
    status = cvl_cl_device_discover((cvl_cl_platform_filter_t){0}, 1, &gpu_dev, CL_DEVICE_TYPE_GPU, &count);

    if (status == CVL_CL_SUCCESS && count > 0)
    {
        TEST_ASSERT(gpu_dev.info.name != NULL, "GPU device name is NULL");
        TEST_ASSERT(gpu_dev.info.vendor != NULL, "GPU device vendor is NULL");
        TEST_ASSERT(gpu_dev.info.max_work_group_size > 0, "GPU max_work_group_size should be > 0, got %zu",
                    gpu_dev.info.max_work_group_size);
        TEST_ASSERT(gpu_dev.info.available, "GPU device should be available");
        TEST_ASSERT(!cvl_cl_device_is_intel_neo_cpu(&gpu_dev), "GPU must not be detected as the Intel NEO CPU backend");
    }
    /* else: no GPU on this system - skip GPU checks */

    /* ---- CPU discovery ---- */
    count = 0;
    status = cvl_cl_device_discover((cvl_cl_platform_filter_t){0}, 1, &cpu_dev, CL_DEVICE_TYPE_CPU, &count);

    if (status == CVL_CL_SUCCESS && count > 0)
    {
        TEST_ASSERT(cpu_dev.info.name != NULL, "CPU device name is NULL");
        TEST_ASSERT(cpu_dev.info.vendor != NULL, "CPU device vendor is NULL");
        TEST_ASSERT(cpu_dev.info.max_work_group_size > 0, "CPU max_work_group_size should be > 0, got %zu",
                    cpu_dev.info.max_work_group_size);
        TEST_ASSERT(cpu_dev.info.available, "CPU device should be available");
    }
    /* else: no CPU on this system - skip CPU checks */

    /* ---- Multi-device discovery ----
     *
     * out_count reports the TOTAL number of matches, which may exceed
     * max_devices.  Only out_devices[0 .. min(total, max_devices)-1]
     * are valid to touch.
     */
    {
        cvl_cl_device_t devices[4];
        unsigned total = 0;

        status = cvl_cl_device_discover((cvl_cl_platform_filter_t){0}, 4, devices, CL_DEVICE_TYPE_ALL, &total);
        if (status == CVL_CL_SUCCESS)
        {
            const unsigned filled = total < 4 ? total : 4;
            TEST_ASSERT(filled > 0, "Discovery returned success but no devices");
            for (unsigned i = 0; i < filled; ++i)
                TEST_ASSERT(devices[i].id != NULL, "device %u id is NULL after discovery", i);
        }
        /* else: no OpenCL devices at all - skip multi-device checks */
    }

    return 0;
}

#else

#include <stdio.h>
int main(void)
{
    printf("OpenCL not available - skipping test.\n");
    return 0;
}

#endif
