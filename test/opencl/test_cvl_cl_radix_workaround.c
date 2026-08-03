/*
 * test_cvl_cl_radix_workaround.c — validation of the Intel NEO CPU backend
 * detection and the radix policy API (cvl_cl_radix_policy_t) on the GPU
 * tree builder.
 *
 * Background: the Intel NEO CPU OpenCL backend (experimental "OpenCL 3.0
 * (Build 0)" device) miscompiles the original per-work-group radix kernels
 * (data-dependent __local indexing → runtime heap corruption).  Kernel-only
 * workarounds were also observed to crash the JIT with layout-dependent
 * probability, so the workaround mode performs the radix sort on the HOST
 * instead (stable qsort, no device kernels for the sort).  See
 * intel-neo-cpu-bug.md at the repository root for the full report.
 *
 * This test:
 *   1. Checks cvl_cl_device_is_intel_neo_cpu() against the actual device.
 *   2. Exercises the radix policy API:
 *      - AUTO resolves to the host-side sort iff the NEO CPU backend is
 *        detected,
 *      - ORIGINAL is refused on the NEO CPU backend with
 *        CVL_CL_ERR_UNSUPPORTED_DEVICE (and allowed elsewhere),
 *      - WORKAROUND always selects the host-side sort,
 *      - invalid policies are rejected.
 *
 * The end-to-end sort/tree-build correctness on the NEO CPU backend is
 * covered by test_cvl_cl_gpu_build's CPU fallback branch.
 */

#ifdef CVL_OPENCL

#include "../test_common.h"
#include "cvl_cl.h"
#include "cvl_cl_compute.h"
#include "cvl_cl_gpu_tree_build.h"
#include "cvl_cl_test_common.h"

/* ------------------------------------------------------------------ */
/*  Main                                                               */
/* ------------------------------------------------------------------ */

int main(void)
{
    cvl_cl_status_t status = CVL_CL_SUCCESS;
    cvl_cl_device_t device = {0};
    cl_context ctx = NULL;
    cl_command_queue queue = NULL;
    cvl_cl_compute_t comp = {0};
    int ret = 1;

    unsigned count = 0;
    status = cvl_cl_device_discover((cvl_cl_platform_filter_t){0}, 1, &device, CL_DEVICE_TYPE_ALL, &count);
    if (status != CVL_CL_SUCCESS || count == 0)
    {
        fprintf(stderr, "No OpenCL device found -- skipping radix workaround test.\n");
        return 0;
    }

    const bool neo_cpu = cvl_cl_device_is_intel_neo_cpu(&device);
    fprintf(stderr, "Device: %s (Intel NEO CPU backend: %s)\n", device.info.name, neo_cpu ? "yes" : "no");

    CVL_CL_CHECK(cvl_cl_ctx_create(&device, &ctx), cleanup);
    CVL_CL_CHECK(cvl_cl_queue_create(ctx, device.id, NULL, &queue), cleanup);

    /* ----------------------------------------------------------------- */
    /* Init the compute backend from the embedded kernel packs:          */
    /* all 6 BH_BUILD kernels + bh_flat_eval.                            */
    /* ----------------------------------------------------------------- */
    {
        const char *kernels[] = {"kernel_morton",   "kernel_radix_hist",  "kernel_radix_scatter",
                                 "kernel_boundary", "kernel_fill_leaves", "kernel_build_internal",
                                 "bh_flat_eval"};
        const unsigned n_kernels = sizeof(kernels) / sizeof(kernels[0]);

        CVL_CL_CHECK(cvl_cl_compute_init(&comp, ctx, queue, &device, CVL_CL_PRECISION_FP64, kernels, n_kernels),
                     cleanup);
    }

    /* ----------------------------------------------------------------- */
    /* Radix policy API on the GPU tree builder                           */
    /* ----------------------------------------------------------------- */
    {
        cvl_cl_gpu_tree_build_t builder = {0};

        /* Default AUTO init.  On the NEO CPU backend the original kernels
         * are refused at init (they would corrupt the runtime), so AUTO must
         * resolve to the host-side sort. */
        status = cvl_cl_gpu_tree_build_init(&builder, &comp, 6, 8, 0);
        CVL_CL_CHECK(status, cleanup);
        TEST_ASSERT(builder.use_host_radix == neo_cpu, "AUTO: use_host_radix=%d but intel_neo_cpu=%d",
                    builder.use_host_radix, neo_cpu);
        printf("AUTO resolved to %s.\n", builder.use_host_radix ? "host-side sort" : "device kernels");

        /* ORIGINAL on NEO CPU must be refused; elsewhere allowed. */
        status = cvl_cl_gpu_tree_build_set_radix_policy(&builder, CVL_CL_RADIX_POLICY_ORIGINAL);
        if (neo_cpu)
        {
            TEST_ASSERT(status == CVL_CL_ERR_UNSUPPORTED_DEVICE,
                        "ORIGINAL policy on NEO CPU expected CVL_CL_ERR_UNSUPPORTED_DEVICE, got %d", status);
            printf("ORIGINAL policy correctly refused on Intel NEO CPU backend.\n");
        }
        else
        {
            CVL_CL_CHECK(status, cleanup);
            TEST_ASSERT(builder.use_host_radix == false, "ORIGINAL: use_host_radix should be false");
        }

        /* WORKAROUND always selects the host-side sort. */
        status = cvl_cl_gpu_tree_build_set_radix_policy(&builder, CVL_CL_RADIX_POLICY_WORKAROUND);
        CVL_CL_CHECK(status, cleanup);
        TEST_ASSERT(builder.use_host_radix == true, "WORKAROUND: use_host_radix should be true");

        /* Back to AUTO. */
        status = cvl_cl_gpu_tree_build_set_radix_policy(&builder, CVL_CL_RADIX_POLICY_AUTO);
        CVL_CL_CHECK(status, cleanup);
        TEST_ASSERT(builder.use_host_radix == neo_cpu, "AUTO after switch: mismatch");

        /* Invalid policy value. */
        status = cvl_cl_gpu_tree_build_set_radix_policy(&builder, (cvl_cl_radix_policy_t)99);
        TEST_ASSERT(status == CVL_CL_ERR_INVALID_PARAM, "Invalid policy expected CVL_CL_ERR_INVALID_PARAM, got %d",
                    status);

        /* Setter on an uninitialised builder must be refused. */
        {
            cvl_cl_gpu_tree_build_t fresh = {0};
            status = cvl_cl_gpu_tree_build_set_radix_policy(&fresh, CVL_CL_RADIX_POLICY_WORKAROUND);
            TEST_ASSERT(status == CVL_CL_ERR_INVALID_PARAM,
                        "Setter on uninitialised builder expected CVL_CL_ERR_INVALID_PARAM, got %d", status);
        }

        cvl_cl_gpu_tree_build_destroy(&builder);
    }

    printf("All radix workaround tests passed.\n");
    ret = 0;

cleanup:
    cvl_cl_compute_destroy(&comp);
    cvl_cl_queue_destroy(&queue);
    cvl_cl_ctx_destroy(&ctx);
    return ret;
}

#else

#include <stdio.h>
int main(void)
{
    printf("OpenCL not available -- skipping test.\n");
    return 0;
}

#endif /* CVL_OPENCL */
