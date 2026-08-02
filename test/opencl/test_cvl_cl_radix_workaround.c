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

#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/*  Helpers for building concatenated kernel source                    */
/* ------------------------------------------------------------------ */

static size_t skip_include_concat(char *dst, size_t dst_cap, const char *src)
{
    size_t pos = 0;
    while (*src && pos < dst_cap - 1)
    {
        const char *nl = strchr(src, '\n');
        size_t line_len = nl ? (size_t)(nl - src + 1) : strlen(src);

        const char *trimmed = src;
        while (*trimmed == ' ' || *trimmed == '\t')
            ++trimmed;

        int is_include = (trimmed[0] == '#' && strncmp(trimmed + 1, "include", 7) == 0);
        int is_pragma_once = (trimmed[0] == '#' && strncmp(trimmed + 1, "pragma once", 11) == 0);

        if (!is_include && !is_pragma_once)
        {
            size_t copy = line_len < dst_cap - 1 - pos ? line_len : dst_cap - 1 - pos;
            memcpy(dst + pos, src, copy);
            pos += copy;
        }

        if (!nl)
            break;
        src = nl + 1;
    }
    dst[pos] = '\0';
    return pos;
}

static char *read_cl_source(const char *filename)
{
    char path[1024];
    int n = snprintf(path, sizeof path, "%s/%s", CVL_CL_SOURCE_DIR, filename);
    TEST_ASSERT(n > 0 && (size_t)n < sizeof path, "Path too long for %s", filename);
    return read_file_to_string(path, 65536);
}

/* Preamble identical to the other build tests (FP64). */
static const char *PREAMBLE = "#ifdef CVL_CL_REAL_FP32\n"
                              "typedef float real_t;\n"
                              "typedef struct { real_t x, y, z; } real3_t;\n"
                              "#else\n"
                              "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
                              "typedef double real_t;\n"
                              "typedef struct { real_t x, y, z; } real3_t;\n"
                              "#endif\n"
                              "\n";

/* ------------------------------------------------------------------ */
/*  Main                                                               */
/* ------------------------------------------------------------------ */

int main(void)
{
    cvl_cl_status_t status = CVL_CL_SUCCESS;
    cvl_cl_device_t device = {0};
    cvl_cl_ctx_t ctx = {0};
    cvl_cl_queue_t queue = {0};
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
    CVL_CL_CHECK(cvl_cl_queue_create(&ctx, NULL, &queue), cleanup);

    /* ----------------------------------------------------------------- */
    /* Compile the full bh_build program (base + original radix kernels) */
    /* ----------------------------------------------------------------- */
    {
        char *build_src = read_cl_source("bh_build.cl.h");
        TEST_ASSERT(build_src != NULL, "Failed to read bh_build.cl.h");

        size_t preamble_len = strlen(PREAMBLE);
        size_t build_len = strlen(build_src);
        char *combined = (char *)malloc(preamble_len + build_len + 1);
        TEST_ASSERT(combined != NULL, "malloc failed for combined source");

        size_t pos = 0;
        memcpy(combined + pos, PREAMBLE, preamble_len);
        pos += preamble_len;
        pos += skip_include_concat(combined + pos, preamble_len + build_len + 1 - pos, build_src);
        combined[pos] = '\0';
        free(build_src);

        const char *kernels[] = {"kernel_morton",   "kernel_radix_hist",  "kernel_radix_scatter",
                                 "kernel_boundary", "kernel_fill_leaves", "kernel_build_internal"};
        const unsigned n_kernels = sizeof(kernels) / sizeof(kernels[0]);

        status = cvl_cl_compute_init(&comp, &ctx, &queue, &device, CVL_CL_PRECISION_FP64, combined, kernels, n_kernels);
        free(combined);
        CVL_CL_CHECK(status, cleanup);
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
