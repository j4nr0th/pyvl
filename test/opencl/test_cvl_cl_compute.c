/*
 * test_cvl_cl_compute.c — Tests for the compute backend bundle
 * (cvl_cl_compute.h): pack compilation + kernel registry.
 *
 * Pipeline tested:
 *   1. init with a subset of names from different packs — the requested
 *      slots resolve to non-NULL cl_kernels, unrequested packs stay NULL.
 *   2. init with an unknown kernel name returns CVL_CL_ERR_NOT_FOUND.
 *   3. init with kernel_names == NULL compiles every pack and fills every
 *      slot (6 + 1 + 1 + 1 = 9 kernels); each resolves.
 *   4. cvl_cl_compute_destroy releases everything — slot lookups
 *      afterwards return NULL.
 *
 * The test skips gracefully (returns 0) when no OpenCL device is available.
 */

#ifdef CVL_OPENCL

#include "../test_common.h"
#include "cvl_cl_compute.h"
#include "cvl_cl_test_common.h"

/* Expected kernel count per pack (BH_BUILD 6 + BH_EVAL 1 + FMM_EVAL 1 +
 * DIRECT_SUM 1 = 9). */
static const unsigned PACK_KERNEL_COUNTS[CVL_CL_PACK_COUNT] = {6, 1, 1, 1};
#define N_ALL_KERNELS 9u

int main(void)
{
    cvl_cl_status_t status = CVL_CL_SUCCESS;
    cvl_cl_device_t device = {0};
    cl_context ctx = NULL;
    cl_command_queue queue = NULL;
    cvl_cl_compute_t comp = {0};
    int ret = 1;

    /* ----------------------------------------------------------------- */
    /* 1. Device discovery (GPU preferred, CPU fallback)                 */
    /* ----------------------------------------------------------------- */
    status = cvl_cl_device_first_gpu(&device);
    if (status != CVL_CL_SUCCESS)
        status = cvl_cl_device_first_cpu(&device);
    if (status != CVL_CL_SUCCESS)
    {
        fprintf(stderr, "No OpenCL device found -- skipping compute test.\n");
        return 0;
    }

    CVL_CL_CHECK(cvl_cl_ctx_create(&device, &ctx), cleanup);
    CVL_CL_CHECK(cvl_cl_queue_create(ctx, device.id, NULL, &queue), cleanup);

    /* ----------------------------------------------------------------- */
    /* 2. Subset across packs: one kernel per pack                       */
    /* ----------------------------------------------------------------- */
    {
        const char *subset[] = {"kernel_morton", "bh_flat_eval", "direct_sum"};
        CVL_CL_CHECK(cvl_cl_compute_init(&comp, ctx, queue, &device, CVL_CL_PRECISION_FP64, subset, 3), cleanup);

        TEST_ASSERT(cvl_cl_compute_kernel(&comp, CVL_CL_PACK_BH_BUILD, CVL_CL_BH_BUILD_MORTON) != NULL,
                    "kernel_morton did not resolve");
        TEST_ASSERT(cvl_cl_compute_kernel(&comp, CVL_CL_PACK_BH_EVAL, CVL_CL_BH_EVAL_FLAT_EVAL) != NULL,
                    "bh_flat_eval did not resolve");
        TEST_ASSERT(cvl_cl_compute_kernel(&comp, CVL_CL_PACK_DIRECT_SUM, CVL_CL_DIRECT_SUM_KERNEL) != NULL,
                    "direct_sum did not resolve");

        /* Kernels of unrequested packs must NOT resolve. */
        TEST_ASSERT(cvl_cl_compute_kernel(&comp, CVL_CL_PACK_FMM_EVAL, CVL_CL_FMM_EVAL_L2P) == NULL,
                    "unrequested pack FMM_EVAL resolved");
        printf("compute subset init OK: 3 kernels across 3 packs\n");
    }

    /* Re-init for the next section (comp is zeroed by destroy). */
    cvl_cl_compute_destroy(&comp);

    /* ----------------------------------------------------------------- */
    /* 3. Unknown kernel name → CVL_CL_ERR_NOT_FOUND                     */
    /* ----------------------------------------------------------------- */
    {
        const char *bad[] = {"no_such_kernel"};
        cvl_cl_status_t st = cvl_cl_compute_init(&comp, ctx, queue, &device, CVL_CL_PRECISION_FP64, bad, 1);
        TEST_ASSERT(st == CVL_CL_ERR_NOT_FOUND, "unknown kernel name returned %s, expected CVL_CL_ERR_NOT_FOUND",
                    cvl_cl_status_str(st));
        /* The failed init must leave the handle in a destroyed (zeroed) state. */
        TEST_ASSERT(cvl_cl_compute_kernel(&comp, CVL_CL_PACK_BH_BUILD, CVL_CL_BH_BUILD_MORTON) == NULL &&
                        cvl_cl_compute_kernel(&comp, CVL_CL_PACK_FMM_EVAL, CVL_CL_FMM_EVAL_L2P) == NULL,
                    "failed init left kernels registered");
        printf("compute unknown-name error OK\n");
    }

    /* ----------------------------------------------------------------- */
    /* 4. NULL kernel_names → every kernel of every pack                  */
    /* ----------------------------------------------------------------- */
    {
        CVL_CL_CHECK(cvl_cl_compute_init(&comp, ctx, queue, &device, CVL_CL_PRECISION_FP64, NULL, 0), cleanup);
        unsigned total = 0;
        for (unsigned p = 0; p < CVL_CL_PACK_COUNT; ++p)
        {
            for (unsigned k = 0; k < PACK_KERNEL_COUNTS[p]; ++k)
            {
                TEST_ASSERT(cvl_cl_compute_kernel(&comp, (cvl_cl_pack_t)p, k) != NULL,
                            "pack %u slot %u did not resolve", p, k);
                ++total;
            }
        }
        TEST_ASSERT(total == N_ALL_KERNELS, "NULL-names init registered %u kernels, expected %u", total, N_ALL_KERNELS);
        printf("compute NULL-names init OK: all %u kernels registered\n", N_ALL_KERNELS);
    }

    /* ----------------------------------------------------------------- */
    /* 5. After destroy, lookups return NULL                             */
    /* ----------------------------------------------------------------- */
    cvl_cl_compute_destroy(&comp);
    for (unsigned p = 0; p < CVL_CL_PACK_COUNT; ++p)
        for (unsigned k = 0; k < PACK_KERNEL_COUNTS[p]; ++k)
            TEST_ASSERT(cvl_cl_compute_kernel(&comp, (cvl_cl_pack_t)p, k) == NULL,
                        "kernel %u/%u resolved after destroy", p, k);
    printf("compute destroy OK: lookups return NULL\n");

    printf("All compute backend tests passed.\n");
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
    printf("OpenCL not available -- skipping compute test.\n");
    return 0;
}

#endif /* CVL_OPENCL */
