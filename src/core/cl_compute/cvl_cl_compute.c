#include "cvl_cl_compute.h"
#include "../opencl/cvl_cl_program.h" /* cvl_cl_program_create (not pulled in by the header) */
#include "cvl_cl_kernel_sources.h"    /* generated: CVL_CL_PACK_SOURCES[] */

#include <assert.h>
#include <stdio.h>  /* stderr */
#include <string.h> /* memset, strcmp */

/* ------------------------------------------------------------------ */
/* Kernel → pack mapping                                              */
/* ------------------------------------------------------------------ */

/* Kernels provided by each pack (see cvl_cl_pack_t). */
static const char *const BH_BUILD_KERNELS[] = {
    "kernel_morton",   "kernel_radix_hist",  "kernel_radix_scatter",
    "kernel_boundary", "kernel_fill_leaves", "kernel_build_internal",
};

static const char *const BH_EVAL_KERNELS[] = {
    "bh_flat_eval",
};

static const char *const BH_COEFFS_KERNELS[] = {
    "kernel_p2m_leaves",
    "kernel_build_internal_m2m",
};

static const char *const FMM_EVAL_KERNELS[] = {
    "fmm_l2p_eval",
};

static const char *const DIRECT_SUM_KERNELS[] = {
    "direct_sum",
};

typedef struct
{
    cvl_cl_pack_t pack;
    const char *const *names;
    unsigned count;
} pack_kernels_t;

static const pack_kernels_t PACK_KERNELS[CVL_CL_PACK_COUNT] = {
    {CVL_CL_PACK_BH_BUILD, BH_BUILD_KERNELS, (unsigned)(sizeof(BH_BUILD_KERNELS) / sizeof(BH_BUILD_KERNELS[0]))},
    {CVL_CL_PACK_BH_EVAL, BH_EVAL_KERNELS, (unsigned)(sizeof(BH_EVAL_KERNELS) / sizeof(BH_EVAL_KERNELS[0]))},
    {CVL_CL_PACK_BH_COEFFS, BH_COEFFS_KERNELS, (unsigned)(sizeof(BH_COEFFS_KERNELS) / sizeof(BH_COEFFS_KERNELS[0]))},
    {CVL_CL_PACK_FMM_EVAL, FMM_EVAL_KERNELS, (unsigned)(sizeof(FMM_EVAL_KERNELS) / sizeof(FMM_EVAL_KERNELS[0]))},
    {CVL_CL_PACK_DIRECT_SUM, DIRECT_SUM_KERNELS,
     (unsigned)(sizeof(DIRECT_SUM_KERNELS) / sizeof(DIRECT_SUM_KERNELS[0]))},
};

/** @brief Human-readable pack name (for build-log diagnostics). */
static const char *pack_name(cvl_cl_pack_t pack)
{
    switch (pack)
    {
    case CVL_CL_PACK_BH_BUILD:
        return "BH_BUILD";
    case CVL_CL_PACK_BH_EVAL:
        return "BH_EVAL";
    case CVL_CL_PACK_BH_COEFFS:
        return "BH_COEFFS";
    case CVL_CL_PACK_FMM_EVAL:
        return "FMM_EVAL";
    case CVL_CL_PACK_DIRECT_SUM:
        return "DIRECT_SUM";
    default:
        return "UNKNOWN";
    }
}

/** @brief Resolve a kernel function name to its pack, or CVL_CL_PACK_COUNT if unknown. */
static cvl_cl_pack_t find_pack_for_kernel(const char *name)
{
    for (unsigned p = 0; p < CVL_CL_PACK_COUNT; ++p)
    {
        for (unsigned k = 0; k < PACK_KERNELS[p].count; ++k)
        {
            if (strcmp(PACK_KERNELS[p].names[k], name) == 0)
                return PACK_KERNELS[p].pack;
        }
    }
    return CVL_CL_PACK_COUNT;
}

/* ------------------------------------------------------------------ */
/* Internal helpers                                                   */
/* ------------------------------------------------------------------ */

/** @brief Compile @p pack once; on build failure print the log to stderr. */
static cvl_cl_status_t compile_pack(cvl_cl_compute_t *comp, cl_context ctx, const cvl_cl_device_t *device,
                                    cvl_cl_precision_t precision, cvl_cl_pack_t pack, char *log, size_t log_capacity)
{
    if (comp->programs[pack] != NULL)
        return CVL_CL_SUCCESS;

    cvl_cl_status_t st = cvl_cl_program_create(
        ctx, device->id, &(cvl_cl_program_desc_t){.source_string = CVL_CL_PACK_SOURCES[pack], .precision = precision},
        log, log_capacity, &comp->programs[pack]);
    if (st == CVL_CL_ERR_PROGRAM_BUILD)
        fprintf(stderr, "cvl_cl_compute_init: pack %s failed to build:\n%s\n", pack_name(pack), log);
    return st;
}

/** @brief Resolve a kernel function name to its canonical slot within a pack. */
static unsigned kernel_slot(cvl_cl_pack_t pack, const char *name)
{
    for (unsigned k = 0; k < PACK_KERNELS[pack].count; ++k)
    {
        if (strcmp(PACK_KERNELS[pack].names[k], name) == 0)
            return k;
    }
    return CVL_CL_MAX_KERNELS_PER_PACK; /* unreachable - find_pack_for_kernel already matched */
}

/** @brief Extract one kernel from its pack program and store it in the registry. */
static void register_kernel(cvl_cl_compute_t *comp, cvl_cl_pack_t pack, unsigned slot, const char *name)
{
    cl_kernel kernel = NULL;
    const cvl_cl_status_t st = cvl_cl_kernel_create(comp->programs[pack], name, &kernel);
    assert(st == CVL_CL_SUCCESS); /* the kernel IS in the pack source */
    (void)st;

    comp->kernels[pack][slot] = kernel;
}

/* ------------------------------------------------------------------ */
/* cvl_cl_compute_init                                                */
/* ------------------------------------------------------------------ */

cvl_cl_status_t cvl_cl_compute_init(cvl_cl_compute_t *comp, cl_context ctx, cl_command_queue queue,
                                    const cvl_cl_device_t *device, cvl_cl_precision_t precision,
                                    const char *kernel_names[], unsigned n_kernels)
{
    assert(comp && device);
    assert(kernel_names != NULL || n_kernels == 0);

    memset(comp, 0, sizeof(*comp));
    comp->ctx = ctx;
    comp->queue = queue;
    comp->device = device;
    comp->precision = precision;

    char log[2048];

    if (kernel_names == NULL)
    {
        /* Compile every pack and register every kernel it provides. */
        for (unsigned p = 0; p < CVL_CL_PACK_COUNT; ++p)
        {
            cvl_cl_status_t st = compile_pack(comp, ctx, device, precision, (cvl_cl_pack_t)p, log, sizeof(log));
            if (st != CVL_CL_SUCCESS)
            {
                cvl_cl_compute_destroy(comp);
                return st;
            }
        }

        for (unsigned p = 0; p < CVL_CL_PACK_COUNT; ++p)
            for (unsigned k = 0; k < PACK_KERNELS[p].count; ++k)
                register_kernel(comp, PACK_KERNELS[p].pack, k, PACK_KERNELS[p].names[k]);

        return CVL_CL_SUCCESS;
    }

    for (unsigned i = 0; i < n_kernels; ++i)
    {
        const char *name = kernel_names[i];
        const cvl_cl_pack_t pack = find_pack_for_kernel(name);
        if (pack == CVL_CL_PACK_COUNT)
        {
            cvl_cl_compute_destroy(comp);
            return CVL_CL_ERR_NOT_FOUND;
        }

        cvl_cl_status_t st = compile_pack(comp, ctx, device, precision, pack, log, sizeof(log));
        if (st != CVL_CL_SUCCESS)
        {
            cvl_cl_compute_destroy(comp);
            return st;
        }

        register_kernel(comp, pack, kernel_slot(pack, name), name);
    }

    return CVL_CL_SUCCESS;
}

/* ------------------------------------------------------------------ */
/* cvl_cl_compute_kernel                                              */
/* ------------------------------------------------------------------ */

cl_kernel cvl_cl_compute_kernel(const cvl_cl_compute_t *comp, cvl_cl_pack_t pack, unsigned kernel_index)
{
    assert(comp);
    assert(pack < CVL_CL_PACK_COUNT);
    assert(kernel_index < CVL_CL_MAX_KERNELS_PER_PACK);

    return comp->kernels[pack][kernel_index];
}

/* ------------------------------------------------------------------ */
/* cvl_cl_compute_destroy                                             */
/* ------------------------------------------------------------------ */

void cvl_cl_compute_destroy(cvl_cl_compute_t *comp)
{
    if (comp == NULL)
        return;

    /* Release kernels (reverse slot order), then programs. */
    for (unsigned p = CVL_CL_PACK_COUNT; p > 0; --p)
    {
        for (unsigned k = CVL_CL_MAX_KERNELS_PER_PACK; k > 0; --k)
        {
            if (comp->kernels[p - 1][k - 1] != NULL)
                clReleaseKernel(comp->kernels[p - 1][k - 1]);
        }
        if (comp->programs[p - 1] != NULL)
            clReleaseProgram(comp->programs[p - 1]);
    }

    memset(comp, 0, sizeof(*comp));
}
