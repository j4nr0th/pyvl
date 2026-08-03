#pragma once
/*
 * High-level compute backend for OpenCL-accelerated tree methods.
 *
 * Bundles the low-level opencl/ wrappers into a single handle:
 *   - borrowed context, queue, and device (raw handles)
 *   - owned compiled programs (one per kernel pack - see cvl_cl_pack_t)
 *   - a registry of named cl_kernel handles
 *   - the cached precision mode
 *
 * Kernel sources are embedded into the library at build time (see
 * gen_kernel_sources.py); cvl_cl_compute_init only needs the kernel
 * NAMES it should extract.  Passing kernel_names == NULL extracts
 * every kernel from every pack.
 *
 * Typical usage:
 * @code
 *   cvl_cl_compute_t comp;
 *   cvl_cl_compute_init(&comp, ctx, queue, &device, CVL_CL_PRECISION_FP64,
 *       (const char*[]){"bh_flat_eval"}, 1);
 *
 *   cl_kernel k = cvl_cl_compute_kernel(&comp, CVL_CL_PACK_BH_EVAL, CVL_CL_BH_EVAL_FLAT_EVAL);
 *   cvl_cl_kernel_set_args(k, ...);
 *   ...
 *   cvl_cl_compute_destroy(&comp);
 * @endcode
 */

#include "../opencl/cvl_cl_common.h"
#include "../opencl/cvl_cl_ctx.h"
#include "../opencl/cvl_cl_device.h"
#include "../opencl/cvl_cl_kernel.h"
#include "cvl_cl_staging_buffer.h"

#include <CL/cl.h>

/* ------------------------------------------------------------------ */
/* Kernel packs                                                       */
/* ------------------------------------------------------------------ */

/**
 * @brief A set of kernels compiled together from one source unit.
 *
 * Each pack corresponds to a .cl.h file (or include chain) whose
 * source is embedded at build time.  Kernel names are resolved to
 * their pack by cvl_cl_compute_init.
 */
typedef enum
{
    CVL_CL_PACK_BH_BUILD = 0, /**< bh_build.cl.h - tree build kernels. */
    CVL_CL_PACK_BH_EVAL,      /**< bh_flat_eval.cl.h - BH evaluation. */
    CVL_CL_PACK_BH_COEFFS,    /**< bh_p2m_m2m.cl.h - multipole coefficients (P2M/M2M). */
    CVL_CL_PACK_FMM_EVAL,     /**< fmm_l2p.cl.h + shared headers - FMM L2P evaluation. */
    CVL_CL_PACK_DIRECT_SUM,   /**< direct_sum.cl.h - direct N-body sum. */
    CVL_CL_PACK_COUNT,        /**< Number of packs (also bounds programs[]). */
} cvl_cl_pack_t;

/** @brief Maximum number of kernels a single pack can hold (BH_BUILD has 6). */
enum
{
    CVL_CL_MAX_KERNELS_PER_PACK = 8
};

/* Canonical kernel slots within each pack.  These index both
 * cvl_cl_compute_t::kernels and cvl_cl_compute_kernel(). */
typedef enum
{
    CVL_CL_BH_BUILD_MORTON = 0,    /**< kernel_morton. */
    CVL_CL_BH_BUILD_RADIX_HIST,    /**< kernel_radix_hist. */
    CVL_CL_BH_BUILD_RADIX_SCATTER, /**< kernel_radix_scatter. */
    CVL_CL_BH_BUILD_BOUNDARY,      /**< kernel_boundary. */
    CVL_CL_BH_BUILD_FILL_LEAVES,   /**< kernel_fill_leaves. */
    CVL_CL_BH_BUILD_INTERNAL,      /**< kernel_build_internal. */
    CVL_CL_BH_BUILD_KERNEL_COUNT,
} cvl_cl_bh_build_kernel_t;

typedef enum
{
    CVL_CL_BH_EVAL_FLAT_EVAL = 0, /**< bh_flat_eval. */
    CVL_CL_BH_EVAL_KERNEL_COUNT,
} cvl_cl_bh_eval_kernel_t;

typedef enum
{
    CVL_CL_BH_COEFFS_P2M = 0,  /**< kernel_p2m_leaves. */
    CVL_CL_BH_COEFFS_INTERNAL, /**< kernel_build_internal_m2m. */
    CVL_CL_BH_COEFFS_KERNEL_COUNT,
} cvl_cl_bh_coeffs_kernel_t;

typedef enum
{
    CVL_CL_FMM_EVAL_L2P = 0, /**< fmm_l2p_eval. */
    CVL_CL_FMM_EVAL_KERNEL_COUNT,
} cvl_cl_fmm_eval_kernel_t;

typedef enum
{
    CVL_CL_DIRECT_SUM_KERNEL = 0, /**< direct_sum. */
    CVL_CL_DIRECT_SUM_KERNEL_COUNT,
} cvl_cl_direct_sum_kernel_t;

/* ------------------------------------------------------------------ */
/* Compute backend handle                                             */
/* ------------------------------------------------------------------ */

typedef struct
{
    /* Borrowed references (caller keeps these alive). */
    cl_context ctx;
    cl_command_queue queue;
    const cvl_cl_device_t *device;

    /* Owned: one compiled program per pack (NULL if that pack was not compiled). */
    cl_program programs[CVL_CL_PACK_COUNT];

    /* Owned kernels, indexed by pack then canonical slot (see the
     * cvl_cl_bh_build_kernel_t / cvl_cl_bh_eval_kernel_t /
     * cvl_cl_fmm_eval_kernel_t / cvl_cl_direct_sum_kernel_t enums).
     * A slot is NULL until its pack has been compiled by init. */
    cl_kernel kernels[CVL_CL_PACK_COUNT][CVL_CL_MAX_KERNELS_PER_PACK];

    /* Cached mode. */
    cvl_cl_precision_t precision;
} cvl_cl_compute_t;

/**
 * @brief Initialise the compute backend.
 *
 * Compiles the packs needed for the requested kernel names (each pack
 * once) and extracts the kernels into the registry.  If @p kernel_names
 * is NULL, every kernel of every pack is registered.
 *
 * @param comp          Uninitialised compute handle.
 * @param ctx           Context (borrowed - must outlive comp).
 * @param queue         Queue (borrowed - must outlive comp).
 * @param device        Device handle (borrowed - must outlive comp).
 * @param precision     FP32 or FP64.
 * @param kernel_names  Array of kernel function names to extract, or NULL for all.
 * @param n_kernels     Number of entries in @p kernel_names.
 * @return CVL_CL_SUCCESS, CVL_CL_ERR_NOT_FOUND for an unknown kernel name,
 *         CVL_CL_ERR_PROGRAM_BUILD on compile failure, or other error.
 */
cvl_cl_status_t cvl_cl_compute_init(cvl_cl_compute_t *comp, cl_context ctx, cl_command_queue queue,
                                    const cvl_cl_device_t *device, cvl_cl_precision_t precision,
                                    const char *kernel_names[], unsigned n_kernels);

/**
 * @brief Return the kernel at a pack's canonical slot.
 *
 * Purely an index into the registry - it cannot fail.  Returns NULL
 * only if that pack was not compiled (its kernels were not requested
 * at init).
 *
 * @param comp          Compute backend.
 * @param pack          Pack the kernel belongs to.
 * @param kernel_index  Canonical slot within the pack (one of the
 *                      cvl_cl_bh_build_kernel_t / cvl_cl_bh_eval_kernel_t /
 *                      cvl_cl_fmm_eval_kernel_t / cvl_cl_direct_sum_kernel_t
 *                      values).
 * @return The cl_kernel handle, or NULL if the pack was not compiled.
 */
cl_kernel cvl_cl_compute_kernel(const cvl_cl_compute_t *comp, cvl_cl_pack_t pack, unsigned kernel_index);

/**
 * @brief Destroy the compute backend, releasing kernels and programs.
 *
 * Safe to call on a zero-initialised or partially-initialised handle.
 * Does NOT release the borrowed ctx, queue, or device.
 *
 * @param comp Compute backend to destroy (may be NULL).
 */
void cvl_cl_compute_destroy(cvl_cl_compute_t *comp);
