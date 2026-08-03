#pragma once
/*
 * Persistent OpenCL tree handle with asynchronous build/eval jobs.
 *
 * This module wraps the flat-tree builder + the P2M/M2M coefficient
 * kernels + the BH tree-code evaluation kernel into a single handle
 * that owns all device buffers (grow-only, reused across rebuilds) and
 * exposes a begin/finish job model:
 *
 *   - cvl_cl_tree_build_begin   - enqueue the host->device uploads and
 *                                 P2M/M2M kernels; returns a job.
 *   - cvl_cl_tree_build_finish  - block until the build job completes.
 *   - cvl_cl_tree_eval_begin    - enqueue an eval (tree-code or direct).
 *   - cvl_cl_tree_eval_finish   - block, convert, read back the result.
 *
 * The host-side steps of a build (Morton sort, flat-tree construction)
 * run inside build_finish; the begin call enqueues the device work that
 * does not depend on those steps (source upload).  Eval jobs may be
 * enqueued once the tree is built; the in-order queue serializes them.
 *
 * Buffers are grow-only: repeated rebuilds with the same or larger
 * sizes reuse the existing device allocations (the key design goal -
 * maximal resource reuse in the background).
 *
 * The tree-code evaluation uses the flat-tree multipole coefficients
 * computed by kernel_p2m_leaves + kernel_build_internal_m2m, which are
 * exact (verified against host multipole_create to ~1e-12).  The GPU
 * tree builder (cvl_cl_gpu_tree_build_t) is NOT used here: its internal
 * node construction has a latent octant bug producing inconsistent
 * trees (see intel-neo-cpu-bug.md / opencl-summary.md notes).
 */

#include "../opencl/cvl_cl_buffer.h"
#include "../opencl/cvl_cl_chain.h"
#include "../opencl/cvl_cl_common.h"
#include "../opencl/cvl_cl_ctx.h"
#include "../opencl/cvl_cl_kernel.h"
#include "cvl_cl_compute.h"
#include "cvl_cl_flat_tree.h"
#include "cvl_cl_staging_buffer.h"

/* ------------------------------------------------------------------ */
/*  Eval mode                                                          */
/* ------------------------------------------------------------------ */

typedef enum
{
    CVL_CL_TREE_EVAL_TREE_CODE = 0, /**< bh_flat_eval: MAC-driven multipole tree-code. */
    CVL_CL_TREE_EVAL_DIRECT,        /**< direct_sum: exact O(N) per target. */
} cvl_cl_tree_eval_mode_t;

/* ------------------------------------------------------------------ */
/*  Tree handle                                                       */
/* ------------------------------------------------------------------ */

/**
 * @brief Persistent OpenCL tree.
 *
 * Owns device buffers for the flat node array, the multipole
 * coefficients, the particle order, and staging buffers for source
 * positions/values and eval targets/results.  All buffers are grow-only
 * and reused across rebuilds.
 */
typedef struct
{
    /* Borrowed compute backend (kernel registry + queue + ctx). */
    cvl_cl_compute_t *compute;
    cvl_cl_precision_t precision;

    /* Tree settings. */
    cvl_cl_flat_tree_settings_t settings;
    unsigned work_order; /**< M2M series order (0 = use settings.order). */

    /* Host-side metadata (updated by each build_finish). */
    unsigned n_sources;
    unsigned n_nodes;
    unsigned n_internal;
    unsigned n_multipole_leaves;
    unsigned n_particle_leaves;
    unsigned n_leaves;
    unsigned max_depth_used;

    /* Device buffers (grow-only). */
    cvl_cl_buffer_t buf_nodes;          /**< [n_nodes] flat nodes (64 B each). */
    cvl_cl_buffer_t buf_particle_order; /**< [n_sources] leaf particle indices. */
    cvl_cl_buffer_t buf_depth_offsets;  /**< [max_depth+2] depth layer starts. */
    cvl_cl_buffer_t buf_coeffs;         /**< [n_nodes * 3 * n_coeffs] multipoles. */
    cvl_cl_buffer_t buf_p2m_scratch;    /**< [n_leaves * 2 * scratch_size] P2M scratch. */
    cvl_cl_buffer_t buf_m2m_scratch;    /**< [n_internal * per-wg] M2M scratch. */
    cvl_cl_buffer_t buf_parent_starts;  /**< [n_internal+1] per-parent child starts. */

    /* Staging buffers. */
    cvl_cl_staging_buffer_t buf_src_pos; /**< [n_sources] source positions. */
    cvl_cl_staging_buffer_t buf_src_val; /**< [n_sources] source strengths. */
    cvl_cl_staging_buffer_t buf_targets; /**< [n_targets] eval points. */
    cvl_cl_staging_buffer_t buf_results; /**< [n_targets] eval output. */

    /* Build state. */
    bool built;                 /**< True after the first successful build_finish. */
    bool build_in_flight;       /**< True between build_begin and build_finish. */
    cvl_cl_chain_t build_chain; /**< Chain for the in-flight build. */
} cvl_cl_tree_t;

/* ------------------------------------------------------------------ */
/*  Job handles                                                       */
/* ------------------------------------------------------------------ */

/**
 * @brief A pending asynchronous build job.
 *
 * Owns a chain on which the build's device work is enqueued.  The host
 * completes the job in cvl_cl_tree_build_finish (which runs the host
 * Morton/flat-tree steps and enqueues the remaining device work).
 */
typedef struct
{
    cvl_cl_tree_t *tree;  /**< Owning tree (borrowed). */
    cvl_cl_chain_t chain; /**< Dependency chain for the build. */
    bool finished;        /**< True once the job has completed. */
    bool cancelled;       /**< True if the tree was destroyed first. */
} cvl_cl_tree_build_job_t;

/**
 * @brief A pending asynchronous eval job.
 *
 * Owns the target/result staging buffers for this eval so concurrent
 * evals do not clash.
 */
typedef struct
{
    cvl_cl_tree_t *tree;             /**< Owning tree (borrowed). */
    cvl_cl_chain_t chain;            /**< Dependency chain for the eval. */
    unsigned n_targets;              /**< Number of targets. */
    cvl_cl_staging_buffer_t targets; /**< Owned staging (grown on demand). */
    cvl_cl_staging_buffer_t results; /**< Owned staging (grown on demand). */
    cvl_cl_tree_eval_mode_t mode;    /**< Eval mode. */
    double theta;                    /**< MAC opening angle (tree-code only). */
    bool finished;                   /**< True once the job has completed. */
    bool cancelled;                  /**< True if the tree was destroyed first. */
} cvl_cl_tree_eval_job_t;

/* ------------------------------------------------------------------ */
/*  Tree lifecycle                                                     */
/* ------------------------------------------------------------------ */

/**
 * @brief Initialise an empty tree handle.
 *
 * @param tree      Uninitialised tree handle.
 * @param compute   Compute backend (borrowed; must outlive the tree).
 * @param precision FP32 or FP64 (should match the compute backend).
 * @param settings  Flat-tree settings (max_depth, critical_particle_count, order).
 * @param work_order M2M internal series order (0 = use settings.order).
 * @return CVL_CL_SUCCESS.
 */
cvl_cl_status_t cvl_cl_tree_init(cvl_cl_tree_t *tree, cvl_cl_compute_t *compute, cvl_cl_precision_t precision,
                                 const cvl_cl_flat_tree_settings_t *settings, unsigned work_order);

/**
 * @brief Release all device buffers owned by the tree.
 *
 * Safe to call with an in-flight build (it is cancelled first) or
 * already-released tree.  Any outstanding jobs are marked cancelled.
 *
 * @param tree Tree to destroy (may be NULL).
 */
void cvl_cl_tree_destroy(cvl_cl_tree_t *tree);

/* ------------------------------------------------------------------ */
/*  Build                                                              */
/* ------------------------------------------------------------------ */

/**
 * @brief Enqueue the upload half of a tree build and return a job.
 *
 * Uploads the source positions/values to the device.  The host-side
 * steps (Morton sort, flat-tree construction) run in
 * cvl_cl_tree_build_finish.  Multiple builds may not overlap; calling
 * while a build is in flight is an error.
 *
 * @param tree    Tree handle.
 * @param n_sources Number of sources.
 * @param sources_coords Source positions [n_sources].
 * @param sources_values Source strengths [n_sources].
 * @param job     Output build job.
 * @return CVL_CL_SUCCESS or CVL_CL_ERR_INVALID_PARAM (build in flight).
 */
cvl_cl_status_t cvl_cl_tree_build_begin(cvl_cl_tree_t *tree, unsigned n_sources,
                                        const real3_t sources_coords[restrict n_sources],
                                        const real3_t sources_values[restrict n_sources], cvl_cl_tree_build_job_t *job);

/**
 * @brief Complete a build job.
 *
 * Runs the host-side tree construction (Morton codes, sort, flat tree),
 * enqueues the node/coeff uploads and the P2M/M2M kernels, waits for
 * completion, and updates the tree metadata.
 *
 * @param job Build job from cvl_cl_tree_build_begin.
 * @param work Host work buffer for the host-side construction.
 * @param work_size Size of the work buffer.
 * @return CVL_CL_SUCCESS, CVL_CL_ERR_BUFFER_SIZE (work too small), or error.
 */
cvl_cl_status_t cvl_cl_tree_build_finish(cvl_cl_tree_build_job_t *job, void *work, size_t work_size);

/**
 * @brief Compute the host work-buffer size for a build.
 *
 * @param n_sources Number of sources.
 * @param max_depth Maximum tree depth.
 * @return Required bytes.
 */
size_t cvl_cl_tree_build_work_size(unsigned n_sources, unsigned max_depth);

/* ------------------------------------------------------------------ */
/*  Eval                                                               */
/* ------------------------------------------------------------------ */

/**
 * @brief Enqueue an eval job and return immediately.
 *
 * @param tree     Tree handle (must be built).
 * @param n_targets Number of targets.
 * @param targets  Target positions [n_targets].
 * @param mode     Eval mode (tree-code or direct).
 * @param theta    MAC opening angle (tree-code only; <= 0 = neighbour).
 * @param job      Output eval job.
 * @return CVL_CL_SUCCESS or error.
 */
cvl_cl_status_t cvl_cl_tree_eval_begin(cvl_cl_tree_t *tree, unsigned n_targets, const real3_t targets[restrict],
                                       cvl_cl_tree_eval_mode_t mode, double theta, cvl_cl_tree_eval_job_t *job);

/**
 * @brief Complete an eval job: wait, convert, and copy results out.
 *
 * @param job     Eval job from cvl_cl_tree_eval_begin.
 * @param out     Output array [n_targets] real3_t.
 * @param scratch_f32 FP32 conversion scratch (3 * n_targets floats) or NULL in FP64.
 * @return CVL_CL_SUCCESS or error.
 */
cvl_cl_status_t cvl_cl_tree_eval_finish(cvl_cl_tree_eval_job_t *job, real3_t out[restrict], float *scratch_f32);

/**
 * @brief Release an eval job without reading the result.
 *
 * @param job Eval job to cancel.
 */
void cvl_cl_tree_eval_cancel(cvl_cl_tree_eval_job_t *job);

/* ------------------------------------------------------------------ */
/*  Sync convenience (single-call build + eval)                        */
/* ------------------------------------------------------------------ */

/**
 * @brief One-shot sync build.
 *
 * Equivalent to build_begin + build_finish.
 */
cvl_cl_status_t cvl_cl_tree_build(cvl_cl_tree_t *tree, unsigned n_sources,
                                  const real3_t sources_coords[restrict n_sources],
                                  const real3_t sources_values[restrict n_sources], void *work, size_t work_size);

/**
 * @brief One-shot sync eval.
 *
 * Equivalent to eval_begin + eval_finish.
 */
cvl_cl_status_t cvl_cl_tree_eval(cvl_cl_tree_t *tree, unsigned n_targets, const real3_t targets[restrict],
                                 cvl_cl_tree_eval_mode_t mode, double theta, real3_t out[restrict], float *scratch_f32);
