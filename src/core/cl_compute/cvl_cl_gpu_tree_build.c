/*
 * GPU-accelerated uniform octree builder - host-side orchestration.
 *
 * Implements the pipeline declared in cvl_cl_gpu_tree_build.h.
 *
 * The pipeline runs fully on the GPU once the bounding box is known:
 *   1. Morton-code generation
 *   2. 64-bit LSD radix sort (8 passes × histogram + scatter)
 *   3. Boundary-depth detection
 *   4. Host-side histogram → tree sizing
 *   5. Leaf compaction + construction (two kernels replace prefix-scan approach)
 *   6. Bottom-up internal node construction
 *
 * All device work is enqueued through a cvl_cl_chain_t dependency
 * stream on the compute backend's queue; cvl_cl_chain_finish() is
 * called at every point where the host must read results back.
 */

#include "cvl_cl_gpu_tree_build.h"
#include "../cvl_radix_sort.h"
#include "../opencl/cvl_cl_chain.h"
#include "../opencl/cvl_cl_helpers.h"

#include <assert.h>
#include <math.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/*  Helper: LSD radix sort on GPU (64-bit keys, 8-bit digits)          */
/* ------------------------------------------------------------------ */

/**
 * @brief Perform a 64-bit LSD radix sort of Morton codes + companion indices
 *        using the original per-work-group __local kernels.
 *
 * On entry @p buf_morton holds the unsorted codes and @p buf_indices
 * holds the initial index permutation (identity).  On exit both are
 * sorted (Morton codes ascending, indices permuted accordingly).
 *
 * @p buf_morton_tmp and @p buf_indices_tmp are used as ping-pong
 * scratch buffers and must be pre-sized to at least @p n elements.
 *
 * The radix histogram buffer is grown as needed.
 *
 * All kernel launches, the histogram read-back, and the prefix upload
 * go through @p chain; cvl_cl_chain_finish() is called once per pass
 * where the host must compute the next prefix.
 *
 * @param builder  Builder (provides buffers and kernel handles).
 * @param chain    Dependency chain on the compute backend's queue.
 * @param work     Host scratch (histogram + prefix, [n_wgs * 256] unsigned).
 * @param n        Number of elements to sort.
 * @return CVL_CL_SUCCESS or error.
 */
static cvl_cl_status_t gpu_radix_sort_orig(cvl_cl_gpu_tree_build_t *builder, cvl_cl_chain_t *chain, void *work,
                                           unsigned n)
{
    cvl_cl_status_t st;
    const unsigned n_wgs = (n + CVL_CL_GPU_BUILD_WG - 1) / CVL_CL_GPU_BUILD_WG;
    const size_t hist_bytes = (size_t)n_wgs * 256u * sizeof(unsigned);

    /* ---- Ensure radix histogram buffer is large enough ---- */
    st = cl_ensure_buffer_chained(&builder->buf_radix_hist, builder->compute->ctx, chain, hist_bytes);
    if (st != CVL_CL_SUCCESS)
        return st;

    /* ---- Host-side histogram + prefix buffer (from work buffer) ---- */
    unsigned *host_h = (unsigned *)work;

    /* ---- 8 passes, one per 8-bit digit of the 64-bit key ---- */
    int pass_odd = 0;

    for (unsigned pass = 0; pass < 8; ++pass)
    {
        const unsigned shift = pass * 8u;

        cl_mem keys_src = pass_odd ? builder->buf_morton_tmp.mem : builder->buf_morton.mem;
        cl_mem idx_src = pass_odd ? builder->buf_indices_tmp.mem : builder->buf_indices.mem;
        cl_mem keys_dst = pass_odd ? builder->buf_morton.mem : builder->buf_morton_tmp.mem;
        cl_mem idx_dst = pass_odd ? builder->buf_indices.mem : builder->buf_indices_tmp.mem;

        /* ----- histogram pass (waits on the previous pass's scatter) ----- */
        cl_kernel kh = cvl_cl_compute_kernel(builder->compute, CVL_CL_PACK_BH_BUILD, CVL_CL_BH_BUILD_RADIX_HIST);
        if (!kh)
            return CVL_CL_ERR_INTERNAL;

        {
            const size_t global = ((n + CVL_CL_GPU_BUILD_WG - 1) / CVL_CL_GPU_BUILD_WG) * CVL_CL_GPU_BUILD_WG;
            const size_t local = CVL_CL_GPU_BUILD_WG;
            st = cvl_cl_chain_ndrange(chain, kh, 1, &global, &local,
                                      (cvl_cl_karg_t[]){
                                          {.type = CVL_CL_KARG_BUFFER, .index = 0, .mem = keys_src},
                                          {.type = CVL_CL_KARG_SCALAR_UINT, .index = 1, .scalar_uint = n},
                                          {.type = CVL_CL_KARG_SCALAR_UINT, .index = 2, .scalar_uint = shift},
                                          {.type = CVL_CL_KARG_BUFFER, .index = 3, .mem = builder->buf_radix_hist.mem},
                                          {},
                                      },
                                      0, NULL, NULL);
            if (st != CVL_CL_SUCCESS)
                return st;
        }

        /* Read histogram to host, then wait - the host computes the
         * per-WG prefix next. */
        if ((st = cvl_cl_chain_read_buffer(chain, &builder->buf_radix_hist, 0, hist_bytes, host_h, 0, NULL, NULL)) !=
                CVL_CL_SUCCESS ||
            (st = cvl_cl_chain_finish(chain)) != CVL_CL_SUCCESS)
            return st;

        /* Compute the exclusive per-WG prefix per digit, PLUS the digit base. */
        unsigned base = 0;
        for (unsigned d = 0; d < 256; ++d)
        {
            unsigned acc = base;
            for (unsigned w = 0; w < n_wgs; ++w)
            {
                const unsigned cnt = host_h[w * 256u + d];
                host_h[w * 256u + d] = acc;
                acc += cnt;
            }
            base = acc;
        }

        /* Upload prefix back to the same device buffer (chained - the
         * scatter below waits on this write). */
        st = cvl_cl_chain_write_buffer(chain, &builder->buf_radix_hist, 0, hist_bytes, host_h, 0, NULL, NULL);
        if (st != CVL_CL_SUCCESS)
            return st;

        /* ----- scatter pass ----- */
        cl_kernel ks = cvl_cl_compute_kernel(builder->compute, CVL_CL_PACK_BH_BUILD, CVL_CL_BH_BUILD_RADIX_SCATTER);
        if (!ks)
            return CVL_CL_ERR_INTERNAL;

        {
            const size_t global = ((n + CVL_CL_GPU_BUILD_WG - 1) / CVL_CL_GPU_BUILD_WG) * CVL_CL_GPU_BUILD_WG;
            const size_t local = CVL_CL_GPU_BUILD_WG;
            st = cvl_cl_chain_ndrange(chain, ks, 1, &global, &local,
                                      (cvl_cl_karg_t[]){
                                          {.type = CVL_CL_KARG_BUFFER, .index = 0, .mem = keys_src},
                                          {.type = CVL_CL_KARG_BUFFER, .index = 1, .mem = idx_src},
                                          {.type = CVL_CL_KARG_BUFFER, .index = 2, .mem = keys_dst},
                                          {.type = CVL_CL_KARG_BUFFER, .index = 3, .mem = idx_dst},
                                          {.type = CVL_CL_KARG_SCALAR_UINT, .index = 4, .scalar_uint = n},
                                          {.type = CVL_CL_KARG_SCALAR_UINT, .index = 5, .scalar_uint = shift},
                                          {.type = CVL_CL_KARG_BUFFER, .index = 6, .mem = builder->buf_radix_hist.mem},
                                          {},
                                      },
                                      0, NULL, NULL);
            if (st != CVL_CL_SUCCESS)
                return st;
        }

        pass_odd ^= 1;
    }

    /* After 8 passes pass_odd == 0, final result is in the even buffers. */

    st = cvl_cl_chain_finish(chain);

    return st;
}

/* ------------------------------------------------------------------ */
/*  Workaround radix sort (Intel NEO CPU backend)                      */
/* ------------------------------------------------------------------ */

/**
 * @brief Perform the Morton-code sort on the HOST.
 *
 * WHY (read carefully - this is why the device-kernel workaround exists):
 * The Intel NEO CPU OpenCL backend ("OpenCL 3.0 (Build 0)") miscompiles
 * kernels whose address computations depend on data loaded from global
 * memory.  The original per-work-group radix kernels use __local memory
 * indexed by a data-dependent digit and crash ~100% of the time.  A
 * kernel-only workaround that avoids __local (one work-item per digit,
 * serial scan) was also observed to crash the runtime's JIT-compiled
 * code with significant probability, and the failure rate depends on the
 * process memory layout (same kernel, same data - crashes in one binary,
 * passes in another).  See intel-neo-cpu-bug.md.
 *
 * Conclusion: no device kernel is reliably safe on this backend.  The
 * robust fallback is to sort on the host using a parallel LSD radix sort
 * (same algorithm as octree_build_morton_sorted).  The device is a CPU
 * anyway, so this is natural; it is deterministic and does not exercise
 * the broken JIT at all.
 *
 * @param builder  Builder (provides buffers).
 * @param chain    Dependency chain on the compute backend's queue.
 * @param work     Host scratch (pairs / pairs_alt / hist / staging).
 * @param n        Number of elements to sort.
 * @return CVL_CL_SUCCESS or error.
 */
static cvl_cl_status_t gpu_radix_sort_host(cvl_cl_gpu_tree_build_t *builder, cvl_cl_chain_t *chain, void *work,
                                           unsigned n)
{
    cvl_cl_status_t st;

    const size_t key_bytes = (size_t)n * sizeof(uint64_t);
    const size_t idx_bytes = (size_t)n * sizeof(unsigned);
    const size_t pair_size = CVL_RADIX_SORT_PAIR_SIZE;
    const size_t pair_bytes = (size_t)n * pair_size;

    /* Partition the work buffer. */
    uint8_t *pairs = (uint8_t *)work;
    uint8_t *pairs_alt = pairs + pair_bytes;
    unsigned *hist = (unsigned *)(pairs_alt + pair_bytes);
    uint8_t *staging = (uint8_t *)hist + CVL_RADIX_BINS * sizeof(unsigned);

    uint64_t *keys_staging = (uint64_t *)staging;
    unsigned *idx_staging = (unsigned *)(staging + key_bytes);

    /* Read the unsorted Morton codes + index permutation (chained - the
     * host sort below needs both, hence the finish). */
    if ((st = cvl_cl_chain_read_buffer(chain, &builder->buf_morton, 0, key_bytes, keys_staging, 0, NULL, NULL)) !=
            CVL_CL_SUCCESS ||
        (st = cvl_cl_chain_read_buffer(chain, &builder->buf_indices, 0, idx_bytes, idx_staging, 0, NULL, NULL)) !=
            CVL_CL_SUCCESS ||
        (st = cvl_cl_chain_finish(chain)) != CVL_CL_SUCCESS)
        return st;

    /* Interleave into (key, idx) pairs for the radix sort. */
    for (unsigned i = 0; i < n; ++i)
    {
        *(uint64_t *)(pairs + (size_t)i * pair_size) = keys_staging[i];
        *(unsigned *)(pairs + (size_t)i * pair_size + sizeof(uint64_t)) = idx_staging[i];
    }

    /* Parallel LSD radix sort on the interleaved pairs. */
    cvl_radix_sort_pairs(pairs, pairs_alt, (size_t)n, pair_size, hist, 1u);

    /* Extract sorted keys and indices from interleaved pairs back to staging. */
    for (unsigned i = 0; i < n; ++i)
    {
        keys_staging[i] = *(const uint64_t *)(pairs + (size_t)i * pair_size);
        idx_staging[i] = *(const unsigned *)(pairs + (size_t)i * pair_size + sizeof(uint64_t));
    }

    /* Write the sorted result back (even buffers, as the kernel path does). */
    if ((st = cvl_cl_chain_write_buffer(chain, &builder->buf_morton, 0, key_bytes, keys_staging, 0, NULL, NULL)) !=
            CVL_CL_SUCCESS ||
        (st = cvl_cl_chain_write_buffer(chain, &builder->buf_indices, 0, idx_bytes, idx_staging, 0, NULL, NULL)) !=
            CVL_CL_SUCCESS ||
        (st = cvl_cl_chain_finish(chain)) != CVL_CL_SUCCESS)
        return st;

    return CVL_CL_SUCCESS;
}

/**
 * @brief Dispatch to the original kernel sort or the host-side workaround.
 *
 * The effective mode is resolved at init / policy-set time and cached in
 * @p builder->use_host_radix.
 */
static cvl_cl_status_t gpu_radix_sort(cvl_cl_gpu_tree_build_t *builder, cvl_cl_chain_t *chain, void *work, unsigned n)
{
    if (builder->use_host_radix)
        return gpu_radix_sort_host(builder, chain, work, n);
    return gpu_radix_sort_orig(builder, chain, work, n);
}

/* ------------------------------------------------------------------ */
/*  Radix kernel mode resolution                                       */
/* ------------------------------------------------------------------ */

/**
 * @brief Resolve the effective radix mode from the policy + device.
 *
 * Validates that the kernels required by the resolved mode are registered
 * in the compute backend, and updates @p builder->use_host_radix.
 *
 * Policy → mode mapping:
 *   ORIGINAL   → original device kernels; refused (CVL_CL_ERR_UNSUPPORTED_DEVICE)
 *                on the Intel NEO CPU backend (which miscompiles them).
 *   WORKAROUND → host-side stable sort (no device kernels; reliable on every
 *                backend, including the NEO CPU).
 *   AUTO       → host-side sort iff the NEO CPU backend is detected.
 *
 * @param builder Builder (compute + intel_neo_cpu must be set).
 * @return CVL_CL_SUCCESS, CVL_CL_ERR_NOT_FOUND (missing kernels) or
 *         CVL_CL_ERR_UNSUPPORTED_DEVICE (ORIGINAL on NEO CPU).
 */
static cvl_cl_status_t resolve_radix_mode(cvl_cl_gpu_tree_build_t *builder)
{
    const cvl_cl_radix_policy_t p = builder->radix_policy;

    if (p == CVL_CL_RADIX_POLICY_ORIGINAL)
    {
        if (builder->intel_neo_cpu)
            return CVL_CL_ERR_UNSUPPORTED_DEVICE;

        if (!cvl_cl_compute_kernel(builder->compute, CVL_CL_PACK_BH_BUILD, CVL_CL_BH_BUILD_RADIX_HIST) ||
            !cvl_cl_compute_kernel(builder->compute, CVL_CL_PACK_BH_BUILD, CVL_CL_BH_BUILD_RADIX_SCATTER))
            return CVL_CL_ERR_NOT_FOUND;

        builder->use_host_radix = false;
        return CVL_CL_SUCCESS;
    }

    if (p == CVL_CL_RADIX_POLICY_WORKAROUND)
    {
        /* Host-side sort - no additional kernels required. */
        builder->use_host_radix = true;
        return CVL_CL_SUCCESS;
    }

    /* AUTO */
    builder->use_host_radix = builder->intel_neo_cpu;
    if (!builder->use_host_radix)
    {
        if (!cvl_cl_compute_kernel(builder->compute, CVL_CL_PACK_BH_BUILD, CVL_CL_BH_BUILD_RADIX_HIST) ||
            !cvl_cl_compute_kernel(builder->compute, CVL_CL_PACK_BH_BUILD, CVL_CL_BH_BUILD_RADIX_SCATTER))
            return CVL_CL_ERR_NOT_FOUND;
    }
    return CVL_CL_SUCCESS;
}

/* ------------------------------------------------------------------ */
/*  cvl_cl_gpu_tree_build_init                                         */
/* ------------------------------------------------------------------ */

/**
 * @brief Compute the work-buffer size for cvl_cl_gpu_tree_build_run.
 *
 * Conservative upper bound covering all temporary host-side buffers
 * needed during a single run (coords, boundary, leaf starts, radix
 * sort work, parent marks, and metadata readback).
 *
 * @param n_sources  Number of particles.
 * @return Required work-buffer size in bytes.
 */
size_t cvl_cl_gpu_tree_build_work_size(unsigned n_sources, unsigned max_depth)
{
    /* A source can contribute an internal ancestor at every tree depth, so
     * the node count is not bounded by 2 * n_sources.  Use the maximum
     * representable depth because this API now receives max_depth. */
    const size_t max_n_total = (size_t)n_sources * (max_depth + 1u);

    size_t sz = 0;
    sz += (size_t)n_sources * sizeof(real3_t);         /* coords_host */
    sz += (size_t)n_sources * sizeof(unsigned);        /* idx_host */
    sz += cvl_radix_sort_work_size((size_t)n_sources); /* radix work */
    sz += (size_t)n_sources * sizeof(int);             /* boundary_host */
    sz += (size_t)n_sources * sizeof(unsigned);        /* leaf_starts_host */
    sz += (size_t)n_sources * 3u * sizeof(unsigned);   /* parent_marks + child_marks + parent_starts_host */
    sz += max_n_total * CVL_CL_GPU_NODE_SIZE;          /* nodes_raw */
    return sz;
}

cvl_cl_status_t cvl_cl_gpu_tree_build_init(cvl_cl_gpu_tree_build_t *builder, cvl_cl_compute_t *compute,
                                           unsigned max_depth, unsigned critical_count, unsigned order)
{
    assert(builder);
    assert(compute);
    assert(max_depth <= CVL_CL_GPU_BUILD_MAX_DEPTH);

    *builder = (cvl_cl_gpu_tree_build_t){0};
    builder->radix_policy = CVL_CL_RADIX_POLICY_AUTO;

    /* Detect the Intel NEO CPU backend (see intel-neo-cpu-bug.md). */
    builder->intel_neo_cpu = cvl_cl_device_is_intel_neo_cpu(compute->device);

    /* Verify the always-required kernels exist in the compute backend. */
    if (!cvl_cl_compute_kernel(compute, CVL_CL_PACK_BH_BUILD, CVL_CL_BH_BUILD_MORTON) ||
        !cvl_cl_compute_kernel(compute, CVL_CL_PACK_BH_BUILD, CVL_CL_BH_BUILD_BOUNDARY) ||
        !cvl_cl_compute_kernel(compute, CVL_CL_PACK_BH_BUILD, CVL_CL_BH_BUILD_FILL_LEAVES) ||
        !cvl_cl_compute_kernel(compute, CVL_CL_PACK_BH_BUILD, CVL_CL_BH_BUILD_INTERNAL))
        return CVL_CL_ERR_NOT_FOUND; /* a required kernel was not registered */

    builder->compute = compute;
    builder->max_depth = max_depth;
    builder->critical_count = critical_count;
    builder->order = order;

    /* Resolve the radix mode for the default (AUTO) policy. */
    {
        cvl_cl_status_t st = resolve_radix_mode(builder);
        if (st != CVL_CL_SUCCESS)
            return st;
    }

    return CVL_CL_SUCCESS;
}

/* ------------------------------------------------------------------ */
/*  cvl_cl_gpu_tree_build_set_radix_policy                             */
/* ------------------------------------------------------------------ */

cvl_cl_status_t cvl_cl_gpu_tree_build_set_radix_policy(cvl_cl_gpu_tree_build_t *builder, cvl_cl_radix_policy_t policy)
{
    assert(builder);
    if (!builder->compute) /* Not initialised (init sets the compute backend). */
        return CVL_CL_ERR_INVALID_PARAM;
    if (policy != CVL_CL_RADIX_POLICY_AUTO && policy != CVL_CL_RADIX_POLICY_ORIGINAL &&
        policy != CVL_CL_RADIX_POLICY_WORKAROUND)
        return CVL_CL_ERR_INVALID_PARAM;

    builder->radix_policy = policy;
    return resolve_radix_mode(builder);
}

/* ------------------------------------------------------------------ */
/*  cvl_cl_gpu_tree_build_run                                          */
/* ------------------------------------------------------------------ */

cvl_cl_status_t cvl_cl_gpu_tree_build_run(cvl_cl_gpu_tree_build_t *builder, cvl_cl_staging_buffer_t *staging_pos,
                                          unsigned n_sources, void *work, size_t work_size)
{
    cvl_cl_status_t st;
    cvl_cl_status_t status = CVL_CL_SUCCESS;

    /* ---- Validate (asserts = contract; error returns = runtime checks) ---- */
    assert(builder);
    assert(builder->compute);
    assert(staging_pos);
    assert(work);

    const unsigned max_depth = builder->max_depth;

    if (n_sources == 0)
        return CVL_CL_SUCCESS;
    if (work_size < cvl_cl_gpu_tree_build_work_size(n_sources, max_depth))
        return CVL_CL_ERR_BUFFER_SIZE;

    /* Safety net: ORIGINAL radix kernels on the Intel NEO CPU backend
     * would corrupt the runtime heap - refuse instead (the policy setter
     * already prevents this, but guard here too). */
    if (builder->intel_neo_cpu && builder->radix_policy == CVL_CL_RADIX_POLICY_ORIGINAL)
        return CVL_CL_ERR_UNSUPPORTED_DEVICE;

    builder->n_sources = n_sources;

    /* Queue + context come from the borrowed compute backend. */
    const cl_command_queue queue = builder->compute->queue;
    const cl_context ctx = builder->compute->ctx;

    /* Dependency stream for all device work.  cvl_cl_chain_finish() is
     * called at every point where the host must read results back; the
     * chain is destroyed on every exit path. */
    cvl_cl_chain_t chain;
    cvl_cl_chain_init(&chain, queue);

    /* ---- Partition work buffer ---- */
    uint8_t *bp = (uint8_t *)work;
    real3_t *coords_host = (real3_t *)bp;
    bp += (size_t)n_sources * sizeof(real3_t);
    unsigned *idx_host = (unsigned *)bp;
    bp += (size_t)n_sources * sizeof(unsigned);
    void *radix_work = bp;
    bp += cvl_radix_sort_work_size((size_t)n_sources);
    int *boundary_host = (int *)bp;
    bp += (size_t)n_sources * sizeof(int);
    unsigned *leaf_starts_host = (unsigned *)bp;
    bp += (size_t)n_sources * sizeof(unsigned);
    unsigned *parent_marks = (unsigned *)bp;
    bp += (size_t)n_sources * sizeof(unsigned);
    unsigned *child_marks = (unsigned *)bp;
    bp += (size_t)n_sources * sizeof(unsigned);
    unsigned *parent_starts_host = (unsigned *)bp;
    bp += (size_t)n_sources * sizeof(unsigned);
    unsigned char *nodes_raw = (unsigned char *)bp; /* last - size varies */

    /* Zero-fill scratch: caller-provided zeros are carved from the START of
     * the nodes_raw region, which is dead until the stage-12 metadata
     * readback overwrites it.  nodes_raw is always >= (max_depth + 2) * 4
     * bytes (the largest zero fill), and the async zero writes complete
     * long before the readback reuses the region. */

    /* ================================================================ */
    /*  Stage 0 - Compute bounding box from staging buffer               */
    /* ================================================================ */

    /* Read back coords from the staging buffer through the chain
     * (blocking - the host needs them for the bounding box).  The
     * staging buffer must be FP64 (24 bytes/element = sizeof(real3_t));
     * the builder's work layout does not reserve FP32 scratch. */
    st = cvl_cl_staging_buffer_read_and_wait(staging_pos, &chain, coords_host, NULL, n_sources, 0);
    if (st != CVL_CL_SUCCESS)
    {
        status = st;
        goto cleanup;
    }

    {
        real3_t bbox_min = coords_host[0];
        real3_t bbox_max = coords_host[0];

        for (unsigned i = 1; i < n_sources; ++i)
        {
            const real_t x = coords_host[i].x;
            const real_t y = coords_host[i].y;
            const real_t z = coords_host[i].z;

            if (x < bbox_min.x)
                bbox_min.x = x;
            if (y < bbox_min.y)
                bbox_min.y = y;
            if (z < bbox_min.z)
                bbox_min.z = z;
            if (x > bbox_max.x)
                bbox_max.x = x;
            if (y > bbox_max.y)
                bbox_max.y = y;
            if (z > bbox_max.z)
                bbox_max.z = z;
        }

        const real_t root_extent =
            fmax(fmax(bbox_max.x - bbox_min.x, bbox_max.y - bbox_min.y), bbox_max.z - bbox_min.z);
        const real_t root_hs = root_extent * (real_t)0.5 + (real_t)1e-12;
        const real_t root_cx = (bbox_min.x + bbox_max.x) * (real_t)0.5;
        const real_t root_cy = (bbox_min.y + bbox_max.y) * (real_t)0.5;
        const real_t root_cz = (bbox_min.z + bbox_max.z) * (real_t)0.5;

        /* coords_host reuse ends here - bbox data fully consumed. */

        /* ================================================================ */
        /*  Stage 1 - Ensure pipeline buffers are sized for n_sources         */
        /* ================================================================ */

        const size_t elem_bytes = n_sources * sizeof(uint64_t);
        const size_t idx_bytes = n_sources * sizeof(unsigned);
        const size_t bd_bytes = n_sources * sizeof(int);
        const size_t bd_hist_bytes = (size_t)(max_depth + 2) * sizeof(unsigned);
        const size_t counter_bytes = 2u * sizeof(unsigned); /* n_leaves_out + particle_counter */

        if ((st = cl_ensure_buffer_chained(&builder->buf_morton, ctx, &chain, elem_bytes)) != CVL_CL_SUCCESS ||
            (st = cl_ensure_buffer_chained(&builder->buf_morton_tmp, ctx, &chain, elem_bytes)) != CVL_CL_SUCCESS ||
            (st = cl_ensure_buffer_chained(&builder->buf_indices, ctx, &chain, idx_bytes)) != CVL_CL_SUCCESS ||
            (st = cl_ensure_buffer_chained(&builder->buf_indices_tmp, ctx, &chain, idx_bytes)) != CVL_CL_SUCCESS ||
            (st = cl_ensure_buffer_chained(&builder->buf_boundary, ctx, &chain, bd_bytes)) != CVL_CL_SUCCESS ||
            (st = cl_ensure_buffer_chained(&builder->buf_bd_hist, ctx, &chain, bd_hist_bytes)) != CVL_CL_SUCCESS ||
            (st = cl_ensure_buffer_chained(&builder->buf_leaf_counter, ctx, &chain, counter_bytes)) != CVL_CL_SUCCESS)
        {
            status = st;
            goto cleanup;
        }

        /* ================================================================ */
        /*  Stage 2 - Initialise indices (identity permutation)              */
        /* ================================================================ */

        {
            for (unsigned i = 0; i < n_sources; ++i)
                idx_host[i] = i;

            /* Non-blocking write - tracked by the chain.  idx_host lives
             * in the work buffer until run() returns, so there is no
             * use-after-free. */
            st = cvl_cl_chain_write_buffer(&chain, &builder->buf_indices, 0, idx_bytes, idx_host, 0, NULL, NULL);
            if (st != CVL_CL_SUCCESS)
            {
                status = st;
                goto cleanup;
            }
        }

        /* ================================================================ */
        /*  Stage 3 - Launch kernel_morton                                   */
        /* ================================================================ */

        {
            cl_kernel km = cvl_cl_compute_kernel(builder->compute, CVL_CL_PACK_BH_BUILD, CVL_CL_BH_BUILD_MORTON);
            if (!km)
            {
                status = CVL_CL_ERR_INTERNAL;
                goto cleanup;
            }

            const size_t global = ((n_sources + CVL_CL_GPU_BUILD_WG - 1) / CVL_CL_GPU_BUILD_WG) * CVL_CL_GPU_BUILD_WG;
            st = cvl_cl_chain_ndrange(&chain, km, 1, &global, NULL,
                                      (cvl_cl_karg_t[]){
                                          {.type = CVL_CL_KARG_BUFFER, .index = 0, .mem = staging_pos->device.mem},
                                          {.type = CVL_CL_KARG_SCALAR_UINT, .index = 1, .scalar_uint = n_sources},
                                          {.type = CVL_CL_KARG_SCALAR_DOUBLE, .index = 2, .scalar_double = root_cx},
                                          {.type = CVL_CL_KARG_SCALAR_DOUBLE, .index = 3, .scalar_double = root_cy},
                                          {.type = CVL_CL_KARG_SCALAR_DOUBLE, .index = 4, .scalar_double = root_cz},
                                          {.type = CVL_CL_KARG_SCALAR_DOUBLE, .index = 5, .scalar_double = root_hs},
                                          {.type = CVL_CL_KARG_BUFFER, .index = 6, .mem = builder->buf_morton.mem},
                                          {},
                                      },
                                      0, NULL, NULL);
            if (st != CVL_CL_SUCCESS)
            {
                status = st;
                goto cleanup;
            }
        }

        /* ================================================================ */
        /*  Stage 4 - Radix sort Morton codes + companion indices             */
        /* ================================================================ */

        st = gpu_radix_sort(builder, &chain, radix_work, n_sources);
        if (st != CVL_CL_SUCCESS)
        {
            status = st;
            goto cleanup;
        }

        /* ================================================================ */
        /*  Stage 5 - Boundary detection                                     */
        /* ================================================================ */

        /* Zero bd_hist via caller-provided zeros carved from the work
         * buffer (nodes_raw start - dead until the stage-12 readback).
         * The async zero write completes before the boundary kernel
         * (in-order queue). */
        {
            memset(nodes_raw, 0, bd_hist_bytes);
            st = zero_device_buffer_chained(&chain, &builder->buf_bd_hist, nodes_raw, bd_hist_bytes);
        }
        if (st != CVL_CL_SUCCESS)
        {
            status = st;
            goto cleanup;
        }

        {
            cl_kernel kb = cvl_cl_compute_kernel(builder->compute, CVL_CL_PACK_BH_BUILD, CVL_CL_BH_BUILD_BOUNDARY);
            if (!kb)
            {
                status = CVL_CL_ERR_INTERNAL;
                goto cleanup;
            }

            const size_t global = ((n_sources + CVL_CL_GPU_BUILD_WG - 1) / CVL_CL_GPU_BUILD_WG) * CVL_CL_GPU_BUILD_WG;
            st = cvl_cl_chain_ndrange(&chain, kb, 1, &global, NULL,
                                      (cvl_cl_karg_t[]){
                                          {.type = CVL_CL_KARG_BUFFER, .index = 0, .mem = builder->buf_morton.mem},
                                          {.type = CVL_CL_KARG_SCALAR_UINT, .index = 1, .scalar_uint = n_sources},
                                          {.type = CVL_CL_KARG_SCALAR_UINT, .index = 2, .scalar_uint = max_depth},
                                          {.type = CVL_CL_KARG_BUFFER, .index = 3, .mem = builder->buf_boundary.mem},
                                          {.type = CVL_CL_KARG_BUFFER, .index = 4, .mem = builder->buf_bd_hist.mem},
                                          {},
                                      },
                                      0, NULL, NULL);
            if (st != CVL_CL_SUCCESS)
            {
                status = st;
                goto cleanup;
            }
        }

        /* ================================================================ */
        /*  Stage 6 - Read bd_hist → compute depth_counts / depth_offsets    */
        /* ================================================================ */

        /* boundary_host already points into the work buffer. */

        {
            unsigned *hist_raw = builder->bd_hist;

            /* Read both arrays through the chain (the reads wait on the
             * boundary kernel + zero write), then wait for the reads -
             * the host needs both arrays. */
            if ((st = cvl_cl_chain_read_buffer(&chain, &builder->buf_bd_hist, 0, bd_hist_bytes, hist_raw, 0, NULL,
                                               NULL)) != CVL_CL_SUCCESS ||
                (st = cvl_cl_chain_read_buffer(&chain, &builder->buf_boundary, 0, (size_t)n_sources * sizeof(int),
                                               boundary_host, 0, NULL, NULL)) != CVL_CL_SUCCESS ||
                (st = cvl_cl_chain_finish(&chain)) != CVL_CL_SUCCESS)
            {
                status = st;
                goto cleanup;
            }

            /* depth_counts[d] = sum_{b=0..d} bd_hist[b] */
            unsigned *dc = builder->depth_counts;
            unsigned acc = 0;
            for (unsigned d = 0; d <= max_depth; ++d)
            {
                unsigned bd_this = 0;
                for (unsigned b = 0; b <= d; ++b)
                    bd_this += hist_raw[b];
                dc[d] = bd_this;
            }

            /* Compute prefix sum → depth_offsets. */
            unsigned *doff = builder->depth_offsets;
            acc = 0;
            for (unsigned d = 0; d <= max_depth; ++d)
            {
                doff[d] = acc;
                acc += dc[d];
            }
            doff[max_depth + 1] = acc;

            builder->n_total = acc;
        }

        const unsigned n_total = builder->n_total;
        const unsigned n_leaves = builder->depth_counts[max_depth];

        if (n_leaves == 0)
        {
            /* No particles or all degenerate → nothing to build. */
            builder->n_internal = 0;
            builder->n_multipole_leaves = 0;
            builder->n_particle_leaves = 0;
            goto cleanup;
        }

        /* ================================================================ */
        /*  Stage 7 - Allocate output buffers                                */
        /* ================================================================ */

        {
            const size_t node_bytes = (size_t)n_total * CVL_CL_GPU_NODE_SIZE;
            const size_t order_bytes = (size_t)n_sources * sizeof(unsigned);
            const size_t doff_bytes = (size_t)(max_depth + 2) * sizeof(unsigned);

            /* Upload depth_offsets to device (chained). */
            if ((st = cl_ensure_buffer_chained(&builder->buf_nodes, ctx, &chain, node_bytes)) != CVL_CL_SUCCESS ||
                (st = cl_ensure_buffer_chained(&builder->buf_particle_order, ctx, &chain, order_bytes)) !=
                    CVL_CL_SUCCESS ||
                (st = cl_ensure_buffer_chained(&builder->buf_depth_offsets, ctx, &chain, doff_bytes)) !=
                    CVL_CL_SUCCESS ||
                (st = cvl_cl_chain_write_buffer(&chain, &builder->buf_depth_offsets, 0, doff_bytes,
                                                builder->depth_offsets, 0, NULL, NULL)) != CVL_CL_SUCCESS)
            {
                status = st;
                goto cleanup;
            }
        }

        /* Ensure leaf_starts buffer is at least n_sources entries. */
        st = cl_ensure_buffer_chained(&builder->buf_leaf_starts, ctx, &chain, (size_t)n_sources * sizeof(unsigned));
        if (st != CVL_CL_SUCCESS)
        {
            status = st;
            goto cleanup;
        }

        /* ================================================================ */
        /*  Stage 8 - Compute leaf starts (host-side, deterministic)         */
        /* ================================================================ */

        {
            /* leaf_starts_host already points into the work buffer. */

            unsigned n_lf = 0;
            for (unsigned i = 0; i < n_sources; ++i)
                if (boundary_host[i] <= (int)max_depth)
                    leaf_starts_host[n_lf++] = i;

            if (n_lf != n_leaves)
            {
                status = CVL_CL_ERR_INTERNAL;
                goto cleanup;
            }

            st = cvl_cl_chain_write_buffer(&chain, &builder->buf_leaf_starts, 0, (size_t)n_lf * sizeof(unsigned),
                                           leaf_starts_host, 0, NULL, NULL);
            if (st != CVL_CL_SUCCESS)
            {
                status = st;
                goto cleanup;
            }
        }

        /* ================================================================ */
        /*  Stage 10 - Build leaf nodes (kernel_fill_leaves)                  */
        /* ================================================================ */

        {
            /* Zero the leaf_counter (both words) via caller-provided zeros
             * carved from the work buffer.  The kernel only accumulates
             * into word 1 (particle_counter) with atomic_add; word 0
             * (n_leaves_out) starts at zero. */
            memset(nodes_raw, 0, counter_bytes);
            st = zero_device_buffer_chained(&chain, &builder->buf_leaf_counter, nodes_raw, counter_bytes);
            if (st != CVL_CL_SUCCESS)
            {
                status = st;
                goto cleanup;
            }
        }

        {
            cl_kernel kf = cvl_cl_compute_kernel(builder->compute, CVL_CL_PACK_BH_BUILD, CVL_CL_BH_BUILD_FILL_LEAVES);
            if (!kf)
            {
                status = CVL_CL_ERR_INTERNAL;
                goto cleanup;
            }

            const unsigned leaf_offset = builder->depth_offsets[max_depth];

            const size_t global = ((n_leaves + CVL_CL_GPU_BUILD_WG - 1) / CVL_CL_GPU_BUILD_WG) * CVL_CL_GPU_BUILD_WG;
            st = cvl_cl_chain_ndrange(
                &chain, kf, 1, &global, NULL,
                (cvl_cl_karg_t[]){
                    {.type = CVL_CL_KARG_BUFFER, .index = 0, .mem = builder->buf_leaf_starts.mem},
                    {.type = CVL_CL_KARG_SCALAR_UINT, .index = 1, .scalar_uint = n_leaves},
                    {.type = CVL_CL_KARG_BUFFER, .index = 2, .mem = builder->buf_boundary.mem},
                    {.type = CVL_CL_KARG_SCALAR_UINT, .index = 3, .scalar_uint = n_sources},
                    {.type = CVL_CL_KARG_SCALAR_UINT, .index = 4, .scalar_uint = max_depth},
                    {.type = CVL_CL_KARG_BUFFER, .index = 5, .mem = builder->buf_nodes.mem},
                    {.type = CVL_CL_KARG_SCALAR_UINT, .index = 6, .scalar_uint = leaf_offset},
                    {.type = CVL_CL_KARG_BUFFER, .index = 7, .mem = builder->buf_particle_order.mem},
                    {.type = CVL_CL_KARG_BUFFER, .index = 8, .mem = builder->buf_leaf_counter.mem},
                    {.type = CVL_CL_KARG_BUFFER, .index = 9, .mem = staging_pos->device.mem},
                    {.type = CVL_CL_KARG_BUFFER, .index = 10, .mem = builder->buf_morton.mem},
                    {.type = CVL_CL_KARG_BUFFER, .index = 11, .mem = builder->buf_indices.mem},
                    {.type = CVL_CL_KARG_SCALAR_DOUBLE, .index = 12, .scalar_double = root_hs},
                    {.type = CVL_CL_KARG_SCALAR_UINT, .index = 13, .scalar_uint = builder->critical_count},
                    {},
                },
                0, NULL, NULL);
            if (st != CVL_CL_SUCCESS)
            {
                status = st;
                goto cleanup;
            }
        }

        /* ================================================================ */
        /*  Stage 11 - Build internal nodes bottom-up                        */
        /* ================================================================ */

        for (int d = (int)max_depth - 1; d >= 0; --d)
        {
            const unsigned depth = (unsigned)d;
            const unsigned n_parents = builder->depth_counts[depth];
            const unsigned parent_off = builder->depth_offsets[depth];
            const unsigned child_off = builder->depth_offsets[depth + 1];
            const unsigned n_children = builder->depth_counts[depth + 1];

            if (n_parents == 0)
                continue;

            /* (a) Host-side child-range computation.
             * parent_marks, child_marks, parent_starts_host point into work buffer. */
            unsigned pm = 0;
            for (unsigned i = 0; i < n_sources; ++i)
                if (boundary_host[i] <= (int)depth)
                    parent_marks[pm++] = i;
            parent_marks[pm] = n_sources; /* sentinel */

            unsigned cm = 0;
            for (unsigned i = 0; i < n_sources; ++i)
                if (boundary_host[i] <= (int)(depth + 1))
                    child_marks[cm++] = i;
            child_marks[cm] = n_sources; /* sentinel */

            if (pm != n_parents || cm != n_children)
            {
                status = CVL_CL_ERR_INTERNAL;
                goto cleanup;
            }

            unsigned g = 0;
            for (unsigned p = 0; p < n_parents; ++p)
            {
                while (g < n_children && child_marks[g] < parent_marks[p])
                    ++g;
                const unsigned g_lo = g;
                while (g < n_children && child_marks[g] < parent_marks[p + 1])
                    ++g;
                parent_starts_host[p] = child_off + g_lo;
            }
            st = cvl_cl_chain_write_buffer(&chain, &builder->buf_leaf_starts, 0, (size_t)n_parents * sizeof(unsigned),
                                           parent_starts_host, 0, NULL, NULL);
            if (st != CVL_CL_SUCCESS)
            {
                status = st;
                goto cleanup;
            }

            /* (b) Build the parents. */
            {
                cl_kernel ki = cvl_cl_compute_kernel(builder->compute, CVL_CL_PACK_BH_BUILD, CVL_CL_BH_BUILD_INTERNAL);
                if (!ki)
                {
                    status = CVL_CL_ERR_INTERNAL;
                    goto cleanup;
                }

                const size_t global =
                    ((n_parents + CVL_CL_GPU_BUILD_WG - 1) / CVL_CL_GPU_BUILD_WG) * CVL_CL_GPU_BUILD_WG;
                st = cvl_cl_chain_ndrange(
                    &chain, ki, 1, &global, NULL,
                    (cvl_cl_karg_t[]){
                        {.type = CVL_CL_KARG_BUFFER, .index = 0, .mem = builder->buf_nodes.mem},
                        {.type = CVL_CL_KARG_SCALAR_UINT, .index = 1, .scalar_uint = depth},
                        {.type = CVL_CL_KARG_SCALAR_UINT, .index = 2, .scalar_uint = n_parents},
                        {.type = CVL_CL_KARG_SCALAR_UINT, .index = 3, .scalar_uint = parent_off},
                        {.type = CVL_CL_KARG_SCALAR_UINT, .index = 4, .scalar_uint = child_off},
                        {.type = CVL_CL_KARG_SCALAR_UINT, .index = 5, .scalar_uint = n_children},
                        {.type = CVL_CL_KARG_SCALAR_DOUBLE, .index = 6, .scalar_double = root_hs},
                        {.type = CVL_CL_KARG_BUFFER, .index = 7, .mem = builder->buf_leaf_starts.mem},
                        {},
                    },
                    0, NULL, NULL);
                if (st != CVL_CL_SUCCESS)
                {
                    status = st;
                    goto cleanup;
                }
            }
        }

        /* ================================================================ */
        /*  Stage 12 - Read back metadata                                    */
        /* ================================================================ */

        {
            const unsigned leaf_offset = builder->depth_offsets[max_depth];

            /* Read the nodes back through the chain (waits on the full
             * build pipeline), then wait for the read.  nodes_raw was
             * partitioned at function start (sized for max_n_total) - its
             * first bytes were reused as the zero-fill scratch, which is
             * safe: those writes completed long ago. */
            const size_t raw_bytes = (size_t)n_total * CVL_CL_GPU_NODE_SIZE;
            if ((st = cvl_cl_chain_read_buffer(&chain, &builder->buf_nodes, 0, raw_bytes, nodes_raw, 0, NULL, NULL)) !=
                    CVL_CL_SUCCESS ||
                (st = cvl_cl_chain_finish(&chain)) != CVL_CL_SUCCESS)
            {
                status = st;
                goto cleanup;
            }

            unsigned n_mp = 0, n_pt = 0;
            for (unsigned i = 0; i < n_leaves; ++i)
            {
                /* BH_KIND_PARTICLE = 1, BH_KIND_MULTIPOLE = 2 */
                const unsigned char kind = nodes_raw[((size_t)(leaf_offset + i) * CVL_CL_GPU_NODE_SIZE) + 49u];
                if (kind == 2)
                    ++n_mp;
                else
                    ++n_pt;
            }

            builder->n_internal = n_total - n_leaves;
            builder->n_multipole_leaves = n_mp;
            builder->n_particle_leaves = n_pt;
        }
    }

cleanup:
    cvl_cl_chain_destroy(&chain);
    return status;
}

/* ------------------------------------------------------------------ */
/*  cvl_cl_gpu_tree_build_destroy                                       */
/* ------------------------------------------------------------------ */

void cvl_cl_gpu_tree_build_destroy(cvl_cl_gpu_tree_build_t *builder)
{
    if (!builder)
        return;

    cvl_cl_buffer_destroy(&builder->buf_morton);
    cvl_cl_buffer_destroy(&builder->buf_morton_tmp);
    cvl_cl_buffer_destroy(&builder->buf_indices);
    cvl_cl_buffer_destroy(&builder->buf_indices_tmp);
    cvl_cl_buffer_destroy(&builder->buf_boundary);
    cvl_cl_buffer_destroy(&builder->buf_radix_hist);
    cvl_cl_buffer_destroy(&builder->buf_bd_hist);
    cvl_cl_buffer_destroy(&builder->buf_nodes);
    cvl_cl_buffer_destroy(&builder->buf_particle_order);
    cvl_cl_buffer_destroy(&builder->buf_depth_offsets);
    cvl_cl_buffer_destroy(&builder->buf_leaf_starts);
    cvl_cl_buffer_destroy(&builder->buf_leaf_counter);

    *builder = (cvl_cl_gpu_tree_build_t){0};
}
