#pragma once
/*
 * Parallel LSD radix sort for (uint64_t key, ...) pairs.
 *
 * Provides a lock-free, OpenMP-parallelised 8-pass radix sort for
 * 64-bit keys.  The sort operates on a flat byte array of interleaved
 * (key, payload) pairs and is stable (preserves input order of equal
 * keys) when the scatter step processes elements in ascending index
 * order.
 *
 * Usage:
 *   // Allocate buffers.
 *   uint8_t *pairs     = malloc(n * pair_size);
 *   uint8_t *pairs_alt = malloc(n * pair_size);
 *   unsigned *hist     = malloc(n_threads * RADIX_BINS * sizeof(unsigned));
 *
 *   // Fill pairs with interleaved (key, payload) data.
 *   cvl_radix_sort_pairs(pairs, pairs_alt, n, pair_size, hist, n_threads);
 *   // Result is in `pairs` (the original pointer - see return note).
 *
 *   free(hist); free(pairs_alt); free(pairs);
 *
 * The caller must provide two same-sized buffers (pairs and pairs_alt)
 * for ping-pong.  After sorting, the result is in whichever buffer the
 * internal pointer variable references (the function swaps local
 * pointers, not the caller's pointers - so the original `pairs` argument
 * always points to the result).
 *
 * Work buffer sizing:
 *   Use CVL_RADIX_SORT_WORK_SIZE(n) to compute the total scratch bytes
 *   needed.  Partition the buffer as:
 *     pairs     = work
 *     pairs_alt = work + n * pair_size
 *     hist      = pairs_alt + n * pair_size
 *     staging   = hist + CVL_RADIX_BINS * sizeof(unsigned)
 *   where pair_size = sizeof(uint64_t) + sizeof(unsigned) = 16.
 */

#include <omp.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* Constants                                                          */
/* ------------------------------------------------------------------ */

/** @brief Number of bits per radix sort pass. */
enum
{
    CVL_RADIX_BITS = 8u,
    /** @brief Number of histogram bins (2^CVL_RADIX_BITS). */
    CVL_RADIX_BINS = (1u << CVL_RADIX_BITS),
    /** @brief Number of passes for 64-bit keys (64 / CVL_RADIX_BITS). */
    CVL_RADIX_PASSES = (64u / CVL_RADIX_BITS),
    /** @brief Size of an interleaved (uint64_t key, unsigned idx) pair. */
    CVL_RADIX_SORT_PAIR_SIZE = sizeof(uint64_t) + sizeof(unsigned),
};

/* ------------------------------------------------------------------ */
/* Work buffer sizing                                                 */
/* ------------------------------------------------------------------ */

/**
 * @brief Compute the total work-buffer size for a host-side radix sort.
 *
 * The buffer must be at least this many bytes and is partitioned as:
 *   pairs [n × pair_size] | pairs_alt [n × pair_size] | hist [CVL_RADIX_BINS × sizeof(unsigned)] | staging [n ×
 * (sizeof(uint64_t) + sizeof(unsigned))]
 *
 * @param n  Number of elements to sort.
 * @return Required work-buffer size in bytes.
 */
static inline size_t cvl_radix_sort_work_size(size_t n)
{
    const size_t pair_bytes = n * CVL_RADIX_SORT_PAIR_SIZE;
    return pair_bytes + pair_bytes + CVL_RADIX_BINS * sizeof(unsigned) + pair_bytes;
}

/* ------------------------------------------------------------------ */
/* Internal: parallel scatter                                         */
/* ------------------------------------------------------------------ */

/**
 * @brief Scatter (key, payload) pairs from src to dst using pre-computed
 *        per-thread offsets.
 *
 * Each thread copies its chunk of elements to the destination using
 * thread-local offset copies, so no cross-thread atomics or locks are
 * needed during the scatter.
 *
 * @param src            Source pair array.
 * @param dst            Destination pair array.
 * @param n              Number of elements.
 * @param pair_size      Size of each interleaved (key, payload) pair in bytes.
 * @param shift          Bit shift for the current radix digit.
 * @param thread_offsets Pre-computed per-thread scatter offsets
 *                       [n_threads × CVL_RADIX_BINS].
 * @param n_threads      Number of OpenMP threads.
 */
static inline void cvl_radix_scatter(const uint8_t *src, uint8_t *dst, size_t n, size_t pair_size, unsigned shift,
                                     const unsigned *thread_offsets, unsigned n_threads)
{
#pragma omp parallel default(none) shared(src, dst, n, pair_size, shift, thread_offsets, n_threads)                    \
    num_threads(n_threads)
    {
        const unsigned tid = (unsigned)omp_get_thread_num();
        const size_t chunk = (n + (size_t)n_threads - 1) / (size_t)n_threads;
        const size_t start = (size_t)tid * chunk;
        const size_t end = start + chunk > n ? n : start + chunk;

        /* Copy per-thread offsets to local array (no cross-thread reads after). */
        unsigned my_offsets[CVL_RADIX_BINS];
        const unsigned *src_off = thread_offsets + (size_t)tid * CVL_RADIX_BINS;
        for (unsigned b = 0; b < CVL_RADIX_BINS; ++b)
            my_offsets[b] = src_off[b];

        for (size_t i = start; i < end; ++i)
        {
            const uint64_t key = *(const uint64_t *)(src + i * pair_size);
            const unsigned bin = (unsigned)((key >> shift) & (CVL_RADIX_BINS - 1));
            const size_t dst_idx = (size_t)my_offsets[bin] * pair_size;
            memcpy(dst + dst_idx, src + i * pair_size, pair_size);
            my_offsets[bin]++;
        }
    }
}

/* ------------------------------------------------------------------ */
/* Public: parallel LSD radix sort                                    */
/* ------------------------------------------------------------------ */

/**
 * @brief Parallel LSD radix sort for interleaved (uint64_t key, payload) pairs.
 *
 * Sorts @p n pairs in-place using 8 passes of histogram → reduce →
 * prefix-sum → scatter.  The key is read from the first 8 bytes of
 * each pair; the remaining @p pair_size - 8 bytes are payload that
 * moves with the key.
 *
 * @param pairs      Array of interleaved (key, payload) pairs [n × pair_size].
 * @param pairs_alt  Same-sized scratch buffer for ping-pong.
 * @param n          Number of elements.
 * @param pair_size  Size of each pair in bytes (must be ≥ 8).
 * @param radix_hist Per-thread histogram scratch [n_threads × CVL_RADIX_BINS].
 * @param n_threads  Number of OpenMP threads.
 */
static inline void cvl_radix_sort_pairs(uint8_t *pairs, uint8_t *pairs_alt, size_t n, size_t pair_size,
                                        unsigned *radix_hist, unsigned n_threads)
{
    for (unsigned pass = 0; pass < CVL_RADIX_PASSES; ++pass)
    {
        const unsigned shift = pass * CVL_RADIX_BITS;

        /* Zero per-thread histograms. */
#pragma omp parallel for default(none) shared(n_threads, radix_hist) schedule(static) num_threads(n_threads)
        for (unsigned t = 0; t < n_threads; ++t)
        {
            unsigned *h = radix_hist + (size_t)t * CVL_RADIX_BINS;
#pragma omp simd
            for (unsigned i = 0; i < CVL_RADIX_BINS; ++i)
                h[i] = 0;
        }

        /* Histogram: each thread counts its chunk. */
#pragma omp parallel default(none) shared(n, pairs, shift, radix_hist, n_threads, pair_size) num_threads(n_threads)
        {
            const unsigned tid = (unsigned)omp_get_thread_num();
            unsigned *h = radix_hist + (size_t)tid * CVL_RADIX_BINS;
            const size_t chunk = (n + (size_t)n_threads - 1) / (size_t)n_threads;
            const size_t start = (size_t)tid * chunk;
            const size_t end = start + chunk > n ? n : start + chunk;
            for (size_t i = start; i < end; ++i)
            {
                const uint64_t key = *(const uint64_t *)(pairs + i * pair_size);
                h[(key >> shift) & (CVL_RADIX_BINS - 1)]++;
            }
        }

        /* Reduce + prefix-sum: combine thread histograms into global offsets,
         * then pre-compute per-thread scatter offsets. */
        {
            unsigned global_hist[CVL_RADIX_BINS];
            for (unsigned b = 0; b < CVL_RADIX_BINS; ++b)
            {
                unsigned sum = 0;
                for (unsigned t = 0; t < n_threads; ++t)
                    sum += radix_hist[(size_t)t * CVL_RADIX_BINS + b];
                global_hist[b] = sum;
            }
            unsigned acc = 0;
            for (unsigned b = 0; b < CVL_RADIX_BINS; ++b)
            {
                const unsigned tmp = global_hist[b];
                global_hist[b] = acc;
                acc += tmp;
            }
            for (unsigned b = 0; b < CVL_RADIX_BINS; ++b)
            {
                unsigned off = global_hist[b];
                for (unsigned t = 0; t < n_threads; ++t)
                {
                    const unsigned cnt = radix_hist[(size_t)t * CVL_RADIX_BINS + b];
                    radix_hist[(size_t)t * CVL_RADIX_BINS + b] = off;
                    off += cnt;
                }
            }
        }

        /* Scatter: move pairs to temp buffer using pre-computed offsets. */
        cvl_radix_scatter(pairs, pairs_alt, n, pair_size, shift, radix_hist, n_threads);

        /* Swap current and temp buffers. */
        {
            uint8_t *tmp = pairs;
            pairs = pairs_alt;
            pairs_alt = tmp;
        }
    }
    /* After 8 passes (even), the sorted result is in the buffer originally
     * pointed to by `pairs`.  The local pointer swap inside the loop means
     * the caller's `pairs` argument still points to the result. */
}
