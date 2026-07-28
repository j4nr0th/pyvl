# Parallelization Patterns for pyvl Core Kernels

This document captures patterns, conventions, and lessons from parallelizing
the Morton-code radix sort in `octree.c`. Use it as a reference when adding
OpenMP parallelism to other functions.

---

## 1. Scratch Buffer Reuse (The `radix_hist` Pattern)

The scratch buffer (`octree_scratch_t`) holds per-thread temporaries that are
only live during specific pipeline stages.  A single region can serve multiple
purposes across stages — no need for dedicated space per use.

**Example — `radix_hist` triple-use in `octree_build_morton_sorted`:**

| Stage | What `radix_hist[tid * RADIX_BINS + b]` stores |
|-------|--------------------------------------------------|
| Depth counting | Per-thread depth histogram (count of nodes at each depth) |
| Depth-node index | Per-thread cursor offset into `sorted_indices` for each depth |
| Radix sort | Per-thread histogram for each radix pass (256 bins) |

These three uses are **sequential** — each finishes before the next starts.
No need to zero between uses (each stage writes its own data).

**Rule:** If a scratch region is dead after stage X and needed before stage Y,
reuse it. Document the reuse in comments.

**Sizing:** Always use named constants, not magic numbers:
```c
// Good
const size_t radix_hist_per_thread = (size_t)RADIX_BINS * sizeof(unsigned);

// Bad
const size_t radix_hist_per_thread = 256u * sizeof(unsigned);
```

---

## 2. Per-Thread Histograms → No Atomics

When multiple threads increment counters keyed by a small domain (depth levels,
histogram bins), use per-thread arrays instead of `#pragma omp atomic`:

```c
// BAD: serialized atomic increments
unsigned depth_buf[256] = {0};
#pragma omp parallel for
for (...) {
    unsigned d = nodes[i].depth;
    #pragma omp atomic
    depth_buf[d]++;
}

// GOOD: per-thread histograms, serial reduce
#pragma omp parallel for
for (unsigned t = 0; t < n_threads; ++t)
    for (unsigned d = 0; d < OCTREE_MAX_DEPTH; ++d)
        radix_hist[t * RADIX_BINS + d] = 0;

#pragma omp parallel
{
    unsigned *h = radix_hist + tid * RADIX_BINS;
    #pragma omp for
    for (...) {
        unsigned d = nodes[i].depth;
        h[d]++;
    }
}
// Serial reduce:
for (unsigned d = 0; d < OCTREE_MAX_DEPTH; ++d)
    for (unsigned t = 0; t < n_threads; ++t)
        depth_buf[d] += radix_hist[t * RADIX_BINS + d];
```

**When to use:** Domain size is small (≤ 1024) and thread count is moderate
(≤ 64). The per-thread array fits in L1 cache.

**When NOT to use:** Domain is large (e.g., particle indices). Use
`#pragma omp atomic` or task-based reduction instead.

---

## 3. Per-Thread Cursors → No Atomics for Scatter

When each thread writes to disjoint output positions determined by a key,
transform per-thread counts into per-thread starting offsets, then scatter:

```c
// Stage 1: per-thread depth counts (as above)

// Stage 2: transform counts → offsets
for (unsigned d = 0; d <= max_depth_found; ++d) {
    unsigned acc = depth_offsets[d];
    for (unsigned t = 0; t < n_threads; ++t) {
        unsigned cnt = radix_hist[t * RADIX_BINS + d];
        radix_hist[t * RADIX_BINS + d] = acc;  // overwrite: count → offset
        acc += cnt;
    }
}

// Stage 3: scatter — each thread uses its own cursor, no atomics
#pragma omp parallel
{
    unsigned *my_cursors = radix_hist + tid * RADIX_BINS;
    #pragma omp for
    for (unsigned i = 0; i < n_nodes; ++i) {
        unsigned d = nodes[i].depth;
        unsigned pos = my_cursors[d];
        sorted_indices[pos] = i;
        my_cursors[d]++;
    }
}
```

**Key insight:** The prefix-sum over (depth, thread) gives each thread a
non-overlapping write region per depth. Thread T writes to
`[offset[T][d], offset[T][d] + count[T][d])` which is disjoint from
thread T' ≠ T.

---

## 4. Conditional Parallelism (`if` clause)

Use `#pragma omp parallel for if (condition)` to avoid parallel overhead
for small workloads:

```c
#pragma omp parallel for if (ndepth > 1024) default(none) shared(...)
for (unsigned j = 0; j < ndepth; ++j) { ... }
```

**Threshold guidelines:**
- **Simple memory ops** (gather, scatter, memcpy): `ndepth > 1024`
- **Compute-heavy** (multipole eval, M2L): `ndepth > 64`
- **Tree traversal** (leaf vlists): always parallel (work per iteration varies)

---

## 5. Named Constants Over Magic Numbers

```c
// GOOD
enum {
    OCTREE_MAX_DEPTH = 256u,   // uint8_t depth field → max 255
    RADIX_BITS      = 8u,
    RADIX_BINS      = (1u << RADIX_BITS),
    RADIX_PASSES    = (64u / RADIX_BITS),
};

// BAD
unsigned depth_buf[256];
for (unsigned pass = 0; pass < 8; ++pass)
```

**Rule:** If a constant appears more than once, name it. If it has a physical
meaning (max depth = 255 because depth is uint8_t), document it.

---

## 6. `default(none)` — Always Use It

Every `#pragma omp parallel` / `parallel for` must use `default(none)` and
list every variable in `shared(...)`:

```c
#pragma omp parallel for default(none) shared(n_nodes, nodes, codes, root_gc, root_hs) schedule(static)
```

**Why:** Without `default(none)`, OpenMP may implicitly determine sharing,
which can cause data races or silently shared variables that should be private.
With `default(none)`, the compiler catches missing variables.

**Exception:** `#pragma omp simd` does not support `default(none)`.

---

## 7. Schedule Selection

| Pattern | Schedule | Why |
|---------|----------|-----|
| Uniform work per iteration | `schedule(static)` | Lowest overhead, good cache |
| Variable work (tree walks) | `schedule(dynamic)` | Load balance |
| Heavy, variable work | `schedule(dynamic, 16)` | Chunked to reduce overhead |
| Nested loops, uniform | `schedule(static, 256)` | Cache-friendly chunking |

---

## 8. Reduction vs Per-Thread + Serial Reduce

| Pattern | When to use |
|---------|-------------|
| `reduction(+:sum)` | Single scalar accumulator |
| `reduction(&&:flag)` | Boolean AND across threads |
| `reduction(max:val)` | Finding maximum |
| Per-thread array + serial reduce | Histogram, multiple counters |

**Rule:** Use `reduction()` for single values. Use per-thread arrays when
the reduction domain has more than ~4 elements.

---

## 9. SIMD Annotations

Add `#pragma omp simd` to loops where the compiler might not auto-vectorize:

```c
#pragma omp simd
for (unsigned b = 0; b < RADIX_BINS; ++b)
    h[b] = 0;
```

**When:** Short fixed-count loops (256 iterations), no aliasing concerns,
simple arithmetic.

**When NOT:** Loops with function calls, pointer-chasing, or complex
indexing that the compiler can't prove independent.

---

## 10. Thread ID via Atomic Counter (No `omp_get_thread_num`)

Every parallel region receives an explicit `n_threads` parameter — never call
`omp_get_max_threads()` or rely on global `OMP_NUM_THREADS`.  Inside the
region, obtain the thread ID via an atomic-capture counter instead of
`omp_get_thread_num()`.  This eliminates the `omp.h` dependency for the
runtime query functions.

**Pattern — split `parallel for` into `parallel` + `for`:**

```c
// BEFORE — depends on omp.h:
#pragma omp parallel for default(none) shared(...) num_threads(n_threads)
for (unsigned i = 0; i < n; ++i) {
    int tid = omp_get_thread_num();
    my_buf[tid] = work(i);
}

// AFTER — no omp_get_thread_num:
unsigned thread_counter = 0;
#pragma omp parallel default(none) shared(thread_counter, ...) num_threads(n_threads)
{
    unsigned tid;
    #pragma omp atomic capture
    tid = thread_counter++;
    #pragma omp for
    for (unsigned i = 0; i < n; ++i) {
        my_buf[tid] = work(i);
    }
}
```

**Why:** `omp_get_thread_num()` requires `#include <omp.h>`.  The atomic
capture runs once per thread at region entry (O(n_threads) total), not per
iteration — negligible overhead.

**Exception:** `#pragma omp parallel for` with `schedule(dynamic)` where
`omp_get_thread_num()` is called inside the loop body.  Split into
`parallel` + `for` as above; the atomic capture still runs once per thread.

**Rule for `n_threads` parameter:**
- Every function that spawns parallel regions must take an explicit
  `unsigned n_threads` parameter.
- Every `#pragma omp parallel` / `parallel for` must include
  `num_threads(n_threads)`.
- Never call `omp_get_max_threads()` or `omp_set_num_threads()`.

---

## 11. Parallelization Opportunities in Current Codebase

### High-Impact Candidates

| Function | File | Current | Opportunity |
|----------|------|---------|-------------|
| `fmm_build_leaf_index_map` | `fmm_tree.c` | Serial O(N) | Simple `#pragma omp parallel for` — each leaf_id write is independent |
| `fmm_assign_local_slices` | `fmm_tree.c` | Serial O(N) | Simple `#pragma omp parallel for` — each node gets independent slice |
| `octree_compute_metadata` | `octree.c` | Serial O(N) | Per-thread cursors for `particle_begin` + `leaf_id` assignment |

### Low-Impact / Not Worth It

| Function | File | Why Not |
|----------|------|---------|
| `octree_count_pass` | `octree.c` | Tree insertion is pointer-chasing — inherently serial |
| `octree_materialize` | `octree.c` | DFS tree walk — serial by nature |
| `multipole_to_local` | `fmm_operators.c` | Loop bounds are order+1 (≤ 10) — overhead > benefit |
| `local_expansion_shift` | `fmm_operators.c` | Same — tiny loop bounds |
| `local_expansion_eval` | `fmm_operators.c` | Same — tiny loop bounds |
| `particle_to_local` | `fmm_operators.c` | Same — tiny loop bounds |
| `barnes_hut_tree_eval` | `barnes_hut_tree.c` | Single-point stack traversal — already parallelized at batch level |
| `octree_compute_depth_ranges` | `octree.c` | O(N) with trivial work per iteration — parallel overhead dominates |

### Already Well-Parallelized

| Function | File | Pattern |
|----------|------|---------|
| `octree_descend` | `octree.c` | `parallel for schedule(static)` |
| `octree_fill_particle_order` | `octree.c` | `parallel for` + `atomic capture` |
| `octree_compute_leaf_centers` | `octree.c` | `parallel for schedule(static)` |
| `octree_build_leaf_multipoles` | `octree.c` | `parallel` + `for reduction(&&)` |
| `octree_upward_sweep_level` | `octree.c` | `parallel for schedule(dynamic, 16)` |
| `fmm_build_leaf_vlists` | `fmm_tree.c` | `parallel for schedule(dynamic)` |
| `fmm_build_per_node_interaction_lists` | `fmm_tree.c` | `parallel for schedule(dynamic)` |
| `fmm_m2l_sweep_mlvl` | `fmm_tree.c` | `parallel for schedule(dynamic, 16)` |
| `fmm_downward_l2l_sweep` | `fmm_tree.c` | Per-depth `parallel for schedule(dynamic, 16)` |
| `fmm_tree_eval_all` | `fmm_tree.c` | `parallel for schedule(dynamic)` |
| `barnes_hut_tree_eval_all` | `barnes_hut_tree.c` | `parallel for schedule(static)` |
| `octree_build_morton_sorted` | `octree.c` | Multiple parallel stages (see above) |

---

## 11. Checklist for Adding Parallelism

1. **Identify independence:** Can iterations run in any order? If not, can you
   use per-thread temporaries to break dependencies?

2. **Choose scratch or stack:** Per-thread data < 4KB → stack array is fine.
   Larger → add to `octree_scratch_sizes_t` / `octree_scratch_t`.

3. **Name the constants:** If you need a magic number, give it an `enum`
   constant with a Doxygen comment explaining why that value.

4. **Write the parallel region:**
   - `default(none)` with explicit `shared(...)`
   - Appropriate `schedule(...)`
   - `if (N > threshold)` for small workloads

5. **Verify with assertions:** Add `assert()` for invariants that the
   parallel code depends on (e.g., cursor overflow checks).

6. **Test:** Run `ctest -R "fmm_tree"` and the full test suite. Compare
   results bit-exact with serial version for small N.

7. **Benchmark:** Measure at N=1000, 10000, 100000. If N=1000 is slower
   than serial, add an `if (N > threshold)` guard.
