.. _pyvl.private_c.fmm_tree:

Fast Multipole Tree
===================

The FMM tree module implements an adaptive octree over vortex-particle
sources, building precomputed interaction lists (V-list and near-field
list) for O(N log N) far-field evaluation.  The tree is defined in
``src/core/fmm_tree.h``.

Unlike the Barnes-Hut tree (:ref:`barnes_hut_tree <pyvl.private_c.barnes_hut_tree>`),
which evaluates each target by walking the tree with a MAC at every node,
the FMM tree pre-computes well-separated interactions at build time.
This gives better constants per-target and is the foundation for the
true O(N) FMM using local expansions.

Build Pipeline
--------------

The build is exposed through a **staged API** (steps 1-7 below):

1. **Size scratch** — :c:func:`fmm_scratch_size` returns the transient
   scratch buffer size from input parameters alone.
2. **Allocate scratch** — caller provides a buffer of that size.
3. **Prepare scratch** — :c:func:`fmm_prepare_scratch` partitions the
   scratch buffer, zeros the topology array, runs the **count pass**
   (:c:func:`octree_count_pass`), and returns the
   :c:type:`octree_count_t` and :c:type:`octree_scratch_t` views.
4. **Size work** — :c:func:`fmm_work_size` returns the persistent work
   buffer size from the count-pass results.
5. **Allocate work** — caller provides the work buffer.
6. **Insert pass** — :c:func:`fmm_tree_insert` runs the full shared
   pipeline (8 common stages + 2 FMM-specific stages) and fills the
   output :c:type:`fmm_tree_t`.
7. **Release scratch** — the caller may free the scratch buffer; the
   work buffer remains owned by the tree handle.

The insert pipeline stages (steps inside stage 6) are:

*Shared octree stages (first 8):*

1. **Count pass** → :c:func:`octree_count_pass` (inside step 3)
2. **Materialise** → :c:func:`octree_materialize`
3. **Descend** → :c:func:`octree_descend`
4. **Metadata** → :c:func:`octree_compute_metadata`
5. **Fill** → :c:func:`octree_fill_particle_order`
6. **Centroids** → :c:func:`octree_compute_leaf_centers`
7. **P2M** → :c:func:`octree_build_leaf_multipoles`
8. **M2M** → :c:func:`octree_run_upward_sweep`

*FMM-specific stages:*

9. **Interaction lists** — for each leaf, classifies every other leaf
   as V-list (well-separated: :math:`|\Delta\mathbf{c}| \ge 3\max(h_a, h_b)`)
   or near-field (everything else).  Stored as CSR arrays.

10. **M2L** (FMM mode only) — converts V-list multipoles to local
    expansions at each leaf via :c:func:`multipole_to_local`.

The L2L (local-to-local) downward-sweep stage was removed during a
refactoring — it was dead code because M2L is done at leaf level only.
The :c:func:`local_expansion_shift` function is still available as a
library function for external callers.

Typical staged usage::

    // 1. Size scratch
    size_t scratch_sz = fmm_scratch_size(n, settings, n_threads);
    // 2. Allocate scratch
    void *scratch = malloc(scratch_sz);
    // 3. Prepare scratch + count
    octree_count_t count;
    octree_scratch_t sview;
    fmm_prepare_scratch(scratch, scratch_sz, n, n_threads,
                        coords, settings, &count, &sview);
    // 4. Size work
    size_t work_sz = fmm_work_size(n, settings, &count);
    // 5. Allocate work
    void *work = malloc(work_sz);
    // 6. Full pipeline
    fmm_tree_t tree;
    fmm_tree_insert(n, n_threads, coords, values, settings,
                    &count, &sview, allocator, work, work_sz, &tree);
    // 7. Free scratch, keep work buffer (owned by tree)
    free(scratch);

Sizing
------

* Scratch buffer: use :c:func:`fmm_scratch_size` (which delegates to
  :c:func:`octree_scratch_size`).

* Work buffer: the FMM tree adds extra regions (leaf indices, local
  coefficients, local slice pointers, interaction list CSR) beyond the
  common ``octree_base_work_sizes_t``.  Use the FMM-specific
  :c:func:`fmm_size_work_buffer` and :c:func:`fmm_total_work_size`.

* Count pass: :c:func:`fmm_prepare_scratch` handles this internally.


Evaluation
----------

Two evaluation modes are available (controlled by ``eval_settings.mode``):

- **Tree-code mode** (``FMM_EVAL_TREE_CODE``): descends to the target's
  leaf, sums V-list multipoles via ``multipole_eval`` for the far-field,
  and performs a direct particle sum over the own leaf and near-field
  neighbours for the near-field.

- **FMM mode** (``FMM_EVAL_FMM``): evaluates the leaf's precomputed
  local expansion via ``local_expansion_eval`` for the far-field, plus
  the same near-field direct sum.  This is a single L2P call per target
  instead of one ``multipole_eval`` per V-list entry, giving O(N) scaling.

Comparison with Barnes-Hut
--------------------------

+-----------------------------------+-------------------------------------------+
| Barnes-Hut                        | FMM                                       |
+===================================+===========================================+
| Per-target tree walk with MAC     | Precomputed V-list; one L2P or            |
|                                   | per-V-list multipole_eval per target      |
+-----------------------------------+-------------------------------------------+
| O(N log N) eval for most distros  | O(N log N) tree-code, O(N) FMM mode       |
+-----------------------------------+-------------------------------------------+
| No extra storage beyond nodes     | Interaction lists + local expansions      |
| and multipole coefficients        | (CSR + coefficient arena)                 |
+-----------------------------------+-------------------------------------------+
| Simple, single-pass eval          | Two passes: build lists then evaluate     |
+-----------------------------------+-------------------------------------------+

Memory Layout
-------------

The tree is stored in a single caller-provided persistent work buffer
(sized by :c:func:`fmm_work_size`).  It is partitioned (in order) into:

- ``nodes`` — ``octree_node_t`` array
- ``particle_order`` — source-index permutation
- ``multipole_coeffs`` — flat arena for multipole coefficients
- ``topo_to_real`` — count-pass topology index to node index map
- ``mp_slices`` — per-node multipole slice pointers
- ``leaf_indices`` — ``leaf_id`` to node index reverse map
- ``local_coeffs`` — flat arena for local expansion coefficients
- ``local_slices`` — per-node local expansion slice pointers
- V-list CSR (offsets + indices)
- Near-field CSR (offsets + indices)

A separate transient scratch buffer (sized by :c:func:`fmm_scratch_size`)
holds build-time temporaries including the topology array and per-thread
multipole-build scratch.  The scratch is freed after insertion; the work
buffer persists as ``tree.buffer``.

API
---

.. c:function:: size_t fmm_scratch_size(unsigned n_sources, const fmm_settings_t *settings, unsigned n_threads)

   Total scratch buffer bytes needed for count + build.

.. c:function:: bool fmm_prepare_scratch(void *scratch_buffer, size_t scratch_size, unsigned n_sources, unsigned n_threads, const real3_t *sources_coords, const fmm_settings_t *settings, octree_count_t *out_count, octree_scratch_t *out_scratch)

   Partition the scratch buffer, zero the topology array, and run the
   count pass.  Returns the node counts and a fully partitioned scratch
   view ready for :c:func:`fmm_tree_insert`.

.. c:function:: size_t fmm_work_size(unsigned n_sources, const fmm_settings_t *settings, const octree_count_t *count)

   Total work buffer bytes needed given a finished count pass.

.. c:function:: bool fmm_tree_insert(unsigned n_sources, unsigned n_threads, const real3_t *sources_coords, const real3_t *sources_values, const fmm_settings_t *settings, const octree_count_t *count, const octree_scratch_t *scratch, const allocator_t *allocator, void *buffer, size_t buffer_size, fmm_tree_t *out)

   Run the full build pipeline into a pre-sized work buffer.  The
   ``count`` and ``scratch`` must come from a prior call to
   :c:func:`fmm_prepare_scratch`.

.. c:function:: bool fmm_tree_build(unsigned n_sources, unsigned n_threads, const real3_t *sources_coords, const real3_t *sources_values, const fmm_settings_t *settings, const allocator_t *allocator, fmm_tree_t *out)

   Convenience wrapper that runs all seven stages internally, allocating
   both scratch and work buffers through ``allocator``.

.. c:function:: unsigned fmm_tree_n_nodes(const fmm_tree_t *tree)

   Total number of nodes.

.. c:function:: size_t fmm_tree_memory_bytes(const fmm_tree_t *tree)

   Total buffer size.

.. c:function:: real3_t fmm_tree_eval(const fmm_tree_t *tree, const real3_t *sources_coords, const real3_t *sources_values, real3_t point, fmm_eval_settings_t eval_settings)

   Evaluate at a single target point.  Uses precomputed V-list (tree-code
   mode) or local expansion (FMM mode) for the far-field, plus direct
   particle sum over the near-field.

.. c:function:: void fmm_tree_eval_all(const fmm_tree_t *tree, const real3_t *sources_coords, const real3_t *sources_values, unsigned n_targets, const real3_t *targets, real3_t *results, fmm_eval_settings_t eval_settings, unsigned n_threads)

   Batched evaluation (OpenMP parallel over targets).
