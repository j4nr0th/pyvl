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

The first 8 build stages are shared with the :ref:`octree foundation
<pyvl.private_c.octree>` — no layout descriptor or wrapper functions
are involved; the shared pipeline operates on ``octree_node_t *``
directly.

1. **Count pass** → :c:func:`octree_count_pass`
2. **Materialise** → :c:func:`octree_materialize`
3. **Descend** → :c:func:`octree_descend`
4. **Metadata** → :c:func:`octree_compute_metadata`
5. **Fill** → :c:func:`octree_fill_particle_order`
6. **Centroids** → :c:func:`octree_compute_leaf_centers`
7. **P2M** → :c:func:`octree_build_leaf_multipoles`
8. **M2M** → :c:func:`octree_upward_sweep_level` (depth loop)

Then the FMM-specific stages:

9. **Interaction lists** — for each leaf, classifies every other leaf
   as V-list (well-separated: :math:`|\Delta\mathbf{c}| \ge 3\max(h_a, h_b)`)
   or near-field (everything else).  Stored as CSR arrays.

10. **M2L + L2L** (FMM mode only) — converts V-list multipoles to local
    expansions at each leaf, then propagates local expansions downward
    from parents to children.

Sizing
------

* Scratch buffer: use the canonical :c:func:`octree_scratch_size` /
  :c:func:`octree_size_scratch` / :c:func:`octree_total_scratch_size`.
  The old ``fmm_size_scratch``, ``fmm_scratch_size``, and
  ``fmm_buffer_size`` wrappers have been removed.

* Work buffer: the FMM tree adds extra regions (leaf indices, local
  coefficients, local slice pointers, interaction list CSR) beyond the
  common ``octree_base_work_sizes_t``.  Use the FMM-specific
  :c:func:`fmm_size_work_buffer` and :c:func:`fmm_total_work_size`.

* Count pass: :c:func:`octree_count_pass` (``fmm_count_pass`` removed).


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
| Per-target tree walk with MAC     | Precomputed V-list; one L2P or per-V-list |
|                                   | multipole_eval per target                 |
+-----------------------------------+-------------------------------------------+
| O(N log N) eval for most distros  | O(N log N) tree-code, O(N) FMM mode       |
+-----------------------------------+-------------------------------------------+
| No extra storage beyond nodes and  | Interaction lists + local expansions      |
| multipole coefficients            | (CSR + coefficient arena)                 |
+-----------------------------------+-------------------------------------------+
| Simple, single-pass eval          | Two passes: build lists then evaluate     |
+-----------------------------------+-------------------------------------------+

Memory Layout
-------------

The tree is stored in a single caller-provided persistent buffer
partitioned (in order) into:

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

A separate transient scratch buffer (sized by :c:func:`octree_scratch_size`)
holds build-time temporaries including the topology array and per-thread
multipole-build scratch.

Data Structures
---------------

.. c:autodoc:: fmm_tree.h

   The ``fmm_node_t``, ``fmm_node_kind_t``, and ``FMM_NODE_*`` constants
   have been **removed** — use ``octree_node_t`` and
   ``OCTREE_NODE_INTERNAL`` / ``OCTREE_NODE_PARTICLE`` /
   ``OCTREE_NODE_MULTIPOLE`` directly.
