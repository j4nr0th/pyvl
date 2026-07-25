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

The tree is built through a sequence of passes (mirroring
:ref:`barnes_hut_tree <pyvl.private_c.barnes_hut_tree>`):

1. **Count pass** — builds an octree topology from source coordinates
   and counts internal nodes, multipole leaves, and particle leaves.
   Reuses the same ``topo_node_t`` count logic as the Barnes-Hut tree.

2. **Materialise** — walks the topology array in DFS pre-order, creating
   ``fmm_node_t`` entries, handing out multipole coefficient slices from
   a flat arena, and resolving child pointers.

3. **Descend** — assigns each source particle to its leaf node (OpenMP
   parallel, atomic leaf-count increments).

4. **Metadata** — computes ``particle_begin`` prefix sums for the
   ``particle_order[]`` permutation and assigns each leaf a unique
   ``leaf_id``.

5. **Fill** — populates ``particle_order[]`` so each leaf's sources are
   contiguous (OpenMP parallel, atomic capture).

6. **Centroids** — replaces leaf geometric centres with :math:`|\Gamma|`-
   weighted centroids for better multipole convergence.

7. **P2M** — builds multipole expansions for each multipole-bearing leaf
   via ``multipole_create`` (OpenMP parallel with per-thread scratch).

8. **M2M** — upward sweep: aggregates child multipoles into each internal
   node via ``multipole_add_shift`` (OpenMP parallel per depth level).

9. **Interaction lists** — for each leaf, classifies every other leaf
   as V-list (well-separated: :math:`|\Delta\mathbf{c}| \ge 3\max(h_a, h_b)`)
   or near-field (everything else).  Stored as CSR arrays.

10. **M2L + L2L** (FMM mode only) — converts V-list multipoles to local
    expansions at each leaf, then propagates local expansions downward
    from parents to children.

Evaluation
----------

Two evaluation modes are available:

- **Tree-code mode** (``FMM_EVAL_TREE_CODE``): descends to the target's
  leaf, sums V-list multipoles via ``multipole_eval`` for the far-field,
  and performs a direct particle sum over the own leaf and near-field
  neighbours for the near-field.

- **FMM mode** (``FMM_EVAL_FMM``): evaluates the leaf's precomputed
  local expansion via ``local_expansion_eval`` for the far-field, plus
  the same near-field direct sum.  This is a single L2P call per target
  instead of one ``multipole_eval`` per V-list entry, giving O(N) scaling.

Memory Layout
-------------

The tree is stored in a single caller-provided persistent buffer
partitioned (in order) into:

- ``nodes`` — ``fmm_node_t`` array
- ``particle_order`` — source-index permutation
- ``multipole_coeffs`` — flat arena for multipole coefficients
- ``topo_to_real`` — count-pass topology index to node index map
- ``mp_slices`` — per-node multipole slice pointers
- ``leaf_indices`` — ``leaf_id`` to node index reverse map
- ``local_coeffs`` — flat arena for local expansion coefficients
- ``local_slices`` — per-node local expansion slice pointers
- V-list CSR (offsets + indices)
- Near-field CSR (offsets + indices)

A separate transient scratch buffer (sourced from input parameters alone)
holds build-time temporaries including the topology array and per-thread
multipole-build scratch.

Data Structures
---------------

.. c:autodoc:: fmm_tree.h
