.. _pyvl.private_c.barnes_hut_tree:

Barnes-Hut Tree
===============

The Barnes-Hut tree accelerates many-body vortex interaction evaluation
by hierarchically grouping sources into cubic cells and replacing dense
source clusters with far-field multipole expansions. The
:doc:`multipole <multipole>` module provides the leaf-level expansion;
this module provides the spatial partitioning and tree-walk evaluation.

The tree **build pipeline** (count pass, materialize, descend, metadata,
fill, centroids, P2M, M2M) is shared with the :doc:`FMM tree <fmm_tree>`
via the :doc:`shared octree foundation <octree>`.  Only the **evaluation**
strategy is unique to the Barnes-Hut tree.

.. contents:: Table of Contents
   :local:


Build pipeline
--------------

The build stages are implemented in :doc:`octree <octree>`.  The pipeline
runs in two passes:

1. **Count pass** (:c:func:`octree_count_pass`) — sources are walked
   through a topologically mutable "topo" array, splitting cells on the
   fly whenever a leaf exceeds the multipole-compression threshold.
   This determines the final tree shape without allocating per-node
   multipole storage.

2. **Insert pass** — runs the shared pipeline:
   :c:func:`octree_materialize` → :c:func:`octree_descend` →
   :c:func:`octree_compute_metadata` → :c:func:`octree_fill_particle_order`
   → :c:func:`octree_compute_leaf_centers` →
   :c:func:`octree_build_leaf_multipoles` → :c:func:`octree_upward_sweep_level`
   (M2M, depth loop).

Both passes operate on caller-provided buffers; the build never calls
``malloc`` beyond the optional allocator fallback in
:c:func:`barnes_hut_tree_build`.

All sizing and counting is done through the canonical :doc:`octree <octree>`
API (:c:func:`octree_size_scratch`, :c:func:`octree_size_work_buffer`,
:c:func:`octree_count_pass`).  The old ``barnes_hut_size_scratch``,
``barnes_hut_count_pass``, ``barnes_hut_buffer_size``, etc., wrappers
have been removed — use the ``octree_*`` equivalents directly.


Subdivision and multipole decisions
-----------------------------------

For each leaf :math:`\ell` with :math:`n_\ell` sources inside a cube of
side :math:`h_\ell` centered at :math:`c_\ell`:

* **Subdivide** if
  :math:`n_\ell > n_\text{crit} \cdot 8`, where :math:`n_\text{crit}` is
  ``critical_particle_count``. This gives 8 children, each receiving
  roughly :math:`n_\text{crit}` particles on average.

* **Become a multipole leaf** if
  :math:`n_\ell \geq n_\text{crit}` but subdivision is not worth it
  yet. The expansion center is the
  :math:`|\Gamma|`-weighted centroid of the source positions (so the
  multipole is centered on the vortex-strength barycentre, not the
  geometric centroid).

* **Become a particle leaf** otherwise — sources are evaluated
  directly.

Empty subtrees are pruned: a cell with no sources and no children is
dropped. This avoids allocating empty internal nodes at the leaves of
the spatial partition.

An additional **centroid subdivision** criterion can be enabled by
setting ``alpha_centroid > 0.0``: if any source is farther than
``alpha_centroid * half_size`` from the cell centre, the cell is
subdivided. This guarantees sources are tightly clustered around the
leaf centre, improving multipole convergence for non-uniform
distributions.


Tagged-union node layout
------------------------

Each node is an :c:type:`octree_node_t` (the node type is shared with the
FMM tree — there is no separate BH node type).  The ``data`` union stores
either eight child pointers (for internal nodes) or a :c:type:`multipole_t`
(for multipole leaves).  Particle leaves use the union for nothing.

Because ``children[k]`` and ``coeffs_x/y/z`` overlap, an internal
node's multipole slice (used during the upward sweep) is stashed in
a parallel side table ``mp_slices[]`` indexed by node id, rather than
in the node struct itself. Multipole leaves have no children and so
use the slice in-place.


Buffer layout
-------------

The insert function partitions the caller-provided buffer into five
contiguous regions (sized by :c:func:`octree_size_work_buffer`):

1. ``octree_node_t nodes[n_total]`` — the materialized tree.
2. ``unsigned particle_order[n_sources]`` — source indices grouped by
   leaf; ``nodes[i].particle_order[k]`` for ``k`` in
   ``[particle_begin, particle_begin + particle_count)`` gives the
   sources of leaf ``i``.
3. ``real_t multipole_coeffs[...]`` — multipole coefficient slices,
   three pointers per multipole-bearing node.
4. ``uint32_t topo_to_real[n_total]`` — count-pass topology index to
   node index mapping.
5. ``real_t *mp_slices[n_total]`` — per-node coefficient slice
   pointers (side table for internal nodes).

A separate caller-provided **scratch buffer** (sized by
:c:func:`octree_scratch_size`) holds the topology array, per-source
leaf indices, and per-thread leaf multipole scratch. The two buffers
are deliberately distinct so the persistent tree buffer can be retained
while the scratch is heap-, arena-, or stack-allocated and freed on
return.


Evaluation
----------

The evaluation uses a tree walk with an iterative stack (no recursion).
At each node the **Multipole Acceptance Criterion** decides whether to
accept the node's aggregated multipole or descend into its children.

Two MAC modes are available (controlled by ``eval_settings.theta``):

* **Neighbour criterion** (:math:`\theta \le 0`, default): accept if
  the target is outside the cell's :math:`3 \times 3 \times 3`
  neighbourhood — :math:`|\Delta x| > 2h \lor |\Delta y| > 2h \lor
  |\Delta z| > 2h`.  Safe for all distributions but slower.

* **Opening-angle criterion** (:math:`\theta > 0`): accept if
  :math:`h / |\Delta r| < \theta`.  Recommended values:
  ``0.3`` (fast, similar accuracy to neighbour), ``0.1`` (moderate),
  ``0.01`` (high mid-field accuracy, slow).

Evaluation is available as single-target (:c:func:`barnes_hut_tree_eval`)
or batched (:c:func:`barnes_hut_tree_eval_all`, OpenMP parallel over
targets).


API
---

.. c:type:: barnes_hut_tree_t

   Caller-visible tree handle.  Uses ``octree_node_t *nodes`` directly
   (no separate BH node type).

.. c:type:: barnes_hut_eval_settings_t

   Evaluation settings. Use ``BARNES_HUT_EVAL_SETTINGS_DEFAULT`` for
   the default configuration (neighbour criterion).

.. c:type:: barnes_hut_work_t

   Partitioned work buffer views (``octree_node_t *nodes``,
   ``unsigned *particle_order``, ``real_t *multipole_coeffs``,
   ``uint32_t *topo_to_real``, ``real_t **mp_slices``).

.. c:function:: bool barnes_hut_tree_insert(unsigned n_sources, unsigned n_threads, const real3_t *sources_coords, const real3_t *sources_values, const barnes_hut_settings_t *settings, void *scratch_buffer, size_t scratch_size, const allocator_t *allocator, void *buffer, size_t buffer_size, barnes_hut_tree_t *out)

   Run the insert pass into pre-allocated buffers.  The scratch buffer
   must be large enough for :c:func:`octree_scratch_size`; the work
   buffer for :c:func:`octree_total_work_size`.  The count pass must
   have been run externally.

.. c:function:: bool barnes_hut_tree_build(unsigned n_sources, unsigned n_threads, const real3_t *sources_coords, const real3_t *sources_values, const barnes_hut_settings_t *settings, const allocator_t *allocator, barnes_hut_tree_t *out)

   Convenience wrapper: runs count + insert in one call, allocating
   both buffers internally through ``allocator``.

.. c:function:: unsigned barnes_hut_tree_n_nodes(const barnes_hut_tree_t *tree)

   Total number of nodes.

.. c:function:: void barnes_hut_tree_depth_stats(const barnes_hut_tree_t *tree, unsigned *min_depth, unsigned *max_depth)

   Depth range observed in the tree.

.. c:function:: size_t barnes_hut_tree_memory_bytes(const barnes_hut_tree_t *tree)

   Total buffer size.

.. c:function:: real3_t barnes_hut_tree_eval(const barnes_hut_tree_t *tree, const real3_t *sources_coords, const real3_t *sources_values, real3_t point, barnes_hut_eval_settings_t eval_settings)

   Evaluate at a single target point.  Walks the tree with an
   iterative stack, deciding the MAC at each internal node.

.. c:function:: void barnes_hut_tree_eval_all(const barnes_hut_tree_t *tree, const real3_t *sources_coords, const real3_t *sources_values, unsigned n_targets, const real3_t *targets, real3_t *results, barnes_hut_eval_settings_t eval_settings, unsigned n_threads)

   Batched evaluation (OpenMP parallel over targets).
