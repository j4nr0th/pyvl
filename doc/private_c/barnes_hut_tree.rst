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

The build is exposed through a **staged API** (steps 1-7 below).  The
stages mirror the underlying :doc:`octree <octree>` pipeline while letting
the caller control allocation:

1. **Size scratch** — :c:func:`barnes_hut_scratch_size` returns the
   transient scratch buffer size from input parameters alone.
2. **Allocate scratch** — caller provides a buffer of that size.
3. **Prepare scratch** — :c:func:`barnes_hut_prepare_scratch` partitions
   the scratch buffer, zeros the topology array, runs the **count pass**
   (:c:func:`octree_count_pass`), and returns the
   :c:type:`octree_count_t` and :c:type:`octree_scratch_t` views.
4. **Size work** — :c:func:`barnes_hut_work_size` returns the persistent
   work buffer size from the count-pass results.
5. **Allocate work** — caller provides the work buffer.
6. **Insert pass** — :c:func:`barnes_hut_tree_insert` runs the full
   shared pipeline (:c:func:`octree_materialize` →
   :c:func:`octree_descend` → :c:func:`octree_compute_metadata` →
   :c:func:`octree_fill_particle_order` →
   :c:func:`octree_compute_leaf_centers` →
   :c:func:`octree_build_leaf_multipoles` → :c:func:`octree_upward_sweep_level`
   (M2M, depth loop)) and fills the output :c:type:`barnes_hut_tree_t`.
7. **Release scratch** — the caller may free the scratch buffer; the work
   buffer remains owned by the tree handle.

:c:func:`barnes_hut_tree_build` is a convenience wrapper that allocates
both buffers internally (through the supplied ``allocator``) and runs
all seven stages.  The build never calls ``malloc`` directly; all
allocation goes through the custom allocator.

Typical staged usage::

    // 1. Size scratch
    size_t scratch_sz = barnes_hut_scratch_size(n, settings, n_threads);
    // 2. Allocate scratch
    void *scratch = malloc(scratch_sz);
    // 3. Prepare scratch + count
    octree_count_t count;
    octree_scratch_t sview;
    barnes_hut_prepare_scratch(scratch, scratch_sz, n, n_threads,
                               coords, settings, &count, &sview);
    // 4. Size work
    size_t work_sz = barnes_hut_work_size(n, settings, &count);
    // 5. Allocate work
    void *work = malloc(work_sz);
    // 6. Full pipeline
    barnes_hut_tree_t tree;
    barnes_hut_tree_insert(n, n_threads, coords, values, settings,
                           &count, &sview, allocator, work, work_sz, &tree);
    // 7. Free scratch, keep work buffer (owned by tree)
    free(scratch);


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

The insert function partitions the caller-provided work buffer into five
contiguous regions (sized by :c:func:`barnes_hut_work_size` / :c:func:`octree_size_work_buffer`):

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

.. c:function:: size_t barnes_hut_scratch_size(unsigned n_sources, const barnes_hut_settings_t *settings, unsigned n_threads)

   Total scratch buffer bytes needed for count + build.  Delegates to
   :c:func:`octree_scratch_size`.

.. c:function:: bool barnes_hut_prepare_scratch(void *scratch_buffer, size_t scratch_size, unsigned n_sources, unsigned n_threads, const real3_t *sources_coords, const barnes_hut_settings_t *settings, octree_count_t *out_count, octree_scratch_t *out_scratch)

   Partition the scratch buffer, zero the topology array, and run the
   count pass (:c:func:`octree_count_pass`).  On return ``out_count``
   holds the node counts and ``out_scratch`` is a fully partitioned
   scratch view ready for :c:func:`barnes_hut_tree_insert`.

   Returns ``false`` if inputs are invalid or scratch is too small.

.. c:function:: size_t barnes_hut_work_size(unsigned n_sources, const barnes_hut_settings_t *settings, const octree_count_t *count)

   Total work buffer bytes needed given a finished count pass.
   Delegates to :c:func:`octree_size_work_buffer` and
   :c:func:`octree_total_work_size`.  Returns 0 if the count is
   empty or NULL.

.. c:function:: bool barnes_hut_tree_insert(unsigned n_sources, unsigned n_threads, const real3_t *sources_coords, const real3_t *sources_values, const barnes_hut_settings_t *settings, const octree_count_t *count, const octree_scratch_t *scratch, const allocator_t *allocator, void *buffer, size_t buffer_size, barnes_hut_tree_t *out)

   Run the full build pipeline into a pre-sized work buffer.  The
   ``count`` and ``scratch`` must come from a prior call to
   :c:func:`barnes_hut_prepare_scratch`.  The work buffer must be at
   least :c:func:`barnes_hut_work_size` bytes.

.. c:function:: bool barnes_hut_tree_build(unsigned n_sources, unsigned n_threads, const real3_t *sources_coords, const real3_t *sources_values, const barnes_hut_settings_t *settings, const allocator_t *allocator, barnes_hut_tree_t *out)

   Convenience wrapper: runs all seven stages internally, allocating
   both scratch and work buffers through ``allocator``.

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
