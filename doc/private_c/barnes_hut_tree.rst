Barnes-Hut Tree
===============

The Barnes-Hut tree accelerates many-body vortex interaction evaluation
by hierarchically grouping sources into cubic cells and replacing dense
source clusters with far-field multipole expansions. The
:doc:`multipole <multipole>` module provides the leaf-level expansion;
this module provides the spatial partitioning that decides *when* to
use a multipole and *how* to walk the resulting tree.

.. contents:: Table of Contents
   :local:


Algorithm overview
------------------

The build proceeds in two passes:

1. **Count pass (sequential)** — sources are walked through a topologically
   mutable "topo" array, splitting cells on the fly whenever a leaf
   exceeds the multipole-compression threshold. This determines the
   final tree shape (which cells are internal, which carry multipoles,
   which are particle leaves) without yet allocating per-node
   multipole storage.

2. **Insert pass (parallel)** — given the topology from the count pass,
   the per-cell multipole coefficients are laid out in a single
   contiguous buffer, every source is descended through the tree and
   assigned to a leaf, and finally child multipoles are aggregated into
   their parents via :c:func:`multipole_add_shift` (an upward sweep).

Both passes operate on caller-provided buffers; the build never calls
``malloc`` beyond transient per-thread scratch inside the leaf build
loop.


Subdivision and multipole decisions
-----------------------------------

For each leaf :math:`\ell` with :math:`n_\ell` sources inside a cube of
side :math:`h_\ell` centered at :math:`c_\ell`:

* **Subdivide** if
  :math:`n_\ell > n_\text{crit} (P+1)^3`, where :math:`P` is the
  ``order`` setting and :math:`n_\text{crit}` is
  ``critical_particle_count``. This threshold equates the cost of
  building a multipole expansion of order :math:`P` at the parent
  (which has :math:`(P+1)^3` coefficients and the cost of evaluating
  it at a target) against the cost of evaluating the same child
  expansion at every target. Below this ratio a child multipole is
  cheaper than aggregating into a parent.

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


Tagged-union node layout
------------------------

Each node in the public :c:type:`bh_node_t` is a tagged union of two
shapes that share storage:

* **Internal** — eight child pointers (``children[8]``), centred on
  the cell, half-size, depth, particle count of zero.
* **Multipole leaf** — multipole ``coeffs_x/y/z`` pointers into the
  buffer, order, center, and particle range.

Because ``children[k]`` and ``coeffs_x/y/z`` overlap, an internal
node's multipole slice (used during the upward sweep) is stashed in
a parallel side table ``mp_slices[]`` indexed by node id, rather than
in the node struct itself. Multipole leaves have no children and so
use the slice in-place.

This means a call to :c:func:`barnes_hut_tree_insert` performs a
two-pass tree materialisation: first it assigns bh_node_t indices in
DFS pre-order and writes child pointers as topo-index sentinels,
then it walks the topo array a second time to resolve the
sentinels into real ``&nodes[child]`` pointers.


Buffer layout
-------------

:c:func:`barnes_hut_buffer_size` returns the total bytes needed for a
given ``n_sources`` and ``settings``. The insert function partitions
the caller-provided buffer into five contiguous regions:

1. ``bh_node_t nodes[n_total]`` — the materialized tree.
2. ``unsigned particle_order[n_sources]`` — source indices grouped by
   leaf; ``nodes[i].particle_order[k]`` for ``k`` in
   ``[particle_begin, particle_begin + particle_count)`` gives the
   sources of leaf ``i``.
3. ``real_t multipole_coeffs[3 * n_mp * n_coeffs]`` — multipole
   coefficient slices, three pointers per multipole-bearing node.
4. ``real_t scratch[multipole_scratch_size(work_order)]`` — transient
   scratch for the multipole builder.
5. ``real_t shift_exp[3 * (work_order+1)^2]``,
   ``real_t pse[2 * n_coeffs(work_order)]`` — work buffers for
   :c:func:`multipole_add_shift`.

A separate caller-provided **scratch buffer** holds the topology
array (used during the count pass), the per-source leaf indices,
and per-thread leaf multipole scratch. Its required size is
returned by :c:func:`barnes_hut_scratch_size`; the layouts of the
output buffer and the scratch buffer are deliberately distinct so
that the (much larger) persistent tree buffer and the
(input-sized, transient) scratch buffer can be allocated
independently — for example the scratch might live on the stack,
in arena-reserved storage, or be reused across multiple trees.


Parallelism
-----------

The count pass is inherently sequential (each source's walk mutates the
topo array). The insert pass is parallelised in three phases:

* **Leaf-count descent** (``descend_for_each_source``) — sources are
  walked independently. Per-leaf counter increments are atomic.
* **Per-leaf multipole build** — each thread allocates its own
  ``leaf_cur``, ``leaf_nxt``, ``leaf_coords``, ``leaf_values`` scratch
  and processes a disjoint subset of leaves. The OMP work-sharing
  uses a dynamic schedule because leaf build cost varies widely
  with the number of sources per leaf.
* **Upward sweep** — within a given depth level, sibling internal
  nodes are independent and can be processed in parallel. Different
  depth levels are processed sequentially (deepest first) because a
  parent at depth ``d`` reads the slices of its children at depth
  ``d+1``.

OMP thread count is taken from ``settings.n_threads``: a value of 0
usespersistent tree buffer contains nodes, particle-order, multipole
coefficients, and upward-sweep scratch. Transient scratch is held in
the caller-provided ``scratch_buffer`` argument of every public entry
point, sized by :c:func:`barnes_hut_scratch_size`. The two are
deliberately separated so the persistent buffer can be retained for
walk-time use while the scratch is heap-, arena-, or stack-allocated
and freed on return.

The remaining allocations — specifically the small, count-pass-output-
sized residual buffers (``topo_to_real`` index, ``mp_slices`` side
table) — are routed through the ``allocator`` argument
(``const allocator_t *``) of every public entry point. Passing
``NULL`` selects a libc-backed default; callers with arena, pool, or
instrumentation needs can plug in their own callbacks.

The per-thread multipole scratch uses the **worst-case** bound
(``n_sources * 3 * sizeof(real_t)`` for ``leaf_coords`` /
``leaf_values``) allocated once per thread and reused, so there is
no per-leaf allocation. Callers that need tighter bounds can
expose their own allocator that grows or pools differently.


API
---

.. c:type:: bh_node_kind_t

   Discriminator for a Barnes-Hut node.

.. c:type:: barnes_hut_settings_t

   User-tunable hyperparameters.

.. c:type:: bh_node_t

   Single node of the Barnes-Hut tree.

.. c:type:: barnes_hut_tree_t

   Caller-visible tree handle.

.. c:function:: size_t barnes_hut_buffer_size(unsigned n_sources, const barnes_hut_settings_t *settings)

   Return the total bytes required to hold a tree.

   :param n_sources: Number of source points.
   :param settings: Build settings.
   :return: Required buffer size in bytes, or 0 on invalid input.

.. c:function:: size_t barnes_hut_scratch_size(unsigned n_sources, unsigned n_threads, const barnes_hut_settings_t *settings)

   Return the bytes needed for the transient scratch buffer.

   :param n_sources: Number of source points.
   :param n_threads: Number of OpenMP threads.
   :param settings: Build settings.
   :return: Required scratch size in bytes, or 0 on invalid input.

.. c:function:: unsigned barnes_hut_tree_n_nodes(const barnes_hut_tree_t *tree)

   Return the total number of nodes in the tree.

   :param tree: Tree handle.
   :return: Node count.

.. c:function:: void barnes_hut_tree_depth_stats(const barnes_hut_tree_t *tree, unsigned *min_depth, unsigned *max_depth)

   Fill the depth statistics for the tree.

   :param tree: Tree handle.
   :param min_depth: Output for minimum depth.
   :param max_depth: Output for maximum depth.

.. c:function:: size_t barnes_hut_tree_memory_bytes(const barnes_hut_tree_t *tree)

   Return the total bytes occupied by the tree in its buffer.

   :param tree: Tree handle.
   :return: Total buffer bytes.

.. c:function:: bool barnes_hut_tree_insert(unsigned n_sources, unsigned n_threads, const real3_t *sources_coords, const real3_t *sources_values, const barnes_hut_settings_t *settings, void *scratch_buffer, size_t scratch_size, const allocator_t *allocator, void *buffer, size_t buffer_size, barnes_hut_tree_t *out)

   Build the tree in-place in a caller-provided buffer.

   :param n_sources: Number of source points.
   :param n_threads: Number of OpenMP threads (>= 1).
   :param sources_coords: Coordinates of the source points.
   :param sources_values: Vector source strengths.
   :param settings: Build settings.
   :param scratch_buffer: Transient scratch buffer.
   :param scratch_size: Size of scratch_buffer.
   :param allocator: Allocator callbacks (NULL for libc default).
   :param buffer: Persistent storage for the output tree.
   :param buffer_size: Size of buffer.
   :param out: Out-parameter for the populated tree handle.
   :return: true on success.

.. c:function:: bool barnes_hut_tree_build(unsigned n_sources, unsigned n_threads, const real3_t *sources_coords, const real3_t *sources_values, const barnes_hut_settings_t *settings, const allocator_t *allocator, barnes_hut_tree_t *out)

   Convenience wrapper: run count + insert in one call, allocating buffers internally.

   :param n_sources: Number of source points.
   :param n_threads: Number of OpenMP threads (>= 1).
   :param sources_coords: Coordinates of the source points.
   :param sources_values: Vector source strengths.
   :param settings: Build settings.
   :param allocator: Allocator callbacks (NULL for libc default).
   :param out: Out-parameter for the populated tree handle.
   :return: true on success.

.. c:function:: real3_t barnes_hut_tree_eval(const barnes_hut_tree_t *tree, const real3_t *sources_coords, const real3_t *sources_values, real3_t point, barnes_hut_eval_settings_t eval_settings)

   Evaluate the tree at a single target point.

   :param tree: Built tree handle.
   :param sources_coords: Source coordinates.
   :param sources_values: Source strengths.
   :param point: Target evaluation point.
   :param eval_settings: Multipole acceptance settings.
   :return: Induced velocity at the point.

.. c:function:: void barnes_hut_tree_eval_all(const barnes_hut_tree_t *tree, const real3_t *sources_coords, const real3_t *sources_values, unsigned n_targets, const real3_t *targets, real3_t *results, barnes_hut_eval_settings_t eval_settings, unsigned n_threads)

   Evaluate the tree at multiple target points (batched, OpenMP).

   :param tree: Built tree handle.
   :param sources_coords: Source coordinates.
   :param sources_values: Source strengths.
   :param n_targets: Number of target points.
   :param targets: Array of target points.
   :param results: Output array for induced velocities.
   :param eval_settings: Multipole acceptance settings.
   :param n_threads: OpenMP thread count.
