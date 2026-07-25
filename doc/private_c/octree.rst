.. _pyvl.private_c.octree:

Shared Octree Foundation
========================

The octree module (``src/core/octree.h``, ``src/core/octree.c``) provides
the canonical data structures, sizing API, and build pipeline used by both
the :ref:`Barnes-Hut tree <pyvl.private_c.barnes_hut_tree>` and the
:ref:`FMM tree <pyvl.private_c.fmm_tree>`.  It implements the adaptive
octree topology and multipole-construction stages once, eliminating
~1500 lines of near-identical code duplication.

.. contents:: Table of Contents
   :local:


Design
------

Both tree types share the same build stages and operate on a single node
type :c:type:`octree_node_t`.  The shared pipeline accesses node fields
directly — no layout descriptor indirection is needed.

The build stages are:

1. **Count pass** — walks source coordinates, builds a transient topology
   array, and returns exact node counts (internal, multipole, particle).

2. **Materialise** — walks the topology in DFS pre-order, creating node
   entries, assigning coefficient slices from a flat arena, and resolving
   child pointers.

3. **Descend** — assigns each source to its leaf node (OpenMP parallel
   with atomic leaf-count increments).

4. **Metadata** — computes ``particle_begin`` prefix sums, resets
   ``particle_count``, assigns unique ``leaf_id`` to each leaf.

5. **Fill** — populates ``particle_order[]`` so each leaf's sources are
   contiguous (OpenMP parallel, atomic capture).

6. **Centroids** — replaces leaf geometric centres with
   :math:`|\Gamma|`-weighted centroids.

7. **P2M** — builds multipole expansions for each multipole-bearing leaf
   via :c:func:`multipole_create`.

8. **M2M** — upward sweep: aggregates child multipoles into each internal
   node via :c:func:`multipole_add_shift`.

The FMM tree adds three extra stages (interaction lists, M2L, L2L) on
top of this shared base.

The unified node type
---------------------

.. code-block:: c

    typedef enum {
        OCTREE_NODE_INTERNAL = 0,
        OCTREE_NODE_PARTICLE = 1,
        OCTREE_NODE_MULTIPOLE = 2,
    } octree_node_kind_t;

    typedef struct octree_node {
        octree_node_kind_t kind;
        unsigned depth;
        real3_t center;
        real_t half_size;
        unsigned particle_begin;
        unsigned particle_count;
        int32_t leaf_id;
        union {
            struct { struct octree_node *children[8]; } internal;
            multipole_t mp;
        } data;
    } octree_node_t;

The old separate ``bh_node_t`` / ``fmm_node_t`` types and their
``BH_NODE_*`` / ``FMM_NODE_*`` discriminator constants have been
removed — use ``octree_node_t`` and ``OCTREE_NODE_*`` directly
everywhere.

The old ``barnes_hut_*`` / ``fmm_*`` wrapper functions for sizing
and counting have been removed — use the ``octree_*`` functions
below directly.


Buffer-passing API
------------------

All functions use caller-owned buffers with explicit sizes; nothing is
allocated internally.  The two-tier buffer strategy:

* **Scratch buffer** (transient) — sized from input parameters alone
  via :c:func:`octree_scratch_size`.  Holds the topology array,
  per-source leaf maps, and per-thread multipole-build scratch.
* **Work buffer** (persistent) — sized after the count pass via
  :c:func:`octree_size_work_buffer`.  Holds the materialised node array,
  coefficient storage, and (for FMM) interaction lists and local
  expansions.

Typical usage (two-pass build)::

    // 1. Count pass — learn topology size
    octree_scratch_t scratch = octree_scratch_partition(n_threads, scratch_buf,
                                octree_size_scratch(n_sources, settings));
    memset(scratch.topo, 0, (8*n_sources+1) * sizeof(topo_node_t));
    octree_count_t count = octree_count_pass(n_sources, coords, settings,
                             scratch.topo, scratch.source_leaf_topo);

    // 2. Size, allocate, and insert
    octree_base_work_sizes_t ws = octree_size_work_buffer(n_sources, settings, count);
    void *work = malloc(octree_total_work_size(ws));
    octree_materialize(scratch.topo, ..., work, ...);
    octree_descend(...); octree_compute_metadata(...);
    octree_fill_particle_order(...); octree_compute_leaf_centers(...);
    octree_build_leaf_multipoles(...);
    // ... upward sweep, then BH/FMM-specific stages


Sizing API (canonical)
----------------------

.. c:function:: octree_scratch_sizes_t octree_size_scratch(unsigned n_sources, const octree_settings_t *settings)

   Compute per-region scratch sizes from input parameters alone.

.. c:function:: size_t octree_total_scratch_size(octree_scratch_sizes_t sizes, unsigned n_threads)

   Total scratch buffer size from per-region sizes and thread count.

.. c:function:: size_t octree_scratch_size(unsigned n_sources, unsigned n_threads, const octree_settings_t *settings)

   Convenience wrapper around both of the above.

.. c:function:: octree_scratch_t octree_scratch_partition(unsigned n_thread_partitions, void *buffer, octree_scratch_sizes_t sizes)

   Carve the pre-allocated scratch buffer into the :c:type:`octree_scratch_t` view.

.. c:function:: octree_base_work_sizes_t octree_size_work_buffer(unsigned n_sources, const octree_settings_t *settings, octree_count_t count)

   Work buffer sizes from count-pass results (common BH/FMM regions).

.. c:function:: size_t octree_total_work_size(octree_base_work_sizes_t sizes)

   Total work buffer size in bytes.

.. c:function:: size_t octree_buffer_size(unsigned n_sources, const octree_settings_t *settings)

   Pessimistic upper bound on buffer size (useful when the user wants
   a single pre-allocated buffer without running the count pass first).

.. c:function:: octree_count_t octree_count_pass(unsigned n_sources, const real3_t *sources_coords, const octree_settings_t *settings, topo_node_t *topo, uint32_t *source_leaf)

   Sequential count pass: walk sources, split cells as needed, return
   node counts.

.. c:autodoc:: octree.h

   The ``octree_node_layout_t`` type, ``OCTREE_NO_LEAF_ID`` sentinel,
   and all ``octree_r_*`` / ``octree_w_*`` accessor macros have been
   **removed** from the header.  The pipeline now operates on
   ``octree_node_t *`` with direct field access — no indirection.


Relationship to BH and FMM trees
---------------------------------

+---------------------+-----------------------------------------------+--------------------------------------------+
| Aspect              | Barnes-Hut                                    | FMM                                        |
+=====================+===============================================+============================================+
| Node type           | ``octree_node_t``                             | ``octree_node_t``                          |
+---------------------+-----------------------------------------------+--------------------------------------------+
| Shared stages       | Count, materialise, descend, metadata, fill,  | (same 8 shared stages)                     |
|                     | centroids, P2M, M2M                           |                                            |
+---------------------+-----------------------------------------------+--------------------------------------------+
| FMM-specific stages | —                                             | Interaction lists, M2L, L2L                |
+---------------------+-----------------------------------------------+--------------------------------------------+
| Evaluation          | Tree walk with MAC (opening-angle or neighbour)| Tree-code mode per V-list, or FMM mode     |
|                     | criterion)                                    | (precomputed local expansions)             |
+---------------------+-----------------------------------------------+--------------------------------------------+
| Sizing API          | ``octree_*`` (`barnes_hut_*` wrappers removed)| ``octree_*`` for scratch; ``fmm_size_work_buffer`` for FMM-specific work layout |
+---------------------+-----------------------------------------------+--------------------------------------------+
