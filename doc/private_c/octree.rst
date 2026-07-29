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
   child pointers.  Sets ``geom_center`` (invariant geometric cell centre,
   never changed) and ``center`` (may be updated to Γ-weighted centroid).

3. **Descend** — assigns each source to its leaf node (OpenMP parallel
   with atomic leaf-count increments).

4. **Metadata** — computes ``particle_begin`` prefix sums, resets
   ``particle_count``, assigns unique ``leaf_id`` to each leaf.
   Parallelised (per-thread depth ranges, atomic-capture thread IDs).

5. **Fill** — populates ``particle_order[]`` so each leaf's sources are
   contiguous (OpenMP parallel, atomic capture).

6. **Centroids** — replaces leaf geometric centres with
   :math:`|\Gamma|`-weighted centroids.

7. **P2M** — builds multipole expansions for each multipole-bearing leaf
   via :c:func:`multipole_create`.

8. **M2M** — upward sweep: aggregates child multipoles into each internal
   node via :c:func:`multipole_add_shift`.  Parallelised per depth level
   (atomic-capture thread IDs, per-thread shift_exp/pse scratch).

The FMM tree adds several extra stages (Morton sort, interaction lists,
M2L, L2L) on top of this shared base — see :ref:`fmm_tree <pyvl.private_c.fmm_tree>`.

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
        real3_t geom_center; /**< Geometric cell center (set during materialize, never changed). */
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

The BH and FMM tree modules provide **staged build wrappers** that
encapsulate the two-pass protocol and expose the sizing and count
functions :ref:`barnes_hut_tree <pyvl.private_c.barnes_hut_tree>` and
:ref:`fmm_tree <pyvl.private_c.fmm_tree>`.


Buffer-passing API
------------------

All functions use caller-owned buffers with explicit sizes; nothing is
allocated internally.  The two-tier buffer strategy:

* **Scratch buffer** (transient) — sized from input parameters alone
  via :c:func:`octree_scratch_size`.  Holds the topology array,
  per-source leaf maps, per-thread multipole-build scratch, and the
  radix-sort histogram (``radix_hist``, n_threads × 256 unsigned).
* **Work buffer** (persistent) — sized after the count pass via
  :c:func:`octree_size_work_buffer`.  Holds the materialised node array,
  coefficient storage, and (for FMM) interaction lists, local
  expansions, Morton sort arrays, and parent index arrays.

The BH and FMM tree modules provide **staged build wrappers** that
encapsulate the two-pass protocol — see
:ref:`barnes_hut_tree <pyvl.private_c.barnes_hut_tree>` and
:ref:`fmm_tree <pyvl.private_c.fmm_tree>` for details.

Typical usage with the staged wrappers::

    // 1. Size scratch
    size_t scratch_sz = method_scratch_size(n, settings, n_threads);
    // 2. Allocate scratch
    void *scratch = malloc(scratch_sz);
    // 3. Prepare scratch + count pass
    octree_count_t count;
    octree_scratch_t sview;
    method_prepare_scratch(scratch, scratch_sz, n, n_threads, coords,
                           settings, &count, &sview);
    // 4. Size work buffer
    size_t work_sz = method_work_size(n, settings, &count);
    // 5. Allocate work buffer
    void *work = malloc(work_sz);
    // 6. Full pipeline
    method_tree_t tree;
    method_tree_insert(n, n_threads, coords, values, settings,
                       &count, &sview, allocator, work, work_sz, &tree);
    // 7. Free scratch, keep work buffer
    free(scratch);


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


Morton 3D helpers (static inline)
---------------------------------

.. c:function:: uint64_t morton_split_21(uint64_t x)

   Split 21 low bits by interleaving with zeros (3D Morton code helper).
   Bit-parallel spread via successive shift-and-mask stages.

.. c:function:: uint64_t morton_3d(real3_t p, real3_t root_center, real_t root_half_size)

   Compute 63-bit Morton code for a point within the root cell.
   Uses 21 bits per coordinate (2^21 grid per axis).
   Clamps input to [0, 1) range.

.. c:function:: void morton_range_bounds(const unsigned *sorted_indices, const uint64_t *codes, size_t offset, size_t count, uint64_t code_min, uint64_t code_max, size_t *out_begin, size_t *out_end)

   Binary-search the Morton-sorted index array for a code range.
   Returns the half-open interval [out_begin, out_end) of indices
   whose Morton codes fall within [code_min, code_max].

.. c:function:: bool octree_build_morton_sorted(unsigned n_nodes, const octree_node_t *nodes, uint64_t *codes, unsigned *sorted_indices, unsigned *depth_offsets, uint8_t *pairs_temp, unsigned *radix_hist, unsigned max_depth, unsigned *out_max_depth_found, unsigned n_threads)

   Build per-depth Morton-sorted index arrays using parallel LSD radix sort.
   Uses ``geom_center`` (invariant) for stable spatial ordering.

   **Algorithm:**
   1. Compute 63-bit Morton codes for all nodes (parallel).
   2. Count nodes per depth using per-thread histograms in ``radix_hist``.
   3. Prefix-sum to build ``depth_offsets``.
   4. Scatter node indices into per-depth buckets (parallel).
   5. For each depth level with >1 node, run parallel LSD radix sort
      (8 passes, 8 bits/pass, histogram + scatter per pass).

   **Output:**
   - ``depth_offsets[d]`` = start offset for depth d (0..max_depth_found).
   - ``depth_offsets[max_depth_found+1]`` = total count (end sentinel).
   - ``sorted_indices`` = node indices sorted by Morton code within each depth.

   **Pre-conditions:**
   - ``radix_hist`` must be [n_threads * 256] from scratch buffer.
   - ``pairs_temp`` must be [2 * n_nodes * (sizeof(uint64_t) + sizeof(unsigned))].

   **Post-conditions:**
   - ``*out_max_depth_found`` = maximum occupied depth.
   - Returns false if n_nodes == 0.


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
| FMM-specific stages | --                                            | Interaction lists, M2L                     |
+---------------------+-----------------------------------------------+--------------------------------------------+
| Evaluation          | Tree walk with MAC (opening-angle or          | Tree-code mode per V-list, or              |
|                     | neighbour criterion)                          | FMM mode (local expansions)                |
+---------------------+-----------------------------------------------+--------------------------------------------+
| Sizing API          | ``barnes_hut_*`` staged wrappers              | ``fmm_*`` staged wrappers                  |
+---------------------+-----------------------------------------------+--------------------------------------------+
