#pragma once

/*
 * Fast Multipole Method tree over vortex-particle sources.
 *
 * This module implements an FMM that competes with the Barnes-Hut tree
 * (@ref barnes_hut_tree_t).  It reuses the same adaptive octree topology
 * (count pass, materialise, upward sweep) and the same multipole expansion
 * (@ref multipole_t) for the far-field.  The key addition over Barnes-Hut
 * is a set of *interaction lists* (V-list / near-field list) that allow
 * O(N log N) evaluation with better constants than the per-target tree
 * walk used by Barnes-Hut.
 *
 * Build pipeline (mirrors barnes_hut_tree):
 *   1. Count pass  — build topology, count nodes/leaves.
 *   2. Materialise — assign node indices, hand out coefficient slices.
 *   3. Descend     — assign sources to leaves.
 *   4. Metadata    — particle_begin prefix sums.
 *   5. Fill        — populate particle_order[].
 *   6. Centroids   — |Γ|-weighted leaf centres.
 *   7. P2M         — build leaf multipoles (multipole_create).
 *   8. M2M         — upward sweep (multipole_add_shift).
 *   9. Lists       — compute V-list and near-field interaction lists.
 *
 * Evaluation (tree-code mode, Phase 1):
 *   For each target, descend to its leaf.  The far-field contribution is
 *   the sum of V-list leaf multipoles evaluated via multipole_eval.  The
 *   near-field contribution is a direct particle sum over the target's
 *   leaf and its near-field neighbours.
 *
 * The tree is stored in a single caller-provided persistent buffer,
 * partitioned into nodes | particle_order | multipole_coeffs |
 * interaction lists.  A separate transient scratch buffer holds all
 * build-time temporaries.
 */

#include "common.h"
#include "multipole.h"

/* ------------------------------------------------------------------ */
/* Settings                                                           */
/* ------------------------------------------------------------------ */

/**
 * @brief Build settings for the FMM tree.
 *
 * Identical to @ref barnes_hut_settings_t — the FMM reuses the same
 * adaptive octree construction parameters.
 */
typedef struct
{
    unsigned order;                   /**< Multipole expansion order (>= 1). */
    unsigned critical_particle_count; /**< Minimum sources before a leaf becomes a multipole cell. */
    unsigned max_depth;               /**< Maximum octree depth (>= 1). */
    unsigned work_order;              /**< Internal expansion order for work buffers (0 = use order). */
    real_t alpha_centroid;            /**< Centroid-based subdivision threshold (0.0 = disabled). */
} fmm_settings_t;

/**
 * @brief Evaluation mode selector. */
typedef enum
{
    FMM_EVAL_TREE_CODE = 0, /**< Tree-code mode: per-V-list multipole_eval (Phase 1). */
    FMM_EVAL_FMM = 1,       /**< Full FMM mode: local expansions + near-field (Phase 2+). */
} fmm_eval_mode_t;

/**
 * @brief Settings for tree evaluation.
 *
 * @c theta controls the near-field / far-field split.  In tree-code mode
 * the V-list is always used for well-separated cells; @c theta is reserved
 * for future opening-angle refinements and currently unused.
 */
typedef struct
{
    double theta;         /**< Reserved for future MAC tuning (currently unused). */
    fmm_eval_mode_t mode; /**< Evaluation mode: tree-code or FMM. */
} fmm_eval_settings_t;

/** @brief Default eval settings. */
#define FMM_EVAL_SETTINGS_DEFAULT ((fmm_eval_settings_t){.theta = 0.0, .mode = FMM_EVAL_TREE_CODE})

/* ------------------------------------------------------------------ */
/* Node types                                                         */
/* ------------------------------------------------------------------ */

/**
 * @brief Discriminator for an FMM tree node.
 */
typedef enum
{
    FMM_NODE_INTERNAL = 0,  /**< Internal node with 8 children. */
    FMM_NODE_PARTICLE = 1,  /**< Leaf storing raw particle indices. */
    FMM_NODE_MULTIPOLE = 2, /**< Leaf storing a multipole expansion. */
} fmm_node_kind_t;

/**
 * @brief Single node of the FMM tree.
 *
 * Same layout as @ref bh_node_t — a tagged union where internal nodes
 * store child pointers and multipole leaves store coefficient pointers
 * inline.
 */
typedef struct fmm_node
{
    fmm_node_kind_t kind;    /**< Node type. */
    unsigned depth;          /**< Octree depth (root = 0). */
    real3_t center;          /**< Cell centre (|Γ|-weighted for leaves). */
    real_t half_size;        /**< Half-extent per axis. */
    unsigned particle_begin; /**< Index into particle_order[]. */
    unsigned particle_count; /**< Number of particles in this leaf. */
    int32_t leaf_id;         /**< Leaf index (0..n_leaves-1), -1 for internal nodes. */
    union {
        struct
        {
            struct fmm_node *children[8]; /**< Child node pointers (internal only). */
        } internal;
        multipole_t mp; /**< Multipole expansion (multipole leaves only). */
    } data;
} fmm_node_t;

/* ------------------------------------------------------------------ */
/* Count-pass and sizing types                                        */
/* ------------------------------------------------------------------ */

/** @brief Result of the count pass (topology sizes). */
typedef struct
{
    unsigned n_internal;         /**< Number of internal nodes. */
    unsigned n_multipole_leaves; /**< Number of multipole leaves. */
    unsigned n_particle_leaves;  /**< Number of particle leaves. */
    unsigned max_depth;          /**< Maximum depth reached. */
} fmm_count_res_t;

/** @brief Per-region scratch byte sizes (input-derived). */
typedef struct
{
    size_t size_topo;                   /**< Topology array bytes. */
    size_t size_source_leaf_topo;       /**< Source→leaf map bytes (topo indices). */
    size_t size_source_leaf_real;       /**< Source→leaf map bytes (real indices). */
    size_t size_leaf_buf_per_thread;    /**< leaf_cur + leaf_nxt per thread. */
    size_t size_leaf_coords_per_thread; /**< Source coords scratch per thread. */
    size_t size_leaf_values_per_thread; /**< Source values scratch per thread. */
    size_t size_shift_exp_per_thread;   /**< M2M shift_exp per thread. */
    size_t size_pse_per_thread;         /**< M2M pse per thread. */
} fmm_scratch_sizes_t;

/** @brief Per-region work-buffer byte sizes (count-derived). */
typedef struct
{
    size_t nodes_bytes;             /**< Bytes for the fmm_node_t array. */
    size_t particle_order_bytes;    /**< Bytes for the particle permutation. */
    size_t multipole_coeffs_bytes;  /**< Bytes for multipole coefficients. */
    size_t topo_to_real_bytes;      /**< Bytes for topo→real index map. */
    size_t mp_slices_bytes;         /**< Bytes for multipole slice pointers. */
    size_t leaf_indices_bytes;      /**< Bytes for leaf-index reverse map. */
    size_t local_coeffs_bytes;      /**< Bytes for local expansion coefficients. */
    size_t local_slices_bytes;      /**< Bytes for local expansion slice pointers. */
    size_t interaction_lists_bytes; /**< Bytes for V-list + near-field CSR. */
} fmm_work_sizes_t;

/* ------------------------------------------------------------------ */
/* Tree handle                                                        */
/* ------------------------------------------------------------------ */

/**
 * @brief FMM tree handle.  Holds views into the single caller-provided buffer.
 */
typedef struct
{
    fmm_settings_t settings; /**< Build settings. */
    real3_t root_center;     /**< Root cell centre. */
    real_t root_half_size;   /**< Root cell half-extent. */

    unsigned n_sources;          /**< Number of source particles. */
    unsigned n_nodes;            /**< Total node count. */
    unsigned n_internal;         /**< Internal node count. */
    unsigned n_multipole_leaves; /**< Multipole leaf count. */
    unsigned n_particle_leaves;  /**< Particle leaf count. */
    unsigned max_depth_reached;  /**< Maximum depth actually reached. */

    uint8_t *buffer;    /**< Owning buffer (NULL if externally managed). */
    size_t buffer_size; /**< Total buffer size in bytes. */

    fmm_node_t *nodes;        /**< Node array (view into buffer). */
    unsigned *particle_order; /**< Leaf→source index permutation. */
    real_t *multipole_coeffs; /**< Contiguous coefficient storage. */
    real_t **mp_slices;       /**< Per-node coefficient slice pointers. */

    /* Leaf index reverse map (leaf_id → node index). */
    unsigned *leaf_indices; /**< Leaf index to node index map [n_leaves]. */

    /* Local expansion storage (FMM mode). */
    real_t *local_coeffs;  /**< Contiguous local expansion coefficients. */
    real_t **local_slices; /**< Per-node local coefficient slice pointers. */

    /* Interaction lists (CSR format). */
    unsigned *vlist_offsets;  /**< V-list CSR offsets (n_leaves+1). */
    unsigned *vlist_indices;  /**< V-list flat source-leaf indices. */
    unsigned *nflist_offsets; /**< Near-field CSR offsets (n_leaves+1). */
    unsigned *nflist_indices; /**< Near-field flat source-leaf indices. */
    unsigned n_leaves;        /**< Number of leaves (for CSR sizing). */
    size_t vlist_count;       /**< Total V-list entries. */
    size_t nflist_count;      /**< Total near-field entries. */
} fmm_tree_t;

/* ------------------------------------------------------------------ */
/* Public API — count pass                                            */
/* ------------------------------------------------------------------ */

/**
 * @brief Run the sequential count pass to determine tree topology.
 *
 * Walks the source coordinates and builds the topology array, counting
 * internal nodes, multipole leaves, and particle leaves.  No coefficients
 * are computed; this pass only determines structural sizes.
 *
 * @param n_sources      Number of source points.
 * @param sources_coords Source coordinates (read-only).
 * @param settings       Build settings.
 * @param topo           Output topology array (pre-allocated from scratch).
 * @param source_leaf    Output per-source leaf index in the topology array.
 * @return Count results.
 */
fmm_count_res_t fmm_count_pass(unsigned n_sources, const real3_t CVL_ARRAY_ARG(sources_coords, restrict n_sources),
                               const fmm_settings_t *settings, void *topo, uint32_t *source_leaf);

/* ------------------------------------------------------------------ */
/* Public API — sizing                                                */
/* ------------------------------------------------------------------ */

/**
 * @brief Compute per-region scratch sizes from input parameters alone.
 *
 * @param n_sources Number of source points.
 * @param settings  Build settings.
 * @return Per-region scratch sizes.
 */
fmm_scratch_sizes_t fmm_size_scratch(unsigned n_sources, const fmm_settings_t *settings);

/**
 * @brief Total scratch buffer size from per-region sizes and thread count.
 *
 * @param sizes     Per-region sizes from @ref fmm_size_scratch.
 * @param n_threads Number of OpenMP threads (>= 1).
 * @return Total scratch buffer size in bytes.
 */
size_t fmm_total_scratch_size(fmm_scratch_sizes_t sizes, unsigned n_threads);

/**
 * @brief Compute the scratch buffer size required by the count and insert passes.
 *
 * @param n_sources Number of source points (must be > 0).
 * @param n_threads Number of OpenMP threads (>= 1).
 * @param settings  Build settings.
 * @return Required scratch size in bytes, or 0 on invalid input.
 */
size_t fmm_scratch_size(unsigned n_sources, unsigned n_threads, const fmm_settings_t *settings);

/**
 * @brief Compute the exact work buffer sizes from count-pass results.
 *
 * @param n_sources      Number of source points.
 * @param settings       Build settings.
 * @param count_pass_res Result from @ref fmm_count_pass.
 * @return A struct with per-region byte sizes.
 */
fmm_work_sizes_t fmm_size_work_buffer(unsigned n_sources, const fmm_settings_t CVL_ARRAY_ARG(settings, restrict),
                                      fmm_count_res_t count_pass_res);

/**
 * @brief Total work buffer size in bytes from per-region sizes.
 *
 * @param sizes Per-region sizes from @ref fmm_size_work_buffer.
 * @return Total buffer size in bytes.
 */
size_t fmm_total_work_size(fmm_work_sizes_t sizes);

/**
 * @brief Compute a pessimistic upper bound on the buffer size.
 *
 * @param n_sources Number of source points (must be > 0).
 * @param settings  Build settings.
 * @return Required buffer size in bytes, or 0 on invalid input.
 */
size_t fmm_buffer_size(unsigned n_sources, const fmm_settings_t *settings);

/* ------------------------------------------------------------------ */
/* Public API — build                                                 */
/* ------------------------------------------------------------------ */

/**
 * @brief Run the insert pass into pre-allocated buffers.
 *
 * @param n_sources      Number of source points (must be > 0).
 * @param n_threads      Number of OpenMP threads (>= 1).
 * @param sources_coords Coordinates of the source points.
 * @param sources_values Vector source strengths.
 * @param settings       Build settings.
 * @param scratch_buffer Transient scratch buffer.
 * @param scratch_size   Size of @p scratch_buffer in bytes.
 * @param allocator      Allocator callbacks. Pass NULL for libc default.
 * @param buffer         Persistent storage for the output tree.
 * @param buffer_size    Size of @p buffer in bytes.
 * @param out            Out-parameter for the populated tree handle.
 * @return true on success, false if buffers are too small.
 */
bool fmm_tree_insert(unsigned n_sources, unsigned n_threads,
                     const real3_t CVL_ARRAY_ARG(sources_coords, restrict n_sources),
                     const real3_t CVL_ARRAY_ARG(sources_values, restrict n_sources),
                     const fmm_settings_t CVL_ARRAY_ARG(settings, restrict), void *scratch_buffer, size_t scratch_size,
                     const allocator_t *allocator, void *buffer, size_t buffer_size, fmm_tree_t *out);

/**
 * @brief Convenience wrapper: run count + insert in one call.
 *
 * @param n_sources      Number of source points.
 * @param n_threads      Number of OpenMP threads (>= 1).
 * @param sources_coords Coordinates of the source points.
 * @param sources_values Vector source strengths.
 * @param settings       Build settings.
 * @param allocator      Allocator callbacks. Pass NULL for libc default.
 * @param out            Out-parameter for the populated tree handle.
 * @return true on success.
 */
bool fmm_tree_build(unsigned n_sources, unsigned n_threads,
                    const real3_t CVL_ARRAY_ARG(sources_coords, restrict n_sources),
                    const real3_t CVL_ARRAY_ARG(sources_values, restrict n_sources),
                    const fmm_settings_t CVL_ARRAY_ARG(settings, restrict), const allocator_t *allocator,
                    fmm_tree_t *out);

/* ------------------------------------------------------------------ */
/* Public API — inspection                                            */
/* ------------------------------------------------------------------ */

/**
 * @brief Total number of nodes stored in @p tree.
 *
 * @param tree Built tree handle.
 * @return Node count.
 */
unsigned fmm_tree_n_nodes(const fmm_tree_t *tree);

/**
 * @brief Total bytes occupied by the tree inside its buffer.
 *
 * @param tree Built tree handle.
 * @return Buffer size in bytes.
 */
size_t fmm_tree_memory_bytes(const fmm_tree_t *tree);

/* ------------------------------------------------------------------ */
/* Public API — evaluation                                            */
/* ------------------------------------------------------------------ */

/**
 * @brief Evaluate the tree at a single target point (tree-code mode).
 *
 * Descends to the target's leaf, then sums the far-field contribution
 * from V-list leaf multipoles (@ref multipole_eval) and the near-field
 * contribution from direct particle sums (@ref particle_kernel) over the
 * target's leaf and its near-field neighbours.
 *
 * @param tree            Built tree handle.
 * @param sources_coords  Source coordinates (same array used to build).
 * @param sources_values  Source strengths.
 * @param point           Target evaluation point.
 * @param eval_settings   Evaluation settings.
 * @return Induced velocity at @p point.
 */
real3_t fmm_tree_eval(const fmm_tree_t *tree, const real3_t CVL_ARRAY_ARG(sources_coords, restrict),
                      const real3_t CVL_ARRAY_ARG(sources_values, restrict), real3_t point,
                      fmm_eval_settings_t eval_settings);

/**
 * @brief Evaluate the tree at multiple target points (batched, OpenMP).
 *
 * Thread-safe: each target is evaluated independently.
 *
 * @param tree            Built tree handle.
 * @param sources_coords  Source coordinates.
 * @param sources_values  Source strengths.
 * @param n_targets       Number of target points.
 * @param targets         Array of target points.
 * @param results         Output array for induced velocities.
 * @param eval_settings   Evaluation settings.
 * @param n_threads       OpenMP thread count.
 */
void fmm_tree_eval_all(const fmm_tree_t *tree, const real3_t CVL_ARRAY_ARG(sources_coords, restrict),
                       const real3_t CVL_ARRAY_ARG(sources_values, restrict), unsigned n_targets,
                       const real3_t CVL_ARRAY_ARG(targets, restrict n_targets),
                       real3_t CVL_ARRAY_ARG(results, restrict n_targets), fmm_eval_settings_t eval_settings,
                       unsigned n_threads);
