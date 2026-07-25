#include "barnes_hut_tree.h"

#include <omp.h>

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* Work buffer partition helper                                       */
/* ------------------------------------------------------------------ */

static barnes_hut_work_t partition_work_buffer(octree_base_work_sizes_t work_sizes, void *buffer)
{
    uint8_t *bp = (uint8_t *)buffer;
    barnes_hut_work_t out = {0};
    out.nodes = (octree_node_t *)bp;
    bp += work_sizes.nodes_bytes;
    out.particle_order = (unsigned *)bp;
    bp += work_sizes.particle_order_bytes;
    out.multipole_coeffs = (real_t *)bp;
    bp += work_sizes.multipole_coeffs_bytes;
    out.topo_to_real = (uint32_t *)bp;
    bp += work_sizes.topo_to_real_bytes;
    out.mp_slices = (real_t **)bp;
    return out;
}

/* ------------------------------------------------------------------ */
/* Staged build helpers                                               */
/* ------------------------------------------------------------------ */

size_t barnes_hut_scratch_size(unsigned n_sources, const barnes_hut_settings_t *settings, unsigned n_threads)
{
    return octree_scratch_size(n_sources, n_threads, (const octree_settings_t *)settings);
}

bool barnes_hut_prepare_scratch(void *scratch_buffer, size_t scratch_size, unsigned n_sources, unsigned n_threads,
                                const real3_t sources_coords[restrict n_sources],
                                const barnes_hut_settings_t settings[restrict], octree_count_t *out_count,
                                octree_scratch_t *out_scratch)
{
    if (scratch_buffer == NULL || out_count == NULL || out_scratch == NULL)
        return false;
    if (n_sources == 0 || settings == NULL || sources_coords == NULL)
        return false;

    const octree_scratch_sizes_t sz = octree_size_scratch(n_sources, (const octree_settings_t *)settings);
    if (scratch_size < octree_total_scratch_size(sz, n_threads))
        return false;

    *out_scratch = octree_scratch_partition(n_threads, scratch_buffer, sz);
    const size_t topo_bytes = (8u * (size_t)n_sources + 1u) * sizeof(topo_node_t);
    memset(out_scratch->topo, 0, topo_bytes);

    *out_count = octree_count_pass(n_sources, sources_coords, (const octree_settings_t *)settings, out_scratch->topo,
                                   out_scratch->source_leaf_topo);
    return true;
}

size_t barnes_hut_work_size(unsigned n_sources, const barnes_hut_settings_t *settings, const octree_count_t *count)
{
    if (count == NULL || count->n_internal + count->n_multipole_leaves + count->n_particle_leaves == 0)
        return 0;
    const octree_base_work_sizes_t ws = octree_size_work_buffer(n_sources, (const octree_settings_t *)settings, *count);
    return octree_total_work_size(ws);
}

/* ------------------------------------------------------------------ */
/* Insert pass (pre-counted, pre-partitioned scratch)                 */
/* ------------------------------------------------------------------ */

bool barnes_hut_tree_insert(unsigned n_sources, unsigned n_threads, const real3_t sources_coords[restrict n_sources],
                            const real3_t sources_values[restrict n_sources],
                            const barnes_hut_settings_t settings[restrict], const octree_count_t *count,
                            const octree_scratch_t *scratch, const allocator_t *allocator, void *buffer,
                            size_t buffer_size, barnes_hut_tree_t *out)
{
    if (!buffer || !out || !scratch || !count)
        return false;
    if (n_sources == 0 || settings == NULL || sources_coords == NULL || sources_values == NULL)
        return false;

    const unsigned n_topo = count->n_internal + count->n_multipole_leaves + count->n_particle_leaves;
    if (n_topo == 0)
        return false;

    const octree_base_work_sizes_t ws = octree_size_work_buffer(n_sources, (const octree_settings_t *)settings, *count);
    if (buffer_size < octree_total_work_size(ws))
        return false;
    const barnes_hut_work_t work = partition_work_buffer(ws, buffer);

    /* Shared pipeline. */
    octree_materialize(scratch->topo, n_topo, (const octree_settings_t *)settings, work.topo_to_real, work.nodes,
                       work.multipole_coeffs, work.mp_slices, n_threads);
    octree_descend(n_sources, sources_coords, work.nodes, scratch->source_leaf_real, n_threads);

    const unsigned n_mp = octree_compute_metadata(n_topo, work.nodes, count->max_depth, NULL, NULL);

    octree_fill_particle_order(n_sources, scratch->source_leaf_real, work.nodes, work.particle_order, n_threads);
    octree_compute_leaf_centers(n_topo, work.nodes, work.particle_order, sources_coords, sources_values, n_threads);

    if (n_mp > 0)
    {
        if (!octree_build_leaf_multipoles(n_topo, work.nodes, work.particle_order, sources_coords, sources_values,
                                          (const octree_settings_t *)settings, *scratch, n_sources, n_threads,
                                          work.mp_slices, n_mp, allocator))
            return false;
    }

    /* Upward sweep (M2M). */
    octree_run_upward_sweep(n_topo, work.nodes, count->max_depth, (const octree_settings_t *)settings, work.mp_slices,
                            scratch, work.particle_order, sources_coords, sources_values, n_threads);

    /* Populate tree handle. */
    out->settings = *settings;
    out->root_center = work.nodes[0].center;
    out->root_half_size = work.nodes[0].half_size;
    out->n_sources = n_sources;
    out->n_nodes = n_topo;
    out->n_internal = count->n_internal;
    out->n_multipole_leaves = n_mp;
    out->n_particle_leaves = count->n_particle_leaves;
    out->max_depth_reached = count->max_depth;
    out->buffer = (uint8_t *)buffer;
    out->buffer_size = buffer_size;
    out->nodes = work.nodes;
    out->particle_order = work.particle_order;
    out->multipole_coeffs = work.multipole_coeffs;
    out->mp_slices = work.mp_slices;

    return true;
}

bool barnes_hut_tree_build(unsigned n_sources, unsigned n_threads, const real3_t sources_coords[restrict n_sources],
                           const real3_t sources_values[restrict n_sources],
                           const barnes_hut_settings_t settings[restrict], const allocator_t *allocator,
                           barnes_hut_tree_t *out)
{
    void *scratch_buffer = NULL;
    bool ret = false;

    if (!out || n_sources == 0 || settings == NULL || sources_coords == NULL || sources_values == NULL)
        return false;

    /* 1. Size scratch. */
    const size_t needed_scratch = barnes_hut_scratch_size(n_sources, settings, n_threads);
    if (needed_scratch == 0)
        return false;

    /* 2. Allocate scratch. */
    scratch_buffer = octree_alloc(allocator, needed_scratch);
    if (!scratch_buffer)
        return false;

    /* 3. Prepare scratch (partition + count). */
    octree_count_t count;
    octree_scratch_t scratch;
    if (!barnes_hut_prepare_scratch(scratch_buffer, needed_scratch, n_sources, n_threads, sources_coords, settings,
                                    &count, &scratch))
        goto cleanup;

    /* 4. Size work buffer. */
    const size_t total_work_size = barnes_hut_work_size(n_sources, settings, &count);
    if (total_work_size == 0)
        goto cleanup;

    /* 5. Allocate work buffer. */
    void *work_buffer = octree_alloc(allocator, total_work_size);
    if (!work_buffer)
        goto cleanup;

    /* 6. Full pipeline (insert). */
    if (!barnes_hut_tree_insert(n_sources, n_threads, sources_coords, sources_values, settings, &count, &scratch,
                                allocator, work_buffer, total_work_size, out))
    {
        octree_free(allocator, work_buffer);
        goto cleanup;
    }

    ret = true;

cleanup:
    /* 7. Release scratch. */
    octree_free(allocator, scratch_buffer);
    return ret;
}

unsigned barnes_hut_tree_n_nodes(const barnes_hut_tree_t *tree)
{
    return tree ? tree->n_nodes : 0;
}

void barnes_hut_tree_depth_stats(const barnes_hut_tree_t *tree, unsigned *min_depth, unsigned *max_depth)
{
    if (tree == NULL || tree->n_nodes == 0)
    {
        if (min_depth)
            *min_depth = 0;
        if (max_depth)
            *max_depth = 0;
        return;
    }
    unsigned lo = tree->nodes[0].depth;
    unsigned hi = tree->nodes[0].depth;
    for (unsigned i = 1; i < tree->n_nodes; ++i)
    {
        const unsigned d = tree->nodes[i].depth;
        if (d < lo)
            lo = d;
        if (d > hi)
            hi = d;
    }
    if (min_depth)
        *min_depth = lo;
    if (max_depth)
        *max_depth = hi;
}

size_t barnes_hut_tree_memory_bytes(const barnes_hut_tree_t *tree)
{
    return tree ? tree->buffer_size : 0;
}

/* ------------------------------------------------------------------ */
/* Particle kernel and MAC                                            */
/* ------------------------------------------------------------------ */

/**
 * @brief Multipole Acceptance Criterion (MAC).
 *
 * When `theta <= 0` (default): neighbour criterion — accept if the target
 * point is outside the cell's @f$ 3 \times 3 \times 3 @f$ neighbourhood
 * (@f$ |\Delta x| > 2 h @f$ or @f$ |\Delta y| > 2 h @f$ or
 * @f$ |\Delta z| > 2 h @f$).
 *
 * When `theta > 0`: opening-angle criterion — accept if
 * @f$ h / |r| < \theta @f$.
 */
static inline bool mac_accept(const octree_node_t *node, real3_t point, double theta)
{
    const real3_t diff = real3_sub(point, node->center);
    if (theta <= 0.0)
    {
        /* Neighbour criterion: outside 3x3x3 cell neighbourhood. */
        return fabs(diff.x) > 2.0 * node->half_size || fabs(diff.y) > 2.0 * node->half_size ||
               fabs(diff.z) > 2.0 * node->half_size;
    }
    /* Opening-angle criterion. */
    const real_t dist = real3_mag(diff);
    if (dist < 1e-30)
        return false;
    return node->half_size / dist < theta;
}

real3_t barnes_hut_tree_eval(const barnes_hut_tree_t *tree, const real3_t CVL_ARRAY_ARG(sources_coords, restrict),
                             const real3_t CVL_ARRAY_ARG(sources_values, restrict), real3_t point,
                             barnes_hut_eval_settings_t eval_settings)
{
    if (tree == NULL || tree->nodes == NULL || tree->n_nodes == 0)
        return (real3_t){.x = 0, .y = 0, .z = 0};

    real3_t result = {.x = 0, .y = 0, .z = 0};

    enum
    {
        EVAL_STACK_MAX = 1024
    };
    uint32_t stack[EVAL_STACK_MAX];
    int sp = 0;
    stack[sp++] = 0;

    const double theta = eval_settings.theta;
    const unsigned order = tree->settings.order;
    const size_t n_coeffs = multipole_num_coeffs(order);

    while (sp > 0)
    {
        sp -= 1;
        const uint32_t idx = stack[sp];
        const octree_node_t *node = &tree->nodes[idx];

        /* Hot path: OCTREE_NODE_INTERNAL is most common, especially near root. */
        if (CVL_EXPECT_CONDITION(node->kind == OCTREE_NODE_INTERNAL))
        {
            if (mac_accept(node, point, theta))
            {
                /* Evaluate the internal node's aggregated multipole.
                 * Prefetch coefficient arrays before constructing the
                 * multipole_t to hide memory latency. */
                real_t *slice = tree->mp_slices[idx];
                if (CVL_EXPECT_CONDITION(slice != NULL))
                {
                    CVL_PREFETCH(slice, 0, 3);
                    CVL_PREFETCH(slice + n_coeffs, 0, 3);
                    CVL_PREFETCH(slice + 2u * n_coeffs, 0, 3);
                    const multipole_t mp = {.order = order,
                                            .center = node->center,
                                            .coeffs_x = slice,
                                            .coeffs_y = slice + n_coeffs,
                                            .coeffs_z = slice + 2u * n_coeffs};
                    result = real3_add(result, multipole_eval(&mp, point));
                }
            }
            else
            {
                /* Descend into children. */
                for (int k = 0; k < 8; ++k)
                {
                    const octree_node_t *child = node->data.internal.children[k];
                    if (CVL_EXPECT_CONDITION(child == NULL))
                        continue;
                    if ((size_t)sp + 1 > EVAL_STACK_MAX)
                        break;
                    stack[sp++] = (uint32_t)(child - tree->nodes);
                }
            }
        }
        else
        {
            /* OCTREE_NODE_MULTIPOLE or OCTREE_NODE_PARTICLE — if the leaf has a
             * far-field multipole and the MAC accepts it, use the multipole.
             * Otherwise fall back to direct particle sum for accuracy. */
            real_t *slice = tree->mp_slices[idx];
            if (slice != NULL && mac_accept(node, point, theta))
            {
                CVL_PREFETCH(slice, 0, 3);
                CVL_PREFETCH(slice + n_coeffs, 0, 3);
                CVL_PREFETCH(slice + 2u * n_coeffs, 0, 3);
                const multipole_t mp = {.order = order,
                                        .center = node->data.mp.center,
                                        .coeffs_x = slice,
                                        .coeffs_y = slice + n_coeffs,
                                        .coeffs_z = slice + 2u * n_coeffs};
                result = real3_add(result, multipole_eval(&mp, point));
            }
            else
            {
                const unsigned begin = node->particle_begin;
                const unsigned end = begin + node->particle_count;
                for (unsigned k = begin; k < end; ++k)
                {
                    const unsigned src = tree->particle_order[k];
                    const real3_t dr = real3_sub(point, sources_coords[src]);
                    result = real3_add(result, particle_kernel(sources_values[src], dr));
                }
            }
        }
    }

    return result;
}

void barnes_hut_tree_eval_all(const barnes_hut_tree_t *tree, const real3_t CVL_ARRAY_ARG(sources_coords, restrict),
                              const real3_t CVL_ARRAY_ARG(sources_values, restrict), unsigned n_targets,
                              const real3_t CVL_ARRAY_ARG(targets, restrict n_targets),
                              real3_t CVL_ARRAY_ARG(results, restrict n_targets),
                              barnes_hut_eval_settings_t eval_settings, unsigned n_threads)
{
#pragma omp parallel for default(none) shared(tree, sources_coords, sources_values, n_targets, targets, results,       \
                                                  eval_settings) num_threads(n_threads) schedule(static)
    for (unsigned i = 0; i < n_targets; ++i)
    {
        results[i] = barnes_hut_tree_eval(tree, sources_coords, sources_values, targets[i], eval_settings);
    }
}
