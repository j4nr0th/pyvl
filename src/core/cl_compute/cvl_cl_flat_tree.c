/*
 * Uniform octree builder - flat, level-ordered node array.
 *
 * Implements the builder declared in cvl_cl_flat_tree.h.
 *
 * Morton codes are assumed sorted ascending.  Nodes are built bottom-up:
 * leaves first (grouped by Morton key at max_depth), then internal nodes
 * grouped by parent key at each shallower depth.
 *
 * All output buffers alias the caller's work buffer - no allocation happens.
 */

#include "cvl_cl_flat_tree.h"

#include <assert.h>
#include <math.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* Internal helpers                                                    */
/* ------------------------------------------------------------------ */

/** @brief Clamp depth to the representable range [0, CVL_CL_FLAT_TREE_MAX_DEPTH]. */
static inline unsigned clamp_depth(unsigned depth)
{
    return depth > CVL_CL_FLAT_TREE_MAX_DEPTH ? CVL_CL_FLAT_TREE_MAX_DEPTH : depth;
}

/* ------------------------------------------------------------------ */
/* cvl_cl_flat_tree_count                                              */
/* ------------------------------------------------------------------ */

cvl_cl_status_t cvl_cl_flat_tree_count(unsigned n_sources, const uint64_t morton_codes[restrict],
                                       const cvl_cl_flat_tree_settings_t settings[restrict],
                                       unsigned out_depth_counts[restrict], unsigned *out_n_total,
                                       unsigned *out_max_depth_used)
{
    assert(morton_codes && settings && out_depth_counts && out_n_total && out_max_depth_used);

    const unsigned max_depth = clamp_depth(settings->max_depth);

    if (n_sources == 0)
    {
        for (unsigned d = 0; d <= max_depth; ++d)
            out_depth_counts[d] = 0;
        *out_n_total = 0;
        *out_max_depth_used = 0;
        return CVL_CL_SUCCESS;
    }

    /* For each depth level d (0 … max_depth), count the number of
     * distinct groups formed by the depth-d Morton key. */
    unsigned total = 0;
    for (unsigned d = 0; d <= max_depth; ++d)
    {
        const unsigned shift = cvl_cl_flat_tree_depth_shift(d);
        unsigned unique = 1; /* at least one group when n_sources > 0 */

        for (unsigned i = 1; i < n_sources; ++i)
        {
            if ((morton_codes[i] >> shift) != (morton_codes[i - 1] >> shift))
                ++unique;
        }

        out_depth_counts[d] = unique;
        total += unique;
    }

    *out_n_total = total;
    *out_max_depth_used = max_depth;
    return CVL_CL_SUCCESS;
}

/* ------------------------------------------------------------------ */
/* cvl_cl_flat_tree_build                                              */
/* ------------------------------------------------------------------ */

cvl_cl_status_t cvl_cl_flat_tree_build(unsigned n_sources, const real3_t sources_coords[restrict n_sources],
                                       const unsigned particle_indices[restrict n_sources],
                                       const uint64_t morton_codes[restrict n_sources],
                                       const cvl_cl_flat_tree_settings_t settings[restrict],
                                       cvl_cl_flat_tree_t *out_tree, void *work, size_t work_size)
{
    assert(sources_coords && particle_indices && morton_codes && settings && out_tree && work);
    assert(n_sources > 0);

    const unsigned max_depth = clamp_depth(settings->max_depth);

    /* ---- 1. Count nodes per depth level ---- */
    unsigned depth_counts[CVL_CL_FLAT_TREE_MAX_DEPTH + 2];
    unsigned n_total = 0;
    unsigned max_depth_used = 0;

    cvl_cl_status_t status;
    status = cvl_cl_flat_tree_count(n_sources, morton_codes, settings, depth_counts, &n_total, &max_depth_used);
    if (status != CVL_CL_SUCCESS)
        return status;

    /* ---- 2. Compute root bounding box ---- */
    real3_t bbox_min = sources_coords[0];
    real3_t bbox_max = sources_coords[0];

    for (unsigned i = 1; i < n_sources; ++i)
    {
        const real3_t p = sources_coords[i];
        bbox_min.x = fmin(bbox_min.x, p.x);
        bbox_min.y = fmin(bbox_min.y, p.y);
        bbox_min.z = fmin(bbox_min.z, p.z);
        bbox_max.x = fmax(bbox_max.x, p.x);
        bbox_max.y = fmax(bbox_max.y, p.y);
        bbox_max.z = fmax(bbox_max.z, p.z);
    }

    const real_t root_extent_x = bbox_max.x - bbox_min.x;
    const real_t root_extent_y = bbox_max.y - bbox_min.y;
    const real_t root_extent_z = bbox_max.z - bbox_min.z;
    const real_t root_half_size = fmax(fmax(root_extent_x, root_extent_y), root_extent_z) * 0.5 + (real_t)1e-12;

    /* ---- 3. Partition work buffer ---- */
    const size_t needed = cvl_cl_flat_tree_work_size(n_total, n_sources, max_depth);
    if (work_size < needed)
        return CVL_CL_ERR_BUFFER_SIZE;

    uint8_t *bp = (uint8_t *)work;
    out_tree->nodes = (cvl_cl_flat_node_t *)bp;
    bp += (size_t)n_total * sizeof(cvl_cl_flat_node_t);
    out_tree->particle_order = (unsigned *)bp;
    bp += (size_t)n_sources * sizeof(unsigned);
    out_tree->depth_offsets = (unsigned *)bp;
    bp += (size_t)(max_depth + 2) * sizeof(unsigned);

    /* Zero the node array. */
    memset(out_tree->nodes, 0, (size_t)n_total * sizeof(cvl_cl_flat_node_t));

    cvl_cl_flat_node_t *nodes = out_tree->nodes;
    unsigned *particle_order = out_tree->particle_order;
    unsigned *depth_offsets = out_tree->depth_offsets;

    /* ---- 4. Compute prefix sum -> depth_offsets ---- */
    {
        unsigned acc = 0;
        for (unsigned d = 0; d <= max_depth; ++d)
        {
            depth_offsets[d] = acc;
            acc += depth_counts[d];
        }
        depth_offsets[max_depth + 1] = acc;
    }

    /* ---- 5. Build leaf nodes (depth = max_depth) ---- */
    {
        const unsigned shift = cvl_cl_flat_tree_depth_shift(max_depth);
        unsigned leaf_counter = 0; /* index within leaf layer */
        unsigned particle_counter = 0;

        unsigned group_start = 0;
        for (unsigned i = 0; i < n_sources; ++i)
        {
            /* Does this particle start a new group?  Always true for i == 0;
             * otherwise compare Morton key at max_depth. */
            const unsigned is_new_group = (i == 0) || ((morton_codes[i] >> shift) != (morton_codes[i - 1] >> shift));

            if (is_new_group && i != 0)
            {
                /* Finalise the *previous* group [group_start .. i-1]. */
                const unsigned group_end = i;
                const unsigned n_particles = group_end - group_start;

                /* Centroid of particles in this leaf. */
                real3_t centroid = {{0}};
                for (unsigned j = group_start; j < group_end; ++j)
                {
                    centroid = real3_add(centroid, sources_coords[particle_indices[j]]);
                }
                centroid.x /= (real_t)n_particles;
                centroid.y /= (real_t)n_particles;
                centroid.z /= (real_t)n_particles;

                const unsigned leaf_idx = depth_offsets[max_depth] + leaf_counter;
                cvl_cl_flat_node_t *node = &nodes[leaf_idx];
                node->center = centroid;
                node->half_size = root_half_size / (real_t)(1u << max_depth);
                node->morton_code = morton_codes[group_start];
                node->child_base = -1;
                node->particle_begin = (int32_t)particle_counter;
                node->particle_count = (int16_t)n_particles;
                node->child_mask = 0;
                node->kind = (n_particles > settings->critical_particle_count) ? CVL_CL_FLAT_NODE_MULTIPOLE
                                                                               : CVL_CL_FLAT_NODE_PARTICLE;

                /* Append particle indices to particle_order. */
                for (unsigned j = group_start; j < group_end; ++j)
                    particle_order[particle_counter++] = particle_indices[j];

                ++leaf_counter;
                group_start = i;
            }

            /* Last particle always closes the final group. */
            if (i == n_sources - 1)
            {
                const unsigned group_end = i + 1;
                const unsigned n_particles = group_end - group_start;

                real3_t centroid = {{0}};
                for (unsigned j = group_start; j < group_end; ++j)
                {
                    centroid = real3_add(centroid, sources_coords[particle_indices[j]]);
                }
                centroid.x /= (real_t)n_particles;
                centroid.y /= (real_t)n_particles;
                centroid.z /= (real_t)n_particles;

                const unsigned leaf_idx = depth_offsets[max_depth] + leaf_counter;
                cvl_cl_flat_node_t *node = &nodes[leaf_idx];
                node->center = centroid;
                node->half_size = root_half_size / (real_t)(1u << max_depth);
                node->morton_code = morton_codes[group_start];
                node->child_base = -1;
                node->particle_begin = (int32_t)particle_counter;
                node->particle_count = (int16_t)n_particles;
                node->child_mask = 0;
                node->kind = (n_particles > settings->critical_particle_count) ? CVL_CL_FLAT_NODE_MULTIPOLE
                                                                               : CVL_CL_FLAT_NODE_PARTICLE;

                for (unsigned j = group_start; j < group_end; ++j)
                    particle_order[particle_counter++] = particle_indices[j];

                /* leaf_counter not incremented here - not needed after final group. */
            }
        }
    }

    /* ---- 6. Build internal nodes bottom-up ---- */
    /* Walk depths from max_depth-1 down to 0, grouping children
     * (nodes at depth d+1) by their parent key (Morton code at depth d). */
    for (int d = (int)max_depth - 1; d >= 0; --d)
    {
        const unsigned depth = (unsigned)d;
        const unsigned child_off = depth_offsets[depth + 1];
        const unsigned n_children = depth_counts[depth + 1];
        const unsigned parent_off = depth_offsets[depth];
        const unsigned shift_parent = cvl_cl_flat_tree_depth_shift(depth);
        const unsigned shift_octant = cvl_cl_flat_tree_depth_shift(depth + 1);

        unsigned parent_counter = 0; /* index within parent layer */
        unsigned child_group_start = 0;

        for (unsigned ci = 0; ci < n_children; ++ci)
        {
            const cvl_cl_flat_node_t *child = &nodes[child_off + ci];
            const uint64_t child_mc = child->morton_code;

            /* Parent key = top 3*depth bits of child Morton code. */
            const uint64_t parent_key = child_mc >> shift_parent;

            /* Does a *next* child with a different parent exist? */
            int is_last_in_group = 1; /* assume last initially */
            if (ci + 1 < n_children)
            {
                const cvl_cl_flat_node_t *next = &nodes[child_off + ci + 1];
                const uint64_t next_key = next->morton_code >> shift_parent;
                is_last_in_group = (parent_key != next_key);
            }

            if (is_last_in_group)
            {
                /* All children from child_group_start … ci share the same parent. */
                const unsigned parent_idx = parent_off + parent_counter;
                cvl_cl_flat_node_t *parent = &nodes[parent_idx];

                /* Weighted centroid of child centres.  Leaves use particle_count
                 * as weight; internal children use 1.0. */
                real3_t center_sum = {{0}};
                real_t total_weight = 0;
                uint8_t mask = 0;

                for (unsigned cj = child_group_start; cj <= ci; ++cj)
                {
                    const cvl_cl_flat_node_t *c = &nodes[child_off + cj];
                    const uint8_t octant = (uint8_t)((c->morton_code >> shift_octant) & 0x7u);
                    mask = (uint8_t)(mask | (uint8_t)(1u << octant));

                    const real_t w = (c->kind == CVL_CL_FLAT_NODE_INTERNAL) ? (real_t)1.0 : (real_t)c->particle_count;
                    center_sum.x += c->center.x * w;
                    center_sum.y += c->center.y * w;
                    center_sum.z += c->center.z * w;
                    total_weight += w;
                }

                parent->center.x = center_sum.x / total_weight;
                parent->center.y = center_sum.y / total_weight;
                parent->center.z = center_sum.z / total_weight;
                parent->half_size = root_half_size / (real_t)(1u << depth);
                parent->morton_code = parent_key << shift_parent;
                parent->child_base = (int32_t)(child_off + child_group_start);
                parent->particle_begin = 0;
                parent->particle_count = 0;
                parent->child_mask = mask;
                parent->kind = CVL_CL_FLAT_NODE_INTERNAL;

                ++parent_counter;
                child_group_start = ci + 1;
            }
        }
    }

    /* ---- 7. Populate output metadata ---- */
    {
        unsigned n_internal = 0;
        unsigned n_multipole_leaves = 0;
        unsigned n_particle_leaves = 0;

        for (unsigned i = 0; i < n_total; ++i)
        {
            switch ((cvl_cl_flat_node_kind_t)nodes[i].kind)
            {
            case CVL_CL_FLAT_NODE_INTERNAL:
                ++n_internal;
                break;
            case CVL_CL_FLAT_NODE_MULTIPOLE:
                ++n_multipole_leaves;
                break;
            case CVL_CL_FLAT_NODE_PARTICLE:
                ++n_particle_leaves;
                break;
            }
        }

        out_tree->n_nodes = n_total;
        out_tree->n_internal = n_internal;
        out_tree->n_multipole_leaves = n_multipole_leaves;
        out_tree->n_particle_leaves = n_particle_leaves;
        out_tree->max_depth = max_depth_used;
    }

    return CVL_CL_SUCCESS;
}

/* ------------------------------------------------------------------ */
/* cvl_cl_flat_tree_destroy                                            */
/* ------------------------------------------------------------------ */

void cvl_cl_flat_tree_destroy(cvl_cl_flat_tree_t *tree)
{
    if (!tree)
        return;

    /* The work buffer is caller-owned - just clear the handle. */
    tree->nodes = NULL;
    tree->particle_order = NULL;
    tree->depth_offsets = NULL;
    tree->n_nodes = 0;
    tree->n_internal = 0;
    tree->n_multipole_leaves = 0;
    tree->n_particle_leaves = 0;
    tree->max_depth = 0;
}
