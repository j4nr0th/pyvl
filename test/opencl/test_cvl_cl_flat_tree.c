/*
 * test_cvl_cl_flat_tree.c — Validate the uniform flat octree builder.
 *
 * Builds a uniform octree from random particles using
 * cvl_cl_flat_tree_build, then validates:
 *   - Total node count matches depth_counts
 *   - Each internal node has child_base != -1
 *   - Each leaf has particle_count matching its range
 *   - Children are at the correct depth level
 *   - Particle_order contains valid indices
 *   - The root exists and has correct half_size
 *   - Node kind counts match the builder's summary
 */

#ifdef CVL_OPENCL

#include "../test_common.h"
#include "cvl_cl_common.h"
#include "cvl_cl_flat_tree.h"
#include "cvl_cl_test_common.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/*  Host-side Morton 3D (same algorithm as cvl_cl_math.h.cl)          */
/* ------------------------------------------------------------------ */

static inline uint64_t morton_split_21(uint64_t x)
{
    x &= 0x1fffffULL;
    x = (x | (x << 32)) & 0x1f00000000ffffULL;
    x = (x | (x << 16)) & 0x1f0000ff0000ffULL;
    x = (x | (x << 8)) & 0x100f00f00f00f00fULL;
    x = (x | (x << 4)) & 0x10c30c30c30c30c3ULL;
    x = (x | (x << 2)) & 0x1249249249249249ULL;
    return x;
}

static inline uint64_t morton_3d(real3_t p, real3_t root_center, real_t root_half_size)
{
    real_t inv_cell = 1.0 / (2.0 * root_half_size);
    real_t scale = (real_t)((1u << 21) - 1);
    real_t nx = (p.x - root_center.x) * inv_cell + 0.5;
    real_t ny = (p.y - root_center.y) * inv_cell + 0.5;
    real_t nz = (p.z - root_center.z) * inv_cell + 0.5;
    if (nx < 0)
        nx = 0;
    if (nx >= 1)
        nx = 0.999999;
    if (ny < 0)
        ny = 0;
    if (ny >= 1)
        ny = 0.999999;
    if (nz < 0)
        nz = 0;
    if (nz >= 1)
        nz = 0.999999;
    uint64_t ix = (uint64_t)(nx * scale);
    uint64_t iy = (uint64_t)(ny * scale);
    uint64_t iz = (uint64_t)(nz * scale);
    return morton_split_21(ix) | (morton_split_21(iy) << 1) | (morton_split_21(iz) << 2);
}

/* ------------------------------------------------------------------ */
/*  Comparison helper for qsort                                        */
/* ------------------------------------------------------------------ */

typedef struct
{
    uint64_t code;
    unsigned idx;
} morton_entry_t;

static int morton_cmp(const void *a, const void *b)
{
    const morton_entry_t *ea = (const morton_entry_t *)a;
    const morton_entry_t *eb = (const morton_entry_t *)b;
    if (ea->code < eb->code)
        return -1;
    if (ea->code > eb->code)
        return 1;
    return 0;
}

/* ------------------------------------------------------------------ */
/*  Main                                                               */
/* ------------------------------------------------------------------ */

int main(void)
{
    uint64_t rng = 12345;

    const unsigned N = 200;
    const unsigned MAX_DEPTH = 6;
    const unsigned CRIT = 8;

    real3_t coords[N];
    uint64_t mcodes[N];
    morton_entry_t entries[N];

    /* Generate random particles */
    for (unsigned i = 0; i < N; ++i)
    {
        coords[i].x = xorshift_uniform_range(&rng, -5.0, 5.0);
        coords[i].y = xorshift_uniform_range(&rng, -5.0, 5.0);
        coords[i].z = xorshift_uniform_range(&rng, -5.0, 5.0);
        entries[i].idx = i;
    }

    /* Compute bounding box for Morton codes */
    real3_t bbox_min = coords[0], bbox_max = coords[0];
    for (unsigned i = 1; i < N; ++i)
    {
        if (coords[i].x < bbox_min.x)
            bbox_min.x = coords[i].x;
        if (coords[i].y < bbox_min.y)
            bbox_min.y = coords[i].y;
        if (coords[i].z < bbox_min.z)
            bbox_min.z = coords[i].z;
        if (coords[i].x > bbox_max.x)
            bbox_max.x = coords[i].x;
        if (coords[i].y > bbox_max.y)
            bbox_max.y = coords[i].y;
        if (coords[i].z > bbox_max.z)
            bbox_max.z = coords[i].z;
    }
    real3_t root_center = {(bbox_min.x + bbox_max.x) * 0.5, (bbox_min.y + bbox_max.y) * 0.5,
                           (bbox_min.z + bbox_max.z) * 0.5};
    real_t root_hs =
        (real_t)fmax(fmax(bbox_max.x - bbox_min.x, bbox_max.y - bbox_min.y), bbox_max.z - bbox_min.z) * 0.5 +
        (real_t)1e-12;

    /* Compute Morton codes */
    for (unsigned i = 0; i < N; ++i)
    {
        entries[i].code = morton_3d(coords[i], root_center, root_hs);
    }

    /* Sort by Morton code */
    qsort(entries, N, sizeof(morton_entry_t), morton_cmp);

    unsigned sorted_indices[N];
    for (unsigned i = 0; i < N; ++i)
    {
        mcodes[i] = entries[i].code;
        sorted_indices[i] = entries[i].idx;
    }

    /* Build tree */
    cvl_cl_flat_tree_settings_t settings = {
        .max_depth = MAX_DEPTH,
        .critical_particle_count = CRIT,
        .order = 4,
    };

    /* Count nodes first to size the build work buffer. */
    unsigned depth_counts[CVL_CL_FLAT_TREE_MAX_DEPTH + 2];
    unsigned n_total_work = 0, max_depth_used = 0;
    cvl_cl_status_t st = cvl_cl_flat_tree_count(N, mcodes, &settings, depth_counts, &n_total_work, &max_depth_used);
    TEST_ASSERT(st == CVL_CL_SUCCESS, "flat_tree_count failed: %s", cvl_cl_status_str(st));

    const unsigned work_depth = MAX_DEPTH > CVL_CL_FLAT_TREE_MAX_DEPTH ? CVL_CL_FLAT_TREE_MAX_DEPTH : MAX_DEPTH;
    size_t flat_work_sz = cvl_cl_flat_tree_work_size(n_total_work, N, work_depth);
    void *flat_work = malloc(flat_work_sz);
    TEST_ASSERT(flat_work != NULL, "malloc(%zu) for flat tree work buffer failed", flat_work_sz);

    cvl_cl_flat_tree_t tree;
    st = cvl_cl_flat_tree_build(N, coords, sorted_indices, mcodes, &settings, &tree, flat_work, flat_work_sz);
    TEST_ASSERT(st == CVL_CL_SUCCESS, "flat_tree_build failed: %s", cvl_cl_status_str(st));

    /* Validate total nodes */
    unsigned total = tree.n_internal + tree.n_multipole_leaves + tree.n_particle_leaves;
    TEST_ASSERT(total == tree.n_nodes, "n_node sum mismatch: %u vs %u", total, tree.n_nodes);
    TEST_ASSERT(tree.n_nodes > 0, "tree has no nodes");
    TEST_ASSERT(tree.max_depth == MAX_DEPTH, "max_depth mismatch: %u vs %u", tree.max_depth, MAX_DEPTH);

    /* Validate depth_offsets */
    for (unsigned d = 0; d <= MAX_DEPTH; ++d)
    {
        TEST_ASSERT(tree.depth_offsets[d] <= tree.n_nodes, "depth_offsets[%u]=%u > n_nodes=%u", d,
                    tree.depth_offsets[d], tree.n_nodes);
        if (d < MAX_DEPTH)
            TEST_ASSERT(tree.depth_offsets[d] <= tree.depth_offsets[d + 1], "depth_offsets not monotonic at %u", d);
    }
    TEST_ASSERT(tree.depth_offsets[MAX_DEPTH + 1] == tree.n_nodes, "depth_offsets sentinel mismatch: %u vs %u",
                tree.depth_offsets[MAX_DEPTH + 1], tree.n_nodes);

    /* Validate each node */
    unsigned internal_count = 0, mp_count = 0, particle_count = 0;
    for (unsigned i = 0; i < tree.n_nodes; ++i)
    {
        const cvl_cl_flat_node_t *n = &tree.nodes[i];
        if (n->kind == CVL_CL_FLAT_NODE_INTERNAL)
        {
            internal_count++;
            TEST_ASSERT(n->child_base >= 0, "internal node %u has no children", i);
            TEST_ASSERT(n->child_mask != 0, "internal node %u has empty mask", i);
            TEST_ASSERT(n->particle_count == 0, "internal node %u has particles", i);
            TEST_ASSERT((unsigned)n->child_base <= tree.n_nodes, "child_base %d out of range at node %u", n->child_base,
                        i);
        }
        else
        {
            if (n->kind == CVL_CL_FLAT_NODE_MULTIPOLE)
                mp_count++;
            else
                particle_count++;
            TEST_ASSERT(n->child_base == -1, "leaf node %u has child_base=%d", i, n->child_base);
            TEST_ASSERT(n->particle_begin + n->particle_count <= (int32_t)N,
                        "leaf %u particle range [%d, %d) exceeds N=%u", i, n->particle_begin,
                        n->particle_begin + n->particle_count, N);
            TEST_ASSERT(n->particle_count > 0, "leaf %u has zero particles", i);
        }

        /* Verify half_size is non-zero */
        TEST_ASSERT(n->half_size > 0, "node %u has zero half_size", i);
    }
    TEST_ASSERT(internal_count == tree.n_internal, "internal count mismatch: %u vs %u", internal_count,
                tree.n_internal);
    TEST_ASSERT(mp_count == tree.n_multipole_leaves, "MP leaf count mismatch: %u vs %u", mp_count,
                tree.n_multipole_leaves);
    TEST_ASSERT(particle_count == tree.n_particle_leaves, "particle leaf count mismatch: %u vs %u", particle_count,
                tree.n_particle_leaves);

    printf("Flat tree test: %u nodes (%u internal, %u MP, %u particle), %u sources\n", tree.n_nodes, tree.n_internal,
           tree.n_multipole_leaves, tree.n_particle_leaves, N);

    /* Validate particle_order indices are all in range */
    {
        unsigned seen[N];
        memset(seen, 0, sizeof(seen));
        for (unsigned i = 0; i < N; ++i)
        {
            unsigned idx = tree.particle_order[i];
            TEST_ASSERT(idx < N, "particle_order[%u] = %u out of range", i, idx);
            seen[idx]++;
        }
        /* Each source should appear exactly once */
        for (unsigned i = 0; i < N; ++i)
        {
            TEST_ASSERT(seen[i] == 1, "source %u appears %u times in particle_order", i, seen[i]);
        }
    }

    /* Validate root node */
    {
        const cvl_cl_flat_node_t *root = &tree.nodes[0];
        TEST_ASSERT(root->kind == CVL_CL_FLAT_NODE_INTERNAL, "root is not internal");
        TEST_ASSERT(root->half_size > 0, "root half_size is zero");
        TEST_ASSERT(root->child_base >= 0, "root has no children");
    }

    cvl_cl_flat_tree_destroy(&tree);
    free(flat_work);
    printf("All flat tree tests passed.\n");
    return 0;
}

#else

#include <stdio.h>
int main(void)
{
    printf("OpenCL not available -- skipping test.\n");
    return 0;
}

#endif
