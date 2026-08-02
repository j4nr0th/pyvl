/*
 * test_cvl_cl_bh_eval.c — End-to-end test of the BH flat-tree
 * GPU evaluation kernel.
 *
 * 1. Build a flat octree from random particles (CPU side)
 * 2. Upload tree + particle data to GPU
 * 3. Compile + launch bh_flat_eval kernel (order=0, theta=1e-15)
 * 4. Read results back and compare with CPU direct sum
 *
 * With order=0 and a tiny theta the opening-angle MAC never accepts
 * (half_size/dist ≫ 1e-15 for any node), so the kernel traverses
 * the full tree to every leaf and does direct particle summation,
 * which must match the host-side direct sum.
 */

#ifdef CVL_OPENCL

#include "../test_common.h"
#include "cvl_cl.h"
#include "cvl_cl_compute.h"
#include "cvl_cl_flat_tree.h"
#include "cvl_cl_staging_buffer.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/*  Self-contained OpenCL C kernel source                              */
/* ------------------------------------------------------------------ */
/*
 * The preamble defines real_t/real3_t with FP32/FP64 switching
 * (matching cvl_cl_program_create's -DCVL_CL_REAL_FP32 flag).
 * The kernel body is the same as bh_flat_eval.cl.h but without
 * the #ifdef __OPENCL_C_VERSION__ guard (always active in OpenCL C).
 */
static const char *KERNEL_SOURCE =
    "#ifdef CVL_CL_REAL_FP32\n"
    "typedef float real_t;\n"
    "typedef struct { real_t x, y, z; } real3_t;\n"
    "#else\n"
    "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
    "typedef double real_t;\n"
    "typedef struct { real_t x, y, z; } real3_t;\n"
    "#endif\n"
    "\n"
    /* ---- bh_node_kind_t ---- */
    "typedef enum { BH_NODE_INTERNAL=0, BH_NODE_PARTICLE=1, BH_NODE_MULTIPOLE=2, } bh_node_kind_t;\n"
    "\n"
    /* ---- bh_flat_node_t (64 bytes, layout matches cvl_cl_flat_node_t) ---- */
    "typedef struct {\n"
    "    real3_t center;\n"
    "    real_t  half_size;\n"
    "    ulong   morton_code;\n"
    "    int     child_base;\n"
    "    int     particle_begin;\n"
    "    uchar   child_mask;\n"
    "    uchar   kind;\n"
    "    short   particle_count;\n"
    "    uchar   pad[12];\n"
    "} bh_flat_node_t;\n"
    "\n"
    /* ---- bh_mac_accept ---- */
    "static inline bool bh_mac_accept(bh_flat_node_t node, real3_t point, real_t theta)\n"
    "{\n"
    "    real3_t diff;\n"
    "    diff.x = point.x - node.center.x;\n"
    "    diff.y = point.y - node.center.y;\n"
    "    diff.z = point.z - node.center.z;\n"
    "    if (theta <= (real_t)0.0)\n"
    "        return fabs(diff.x) > (real_t)2.0 * node.half_size ||\n"
    "               fabs(diff.y) > (real_t)2.0 * node.half_size ||\n"
    "               fabs(diff.z) > (real_t)2.0 * node.half_size;\n"
    "    real_t dist = sqrt(diff.x*diff.x + diff.y*diff.y + diff.z*diff.z);\n"
    "    if (dist < (real_t)1e-30) return false;\n"
    "    return node.half_size / dist < theta;\n"
    "}\n"
    "\n"
    /* ---- bh_child_index ---- */
    "static inline int bh_child_index(bh_flat_node_t node, unsigned int octant)\n"
    "{\n"
    "    uchar mask_below = node.child_mask & ((uchar)(1 << octant) - (uchar)1);\n"
    "    return (int)popcount(mask_below);\n"
    "}\n"
    "\n"
    /* ---- bh_flat_eval kernel ---- */
    "__kernel void bh_flat_eval(\n"
    "    __global const bh_flat_node_t *nodes,\n"
    "    __global const unsigned int   *particle_order,\n"
    "    __global const unsigned int   *depth_offsets,\n"
    "    __global const real_t         *sources_pos,\n"
    "    __global const real_t         *sources_val,\n"
    "    unsigned int                   n_targets,\n"
    "    unsigned int                   order,\n"
    "    real_t                         theta,\n"
    "    __global const real_t         *coeffs,\n"
    "    __global real_t               *results)\n"
    "{\n"
    "    unsigned int tid = get_global_id(0);\n"
    "    if (tid >= n_targets) return;\n"
    "    real3_t point;\n"
    "    point.x = sources_pos[3u * tid];\n"
    "    point.y = sources_pos[3u * tid + 1u];\n"
    "    point.z = sources_pos[3u * tid + 2u];\n"
    "    real3_t acc = {0, 0, 0};\n"
    "    enum { BH_STACK_MAX = 128 };\n"
    "    int stack[BH_STACK_MAX];\n"
    "    int sp = 0;\n"
    "    stack[sp++] = 0;\n"
    "    while (sp > 0)\n"
    "    {\n"
    "        sp -= 1;\n"
    "        int ni = stack[sp];\n"
    "        bh_flat_node_t node = nodes[ni];\n"
    "        if (node.kind == BH_NODE_INTERNAL)\n"
    "        {\n"
    "            if (bh_mac_accept(node, point, theta))\n"
    "            {\n"
    "                if (order > 0)\n"
    "                {\n"
    "                    size_t nc = (size_t)(order+1)*(order+2)*(order+3)*(order+4)/24u;\n"
    "                    size_t base = (size_t)ni * 3u * nc;\n"
    "                    real3_t r_rel;\n"
    "                    r_rel.x = point.x - node.center.x;\n"
    "                    r_rel.y = point.y - node.center.y;\n"
    "                    r_rel.z = point.z - node.center.z;\n"
    "                    real_t rr = r_rel.x*r_rel.x + r_rel.y*r_rel.y + r_rel.z*r_rel.z;\n"
    "                    if (rr > (real_t)1e-30)\n"
    "                    {\n"
    "                        real_t inv_r  = (real_t)1.0 / sqrt(rr);\n"
    "                        real_t inv_r2 = inv_r * inv_r;\n"
    "                        real_t scale  = inv_r2;\n"
    "                        size_t idx    = 0;\n"
    "                        for (unsigned int m = 0; m <= order; ++m)\n"
    "                        {\n"
    "                            real3_t term_m = {0, 0, 0};\n"
    "                            real_t  px     = (real_t)1.0;\n"
    "                            for (unsigned int p = 0; p <= m; ++p)\n"
    "                            {\n"
    "                                real_t py = px;\n"
    "                                for (unsigned int q = 0; q <= m - p; ++q)\n"
    "                                {\n"
    "                                    real_t pz = py;\n"
    "                                    for (unsigned int r = 0; r <= m - p - q; ++r)\n"
    "                                    {\n"
    "                                        term_m.x += coeffs[base + 0u*nc + idx] * pz;\n"
    "                                        term_m.y += coeffs[base + 1u*nc + idx] * pz;\n"
    "                                        term_m.z += coeffs[base + 2u*nc + idx] * pz;\n"
    "                                        pz *= r_rel.z;\n"
    "                                        idx++;\n"
    "                                    }\n"
    "                                    py *= r_rel.y;\n"
    "                                }\n"
    "                                px *= r_rel.x;\n"
    "                            }\n"
    "                            acc.x += term_m.x * scale;\n"
    "                            acc.y += term_m.y * scale;\n"
    "                            acc.z += term_m.z * scale;\n"
    "                            scale *= inv_r2;\n"
    "                        }\n"
    "                    }\n"
    "                }\n"
    "            }\n"
    "            else\n"
    "            {\n"
    "                for (int oct = 7; oct >= 0; --oct)\n"
    "                    if (node.child_mask & (uchar)(1 << oct))\n"
    "                    {\n"
    "                        int ci = bh_child_index(node, (unsigned int)oct);\n"
    "                        int cn = node.child_base + ci;\n"
    "                        if (cn >= 0) stack[sp++] = cn;\n"
    "                    }\n"
    "            }\n"
    "        }\n"
    "        else\n"
    "        {\n"
    "            if (node.kind == BH_NODE_MULTIPOLE && order > 0)\n"
    "            {\n"
    "                if (bh_mac_accept(node, point, theta))\n"
    "                {\n"
    "                    size_t nc = (size_t)(order+1)*(order+2)*(order+3)*(order+4)/24u;\n"
    "                    size_t base = (size_t)ni * 3u * nc;\n"
    "                    real3_t r_rel;\n"
    "                    r_rel.x = point.x - node.center.x;\n"
    "                    r_rel.y = point.y - node.center.y;\n"
    "                    r_rel.z = point.z - node.center.z;\n"
    "                    real_t rr = r_rel.x*r_rel.x + r_rel.y*r_rel.y + r_rel.z*r_rel.z;\n"
    "                    if (rr > (real_t)1e-30)\n"
    "                    {\n"
    "                        real_t inv_r  = (real_t)1.0 / sqrt(rr);\n"
    "                        real_t inv_r2 = inv_r * inv_r;\n"
    "                        real_t scale  = inv_r2;\n"
    "                        size_t idx    = 0;\n"
    "                        for (unsigned int m = 0; m <= order; ++m)\n"
    "                        {\n"
    "                            real3_t term_m = {0, 0, 0};\n"
    "                            real_t  px     = (real_t)1.0;\n"
    "                            for (unsigned int p = 0; p <= m; ++p)\n"
    "                            {\n"
    "                                real_t py = px;\n"
    "                                for (unsigned int q = 0; q <= m - p; ++q)\n"
    "                                {\n"
    "                                    real_t pz = py;\n"
    "                                    for (unsigned int r = 0; r <= m - p - q; ++r)\n"
    "                                    {\n"
    "                                        term_m.x += coeffs[base + 0u*nc + idx] * pz;\n"
    "                                        term_m.y += coeffs[base + 1u*nc + idx] * pz;\n"
    "                                        term_m.z += coeffs[base + 2u*nc + idx] * pz;\n"
    "                                        pz *= r_rel.z;\n"
    "                                        idx++;\n"
    "                                    }\n"
    "                                    py *= r_rel.y;\n"
    "                                }\n"
    "                                px *= r_rel.x;\n"
    "                            }\n"
    "                            acc.x += term_m.x * scale;\n"
    "                            acc.y += term_m.y * scale;\n"
    "                            acc.z += term_m.z * scale;\n"
    "                            scale *= inv_r2;\n"
    "                        }\n"
    "                    }\n"
    "                }\n"
    "                continue;\n"
    "            }\n"
    "            int begin = node.particle_begin;\n"
    "            int count = node.particle_count;\n"
    "            for (int k = 0; k < count; ++k)\n"
    "            {\n"
    "                unsigned int src = particle_order[begin + k];\n"
    "                real3_t dr;\n"
    "                dr.x = point.x - sources_pos[3u * src];\n"
    "                dr.y = point.y - sources_pos[3u * src + 1u];\n"
    "                dr.z = point.z - sources_pos[3u * src + 2u];\n"
    "                real_t r2 = dr.x*dr.x + dr.y*dr.y + dr.z*dr.z;\n"
    "                if (r2 > (real_t)1e-30)\n"
    "                {\n"
    "                    real_t inv_r2 = (real_t)1.0 / r2;\n"
    "                    acc.x += sources_val[3u * src]     * inv_r2;\n"
    "                    acc.y += sources_val[3u * src + 1u] * inv_r2;\n"
    "                    acc.z += sources_val[3u * src + 2u] * inv_r2;\n"
    "                }\n"
    "            }\n"
    "        }\n"
    "    }\n"
    "    results[3u * tid]       = acc.x;\n"
    "    results[3u * tid + 1u]  = acc.y;\n"
    "    results[3u * tid + 2u]  = acc.z;\n"
    "}\n";

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
    cvl_cl_status_t status = CVL_CL_SUCCESS;
    cvl_cl_device_t device = {0};
    cvl_cl_ctx_t ctx = {0};
    cvl_cl_queue_t queue = {0};
    cvl_cl_compute_t comp = {0};
    int ret = 1;

    /* Flat tree data */
    cvl_cl_flat_tree_t tree = {0};
    /* Work buffer for the flat tree build (allocated before build, freed in cleanup). */
    void *flat_work = NULL;

    /* GPU buffers for tree data (non-real3_t, managed with raw cvl_cl_buffer_t) */
    cvl_cl_buffer_t buf_nodes = {0};
    cvl_cl_buffer_t buf_order = {0};
    cvl_cl_buffer_t buf_depth = {0};
    cvl_cl_buffer_t buf_coeffs = {0};

    /* Staging buffers for sources and results */
    cvl_cl_staging_buffer_t buf_src_pos = {0};
    cvl_cl_staging_buffer_t buf_src_val = {0};
    cvl_cl_staging_buffer_t buf_results = {0};

    /* Test parameters */
    uint64_t rng = 12345;
#define N_SOURCES 100
#define N_TARGETS 10
#define MAX_DEPTH 6
#define CRIT 8

    /* ----------------------------------------------------------------- */
    /* 1. Device discovery                                               */
    /* ----------------------------------------------------------------- */
    {
        status = cvl_cl_device_first_gpu(&device);
        if (status != CVL_CL_SUCCESS)
        {
            status = cvl_cl_device_first_cpu(&device);
        }
        if (status != CVL_CL_SUCCESS)
        {
            fprintf(stderr, "No OpenCL device found -- skipping test.\n");
            return 0;
        }
    }

    CVL_CL_CHECK(cvl_cl_ctx_create(&device, &ctx), cleanup);
    CVL_CL_CHECK(cvl_cl_queue_create(&ctx, NULL, &queue), cleanup);

    /* ----------------------------------------------------------------- */
    /* 2. Compute backend: compile kernel                                */
    /* ----------------------------------------------------------------- */
    {
        const char *kernels[] = {"bh_flat_eval"};
        CVL_CL_CHECK(
            cvl_cl_compute_init(&comp, &ctx, &queue, &device, CVL_CL_PRECISION_FP64, KERNEL_SOURCE, kernels, 1),
            cleanup);
    }

    /* ----------------------------------------------------------------- */
    /* 3. Generate test data, build flat tree, upload to GPU             */
    /* ----------------------------------------------------------------- */
    {
        /* ---- Generate source positions + values ---- */
        real3_t sources[N_SOURCES];
        real3_t values[N_SOURCES];
        for (unsigned i = 0; i < N_SOURCES; ++i)
        {
            sources[i].x = xorshift_uniform_range(&rng, -5.0, 5.0);
            sources[i].y = xorshift_uniform_range(&rng, -5.0, 5.0);
            sources[i].z = xorshift_uniform_range(&rng, -5.0, 5.0);
            values[i].x = xorshift_uniform_range(&rng, -1.0, 1.0);
            values[i].y = xorshift_uniform_range(&rng, -1.0, 1.0);
            values[i].z = xorshift_uniform_range(&rng, -1.0, 1.0);
        }

        /* ---- Compute bounding box ---- */
        real3_t bbox_min = sources[0], bbox_max = sources[0];
        for (unsigned i = 1; i < N_SOURCES; ++i)
        {
            if (sources[i].x < bbox_min.x)
                bbox_min.x = sources[i].x;
            if (sources[i].y < bbox_min.y)
                bbox_min.y = sources[i].y;
            if (sources[i].z < bbox_min.z)
                bbox_min.z = sources[i].z;
            if (sources[i].x > bbox_max.x)
                bbox_max.x = sources[i].x;
            if (sources[i].y > bbox_max.y)
                bbox_max.y = sources[i].y;
            if (sources[i].z > bbox_max.z)
                bbox_max.z = sources[i].z;
        }
        real3_t root_center = {(bbox_min.x + bbox_max.x) * 0.5, (bbox_min.y + bbox_max.y) * 0.5,
                               (bbox_min.z + bbox_max.z) * 0.5};
        real_t root_hs =
            (real_t)fmax(fmax(bbox_max.x - bbox_min.x, bbox_max.y - bbox_min.y), bbox_max.z - bbox_min.z) * 0.5 +
            (real_t)1e-12;

        /* ---- Compute + sort Morton codes ---- */
        morton_entry_t entries[N_SOURCES];
        for (unsigned i = 0; i < N_SOURCES; ++i)
            entries[i] = (morton_entry_t){.code = morton_3d(sources[i], root_center, root_hs), .idx = i};
        qsort(entries, N_SOURCES, sizeof(morton_entry_t), morton_cmp);

        uint64_t mcodes[N_SOURCES];
        unsigned sorted_indices[N_SOURCES];
        for (unsigned i = 0; i < N_SOURCES; ++i)
        {
            mcodes[i] = entries[i].code;
            sorted_indices[i] = entries[i].idx;
        }

        /* ---- Build flat tree ---- */
        cvl_cl_flat_tree_settings_t settings = {
            .max_depth = MAX_DEPTH,
            .critical_particle_count = CRIT,
            .order = 0,
        };

        /* Count nodes first to size the build work buffer. */
        unsigned depth_counts[CVL_CL_FLAT_TREE_MAX_DEPTH + 2];
        unsigned n_total_work = 0, max_depth_used = 0;
        status = cvl_cl_flat_tree_count(N_SOURCES, mcodes, &settings, depth_counts, &n_total_work, &max_depth_used);
        TEST_ASSERT(status == CVL_CL_SUCCESS, "flat_tree_count failed: %s", cvl_cl_status_str(status));

        const unsigned work_depth = MAX_DEPTH > CVL_CL_FLAT_TREE_MAX_DEPTH ? CVL_CL_FLAT_TREE_MAX_DEPTH : MAX_DEPTH;
        const size_t flat_work_sz = cvl_cl_flat_tree_work_size(n_total_work, N_SOURCES, work_depth);
        flat_work = malloc(flat_work_sz);
        TEST_ASSERT(flat_work != NULL, "malloc(%zu) for flat tree work buffer failed", flat_work_sz);

        status = cvl_cl_flat_tree_build(N_SOURCES, sources, sorted_indices, mcodes, &settings, NULL, &tree, flat_work,
                                        flat_work_sz);
        TEST_ASSERT(status == CVL_CL_SUCCESS, "flat_tree_build failed: %s", cvl_cl_status_str(status));

        printf("BH eval test: %u nodes, %u sources, %u targets\n", tree.n_nodes, N_SOURCES, N_TARGETS);

        /* ---- Init staging buffers (with correct unified_memory flag) ---- */
        bool unified = cvl_cl_compute_unified_memory(&comp);
        CVL_CL_CHECK(cvl_cl_staging_buffer_init(&buf_src_pos, CVL_CL_PRECISION_FP64, unified, N_SOURCES, NULL),
                     cleanup);
        CVL_CL_CHECK(cvl_cl_staging_buffer_init(&buf_src_val, CVL_CL_PRECISION_FP64, unified, N_SOURCES, NULL),
                     cleanup);
        CVL_CL_CHECK(cvl_cl_staging_buffer_init(&buf_results, CVL_CL_PRECISION_FP64, unified, N_TARGETS, NULL),
                     cleanup);

        CVL_CL_CHECK(cvl_cl_staging_buffer_reserve(&buf_src_pos, &ctx, &queue, N_SOURCES), cleanup);
        CVL_CL_CHECK(cvl_cl_staging_buffer_reserve(&buf_src_val, &ctx, &queue, N_SOURCES), cleanup);
        CVL_CL_CHECK(cvl_cl_staging_buffer_reserve(&buf_results, &ctx, &queue, N_TARGETS), cleanup);

        /* ---- Upload sources via staging buffers ---- */
        {
            cvl_cl_future_t f[2];
            CVL_CL_CHECK(cvl_cl_staging_buffer_write_async(&buf_src_pos, &queue, sources, N_SOURCES, 0, &f[0]),
                         cleanup);
            CVL_CL_CHECK(cvl_cl_staging_buffer_write_async(&buf_src_val, &queue, values, N_SOURCES, 0, &f[1]), cleanup);
            for (int i = 0; i < 2; ++i)
                CVL_CL_CHECK(cvl_cl_future_wait(&f[i]), cleanup);
        }

        /* ---- Upload tree data via raw buffers ---- */
        size_t nodes_bytes = tree.n_nodes * sizeof(cvl_cl_flat_node_t);
        size_t order_bytes = N_SOURCES * sizeof(unsigned);
        size_t depth_bytes = (MAX_DEPTH + 2) * sizeof(unsigned);

        CVL_CL_CHECK(
            cvl_cl_buffer_create(
                &ctx, &(cvl_cl_buffer_desc_t){.access = CVL_CL_BUF_READ_ONLY, .size_bytes = nodes_bytes}, &buf_nodes),
            cleanup);
        CVL_CL_CHECK(
            cvl_cl_buffer_create(
                &ctx, &(cvl_cl_buffer_desc_t){.access = CVL_CL_BUF_READ_ONLY, .size_bytes = order_bytes}, &buf_order),
            cleanup);
        CVL_CL_CHECK(
            cvl_cl_buffer_create(
                &ctx, &(cvl_cl_buffer_desc_t){.access = CVL_CL_BUF_READ_ONLY, .size_bytes = depth_bytes}, &buf_depth),
            cleanup);
        /* Dummy coeffs buffer (never accessed with order=0) */
        CVL_CL_CHECK(cvl_cl_buffer_create(
                         &ctx, &(cvl_cl_buffer_desc_t){.access = CVL_CL_BUF_READ_ONLY, .size_bytes = 1}, &buf_coeffs),
                     cleanup);

        CVL_CL_CHECK(cvl_cl_write_buffer(&queue, &buf_nodes, 0, nodes_bytes, tree.nodes, 0, NULL, NULL), cleanup);
        CVL_CL_CHECK(cvl_cl_write_buffer(&queue, &buf_order, 0, order_bytes, tree.particle_order, 0, NULL, NULL),
                     cleanup);
        CVL_CL_CHECK(cvl_cl_write_buffer(&queue, &buf_depth, 0, depth_bytes, tree.depth_offsets, 0, NULL, NULL),
                     cleanup);
    }

    /* ----------------------------------------------------------------- */
    /* 4. Launch kernel                                                  */
    /* ----------------------------------------------------------------- */
    {
        cvl_cl_kernel_t *k = cvl_cl_compute_kernel(&comp, "bh_flat_eval");
        TEST_ASSERT(k != NULL, "kernel 'bh_flat_eval' not found");

        CVL_CL_CHECK(cvl_cl_kernel_set_args(
                         k,
                         (cvl_cl_karg_t[]){
                             {.type = CVL_CL_KARG_BUFFER, .index = 0, .mem = cvl_cl_buffer_mem(&buf_nodes)},
                             {.type = CVL_CL_KARG_BUFFER, .index = 1, .mem = cvl_cl_buffer_mem(&buf_order)},
                             {.type = CVL_CL_KARG_BUFFER, .index = 2, .mem = cvl_cl_buffer_mem(&buf_depth)},
                             {.type = CVL_CL_KARG_BUFFER, .index = 3, .mem = cvl_cl_buffer_mem(&buf_src_pos.device)},
                             {.type = CVL_CL_KARG_BUFFER, .index = 4, .mem = cvl_cl_buffer_mem(&buf_src_val.device)},
                             {.type = CVL_CL_KARG_SCALAR_UINT, .index = 5, .scalar_uint = N_TARGETS},
                             {.type = CVL_CL_KARG_SCALAR_UINT, .index = 6, .scalar_uint = 0}, /* order = 0 */
                             {.type = CVL_CL_KARG_SCALAR_DOUBLE,
                              .index = 7,
                              .scalar_double = 1e-15}, /* tiny theta forces full descent */
                             {.type = CVL_CL_KARG_BUFFER, .index = 8, .mem = cvl_cl_buffer_mem(&buf_coeffs)},
                             {.type = CVL_CL_KARG_BUFFER, .index = 9, .mem = cvl_cl_buffer_mem(&buf_results.device)},
                             {},
                         }),
                     cleanup);

        const size_t global = N_TARGETS;
        CVL_CL_CHECK(cvl_cl_ndrange(&queue, k, 1, &global, NULL, NULL, 0, NULL, NULL), cleanup);
    }

    /* ----------------------------------------------------------------- */
    /* 5. Read results back                                              */
    /* ----------------------------------------------------------------- */
    real3_t gpu_results[N_TARGETS];
    memset(gpu_results, 0, sizeof(gpu_results));
    CVL_CL_CHECK(cvl_cl_staging_buffer_read_and_wait(&buf_results, &queue, gpu_results, N_TARGETS, 0), cleanup);

    /* ----------------------------------------------------------------- */
    /* 6. CPU reference direct sum + comparison                          */
    /* ----------------------------------------------------------------- */
    {
        /* Regenerate same data from the same RNG seed */
        uint64_t rng_ref = 12345;
        real3_t ref_sources[N_SOURCES];
        real3_t ref_values[N_SOURCES];
        for (unsigned i = 0; i < N_SOURCES; ++i)
        {
            ref_sources[i].x = xorshift_uniform_range(&rng_ref, -5.0, 5.0);
            ref_sources[i].y = xorshift_uniform_range(&rng_ref, -5.0, 5.0);
            ref_sources[i].z = xorshift_uniform_range(&rng_ref, -5.0, 5.0);
            ref_values[i].x = xorshift_uniform_range(&rng_ref, -1.0, 1.0);
            ref_values[i].y = xorshift_uniform_range(&rng_ref, -1.0, 1.0);
            ref_values[i].z = xorshift_uniform_range(&rng_ref, -1.0, 1.0);
        }

        real3_t cpu_results[N_TARGETS];
        memset(cpu_results, 0, sizeof(cpu_results));

        for (unsigned t = 0; t < N_TARGETS; ++t)
        {
            real3_t acc = {0, 0, 0};
            for (unsigned s = 0; s < N_SOURCES; ++s)
            {
                real3_t dr = real3_sub(ref_sources[t], ref_sources[s]);
                acc = real3_add(acc, particle_kernel(ref_values[s], dr));
            }
            cpu_results[t] = acc;
        }

        /* Compare GPU vs CPU */
        double max_abs_err = 0.0, max_rel_err = 0.0;
        unsigned max_err_idx = 0;
        for (unsigned t = 0; t < N_TARGETS; ++t)
        {
            double dx = fabs(gpu_results[t].x - cpu_results[t].x);
            double dy = fabs(gpu_results[t].y - cpu_results[t].y);
            double dz = fabs(gpu_results[t].z - cpu_results[t].z);
            double abs_err = dx > dy ? (dx > dz ? dx : dz) : (dy > dz ? dy : dz);
            double ref = fabs(cpu_results[t].x) > fabs(cpu_results[t].y)
                             ? (fabs(cpu_results[t].x) > fabs(cpu_results[t].z) ? fabs(cpu_results[t].x)
                                                                                : fabs(cpu_results[t].z))
                             : (fabs(cpu_results[t].y) > fabs(cpu_results[t].z) ? fabs(cpu_results[t].y)
                                                                                : fabs(cpu_results[t].z));
            double rel_err = abs_err / (ref + 1e-30);
            if (abs_err > max_abs_err)
            {
                max_abs_err = abs_err;
                max_rel_err = rel_err;
                max_err_idx = t;
            }
        }

        printf("BH eval: max_abs_err=%.2e, max_rel_err=%.2e\n", max_abs_err, max_rel_err);
        TEST_ASSERT(max_abs_err < 1e-12 || max_rel_err < 1e-10,
                    "GPU BH eval mismatch: t=%u max_abs_err=%.2e max_rel_err=%.2e", max_err_idx, max_abs_err,
                    max_rel_err);
    }

    printf("All BH eval tests passed.\n");
    ret = 0;

cleanup:
    cvl_cl_staging_buffer_destroy(&buf_results);
    cvl_cl_staging_buffer_destroy(&buf_src_val);
    cvl_cl_staging_buffer_destroy(&buf_src_pos);
    cvl_cl_buffer_destroy(&buf_coeffs);
    cvl_cl_buffer_destroy(&buf_depth);
    cvl_cl_buffer_destroy(&buf_order);
    cvl_cl_buffer_destroy(&buf_nodes);
    cvl_cl_compute_destroy(&comp);
    cvl_cl_flat_tree_destroy(&tree);
    free(flat_work);
    cvl_cl_queue_destroy(&queue);
    cvl_cl_ctx_destroy(&ctx);
    return ret;
}

#else

#include <stdio.h>
int main(void)
{
    printf("OpenCL not available -- skipping test.\n");
    return 0;
}

#endif
