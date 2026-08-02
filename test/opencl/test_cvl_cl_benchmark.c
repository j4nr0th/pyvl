/*
 * test_cvl_cl_benchmark.c — Comprehensive benchmark + accuracy
 * comparison of three OpenCL evaluation methods (direct sum, BH,
 * FMM) across two precision modes (FP32, FP64).
 *
 * Pipeline:
 *   1. Generate N_SOURCES random particles + N_TARGETS random targets.
 *   2. Build a BH tree and an FMM tree on the CPU.
 *   3. Compute the CPU reference (direct sum) for accuracy comparison.
 *   4. For each method × precision:
 *        a. GPU direct sum (cvl_cl_compute + staging buffers + direct_sum)
 *        b. GPU BH eval (cvl_cl_compute + bh_flat_eval kernel, using a
 *           CPU-built flat tree from cvl_cl_flat_tree_build).
 *        c. GPU FMM L2P eval (cvl_cl_fmm_eval_run, using the CPU-built
 *           FMM tree).
 *   5. Measure GPU time for each (upload + kernel + download).
 *   6. Compare accuracy vs CPU direct sum.
 *   7. Print a summary table.
 *
 * Three separate compute backends are used (comp_direct, comp_bh,
 * comp_fmm) to avoid struct-name conflicts between the BH flat node
 * and the FMM L2P node types.
 *
 * The test skips gracefully when no OpenCL device is available and
 * returns 0; it returns 1 on failure.
 */

#ifdef CVL_OPENCL

#include "../../src/core/barnes_hut_tree.h"
#include "../../src/core/fmm_tree.h"
#include "../../src/core/octree.h"
#include "../test_common.h"
#include "cvl_cl.h"
#include "cvl_cl_compute.h"
#include "cvl_cl_flat_tree.h"
#include "cvl_cl_fmm_eval.h"
#include "cvl_cl_gpu_tree_build.h"
#include "cvl_cl_staging_buffer.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ------------------------------------------------------------------ */
/*  Test parameters                                                    */
/* ------------------------------------------------------------------ */

enum
{
    N_SOURCES = 2000,
    N_TARGETS = 200,
    BH_ORDER = 4,
    FMM_ORDER = 4,
    MAX_DEPTH = 8,
    CRIT_COUNT = 8,
};
#define BH_THETA 0.3

/* ------------------------------------------------------------------ */
/*  Direct-sum kernel source (embedded, same as test_cvl_cl_direct_sum) */
/* ------------------------------------------------------------------ */

static const char *DIRECT_SUM_SOURCE = "#ifdef CVL_CL_REAL_FP32\n"
                                       "typedef float real_t;\n"
                                       "#else\n"
                                       "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
                                       "typedef double real_t;\n"
                                       "#endif\n"
                                       "typedef struct { real_t x, y, z; } ds_real3_t;\n"
                                       "__kernel void direct_sum(\n"
                                       "    __global const real_t *targets,\n"
                                       "    __global const real_t *sources_pos,\n"
                                       "    __global const real_t *sources_val,\n"
                                       "    unsigned n_sources,\n"
                                       "    unsigned n_targets,\n"
                                       "    __global real_t *results\n"
                                       ") {\n"
                                       "    unsigned tid = get_global_id(0);\n"
                                       "    if (tid >= n_targets) return;\n"
                                       "    ds_real3_t pt;\n"
                                       "    pt.x = targets[3*tid]; pt.y = targets[3*tid+1]; pt.z = targets[3*tid+2];\n"
                                       "    ds_real3_t acc = {0, 0, 0};\n"
                                       "    for (unsigned i = 0; i < n_sources; ++i) {\n"
                                       "        ds_real3_t dr;\n"
                                       "        dr.x = pt.x - sources_pos[3*i];\n"
                                       "        dr.y = pt.y - sources_pos[3*i+1];\n"
                                       "        dr.z = pt.z - sources_pos[3*i+2];\n"
                                       "        real_t r2 = dr.x*dr.x + dr.y*dr.y + dr.z*dr.z;\n"
                                       "        if (r2 > (real_t)1e-30) {\n"
                                       "            real_t inv_r2 = (real_t)1.0 / r2;\n"
                                       "            acc.x += sources_val[3*i]   * inv_r2;\n"
                                       "            acc.y += sources_val[3*i+1] * inv_r2;\n"
                                       "            acc.z += sources_val[3*i+2] * inv_r2;\n"
                                       "        }\n"
                                       "    }\n"
                                       "    results[3*tid]   = acc.x;\n"
                                       "    results[3*tid+1] = acc.y;\n"
                                       "    results[3*tid+2] = acc.z;\n"
                                       "}\n";

/* ------------------------------------------------------------------ */
/*  BH flat-eval kernel source (embedded, same as test_cvl_cl_bh_eval) */
/* ------------------------------------------------------------------ */

static const char *BH_FLAT_EVAL_SOURCE =
    "#ifdef CVL_CL_REAL_FP32\n"
    "typedef float real_t;\n"
    "typedef struct { real_t x, y, z; } real3_t;\n"
    "#else\n"
    "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
    "typedef double real_t;\n"
    "typedef struct { real_t x, y, z; } real3_t;\n"
    "#endif\n"
    "\n"
    "typedef enum { BH_NODE_INTERNAL=0, BH_NODE_PARTICLE=1, BH_NODE_MULTIPOLE=2, } bh_node_kind_t;\n"
    "\n"
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
    "static inline int bh_child_index(bh_flat_node_t node, unsigned int octant)\n"
    "{\n"
    "    uchar mask_below = node.child_mask & ((uchar)(1 << octant) - (uchar)1);\n"
    "    return (int)popcount(mask_below);\n"
    "}\n"
    "\n"
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
/*  Helpers for building the concatenated FMM kernel source            */
/* ------------------------------------------------------------------ */

/**
 * @brief Concatenate @p src into @p dst, skipping lines that start
 *        with `#include` or `#pragma once` (directives that OpenCL C
 *        cannot resolve at runtime or that cause warnings).
 *
 * @param dst     Output buffer (NUL-terminated on return).
 * @param dst_cap  Capacity of @p dst (including trailing NUL).
 * @param src     NUL-terminated input string.
 * @return Number of characters written (excluding trailing NUL).
 */
static size_t skip_include_concat(char *dst, size_t dst_cap, const char *src)
{
    size_t pos = 0;
    while (*src && pos < dst_cap - 1)
    {
        const char *nl = strchr(src, '\n');
        size_t line_len = nl ? (size_t)(nl - src + 1) : strlen(src);

        /* Trim leading whitespace to detect directives. */
        const char *trimmed = src;
        while (*trimmed == ' ' || *trimmed == '\t')
            ++trimmed;

        int is_include = (trimmed[0] == '#' && strncmp(trimmed + 1, "include", 7) == 0);
        int is_pragma_once = (trimmed[0] == '#' && strncmp(trimmed + 1, "pragma once", 11) == 0);

        if (!is_include && !is_pragma_once)
        {
            size_t copy = line_len < dst_cap - 1 - pos ? line_len : dst_cap - 1 - pos;
            memcpy(dst + pos, src, copy);
            pos += copy;
        }

        if (!nl)
            break;
        src = nl + 1;
    }
    dst[pos] = '\0';
    return pos;
}

/**
 * @brief Read a .cl.h file into a malloc'd string.
 *
 * The caller must free the returned pointer.
 */
static char *read_cl_source(const char *filename)
{
    char path[1024];
    int n = snprintf(path, sizeof path, "%s/%s", CVL_CL_SOURCE_DIR, filename);
    TEST_ASSERT(n > 0 && (size_t)n < sizeof path, "Path too long for %s", filename);
    return read_file_to_string(path, 65536);
}

/* ------------------------------------------------------------------ */
/*  Preamble — real_t typedef (FP32/FP64 switching)                    */
/* ------------------------------------------------------------------ */

static const char *PREAMBLE = "#ifdef CVL_CL_REAL_FP32\n"
                              "typedef float real_t;\n"
                              "#else\n"
                              "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
                              "typedef double real_t;\n"
                              "#endif\n"
                              "\n"
                              /* uint64_t / size_t are not provided by default
                               * in OpenCL C; cvl_cl_math.h.cl uses them for
                               * Morton codes and fmm_l2p.cl.h uses size_t. */
                              "typedef unsigned long uint64_t;\n"
                              "typedef unsigned long size_t;\n"
                              "\n";

/* Preamble for the GPU tree-build program (section 8b).  The build + eval
 * kernel headers (bh_build.cl.h / bh_flat_eval.cl.h) use real3_t but do NOT
 * define it, and they use the built-in OpenCL C size_t — so this preamble
 * must NOT typedef size_t. */
static const char *BUILD_PREAMBLE = "#ifdef CVL_CL_REAL_FP32\n"
                                    "typedef float real_t;\n"
                                    "typedef struct { real_t x, y, z; } real3_t;\n"
                                    "#else\n"
                                    "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
                                    "typedef double real_t;\n"
                                    "typedef struct { real_t x, y, z; } real3_t;\n"
                                    "#endif\n"
                                    "\n";

/* ------------------------------------------------------------------ */
/*  Host-side Morton 3D helpers                                        */
/*                                                                    */
/*  morton_split_21 / morton_3d are provided by octree.h (included    */
/*  via barnes_hut_tree.h), so we only need the qsort comparator +    */
/*  entry struct here.                                                */
/* ------------------------------------------------------------------ */

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
/*  Timing helper                                                      */
/* ------------------------------------------------------------------ */

/**
 * @brief Return a monotonic timestamp in seconds.
 */
static double now_seconds(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* ------------------------------------------------------------------ */
/*  Accuracy helper                                                    */
/* ------------------------------------------------------------------ */

/**
 * @brief Compact FP32 mirror of cvl_cl_flat_node_t (48 bytes).
 *
 * The embedded bh_flat_eval kernel compiles with real_t=float in FP32
 * mode, so its bh_flat_node_t shrinks from 64 to 48 bytes (center/half_size
 * become floats).  The host flat tree is always 64-byte doubles, so the
 * FP32 path must convert before upload — otherwise the kernel reads
 * garbage field offsets (the FP32 BH previously returned all-NaN).
 */
typedef struct
{
    float center[3];        /* 12 bytes */
    float half_size;        /*  4 bytes */
    uint64_t morton_code;   /*  8 bytes */
    int32_t child_base;     /*  4 bytes */
    int32_t particle_begin; /*  4 bytes */
    uint8_t child_mask;     /*  1 byte  */
    uint8_t kind;           /*  1 byte  */
    int16_t particle_count; /*  2 bytes */
    uint8_t pad[12];        /* 12 bytes → 48 total */
} bh_flat_node_f32_t;

_Static_assert(sizeof(bh_flat_node_f32_t) == 48, "bh_flat_node_f32_t must be 48 bytes");

/**
 * @brief Convert 64-byte double flat nodes to the 48-byte FP32 layout.
 */
static void convert_flat_nodes_f32(const cvl_cl_flat_node_t *src, size_t n_nodes, bh_flat_node_f32_t *dst)
{
    for (size_t i = 0; i < n_nodes; ++i)
    {
        dst[i].center[0] = (float)src[i].center.x;
        dst[i].center[1] = (float)src[i].center.y;
        dst[i].center[2] = (float)src[i].center.z;
        dst[i].half_size = (float)src[i].half_size;
        dst[i].morton_code = src[i].morton_code;
        dst[i].child_base = src[i].child_base;
        dst[i].particle_begin = src[i].particle_begin;
        dst[i].child_mask = src[i].child_mask;
        dst[i].kind = src[i].kind;
        dst[i].particle_count = (int16_t)src[i].particle_count;
    }
}

/**
 * @brief Compute the max relative error between @p gpu and @p ref over
 *        N_TARGETS targets (component-wise max, then relative to the
 *        largest reference component).
 *
 * @param gpu       GPU results [N_TARGETS].
 * @param ref       Reference results [N_TARGETS].
 * @param out_abs   Filled with the max absolute error.
 * @return Max relative error.
 */
static double max_rel_error(const real3_t *gpu, const real3_t *ref, double *out_abs)
{
    double max_abs = 0.0, max_rel = 0.0;
    for (unsigned t = 0; t < N_TARGETS; ++t)
    {
        double dx = fabs((double)gpu[t].x - (double)ref[t].x);
        double dy = fabs((double)gpu[t].y - (double)ref[t].y);
        double dz = fabs((double)gpu[t].z - (double)ref[t].z);
        double abs_err = dx > dy ? (dx > dz ? dx : dz) : (dy > dz ? dy : dz);
        /* NaN results must not be silently ignored (all comparisons with
         * NaN are false, which would report a 0.0 error for garbage). */
        if (isnan(abs_err))
        {
            if (out_abs)
                *out_abs = NAN;
            return NAN;
        }
        double r =
            fabs((double)ref[t].x) > fabs((double)ref[t].y)
                ? (fabs((double)ref[t].x) > fabs((double)ref[t].z) ? fabs((double)ref[t].x) : fabs((double)ref[t].z))
                : (fabs((double)ref[t].y) > fabs((double)ref[t].z) ? fabs((double)ref[t].y) : fabs((double)ref[t].z));
        double rel_err = abs_err / (r + 1e-30);
        if (abs_err > max_abs)
        {
            max_abs = abs_err;
            max_rel = rel_err;
        }
    }
    if (out_abs)
        *out_abs = max_abs;
    return max_rel;
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
    int ret = 1;

    /* Three separate compute backends (one per method) to avoid
     * struct-name conflicts between bh_flat_node_t and the FMM L2P
     * node type.  Each is re-initialised per precision. */
    cvl_cl_compute_t comp_direct = {0};
    cvl_cl_compute_t comp_bh = {0};
    cvl_cl_compute_t comp_fmm = {0};
    cvl_cl_compute_t comp_build = {0};
    cvl_cl_gpu_tree_build_t gpu_builder = {0};
    cvl_cl_staging_buffer_t buf_build_pos = {0};
    cvl_cl_fmm_eval_t fmm_eval = {0};

    /* CPU-built trees. */
    barnes_hut_tree_t bh_tree = {0};
    fmm_tree_t fmm_tree = {0};

    /* Flat BH tree (GPU-friendly layout from cvl_cl_flat_tree_build). */
    cvl_cl_flat_tree_t flat_tree = {0};
    /* Work buffer for the flat tree build (allocated before build, freed in cleanup). */
    void *flat_work = NULL;

    /* GPU buffers for the BH flat tree (raw cvl_cl_buffer_t). */
    cvl_cl_buffer_t buf_bh_nodes = {0};
    cvl_cl_buffer_t buf_bh_order = {0};
    cvl_cl_buffer_t buf_bh_depth = {0};
    cvl_cl_buffer_t buf_bh_coeffs = {0};

    /* Staging buffers shared across methods (re-init per precision). */
    cvl_cl_staging_buffer_t buf_targets = {0};
    cvl_cl_staging_buffer_t buf_src_pos = {0};
    cvl_cl_staging_buffer_t buf_src_val = {0};
    cvl_cl_staging_buffer_t buf_results = {0};

    /* FMM kernel source strings (freed in cleanup). */
    char *src_types = NULL;
    char *src_math = NULL;
    char *src_multipole = NULL;
    char *src_mp_ops = NULL;
    char *src_fmm_ops = NULL;
    char *src_l2p = NULL;
    char *fmm_combined = NULL;

    /* Host data (generated once, reused across all runs). */
    real3_t sources_pos[N_SOURCES];
    real3_t sources_val[N_SOURCES];
    real3_t targets[N_TARGETS];

    /* CPU reference results. */
    real3_t cpu_direct[N_TARGETS];
    real3_t cpu_bh[N_TARGETS];
    real3_t cpu_fmm[N_TARGETS];
    /* CPU references evaluated at the first N_TARGETS SOURCE positions.
     * The bh_flat_eval kernel reads its target point from sources_pos
     * (arg 3), so GPU BH evaluates at source positions — the CPU refs
     * for the BH comparison must use the same points. */
    real3_t cpu_direct_src[N_TARGETS];
    real3_t cpu_bh_src[N_TARGETS];

    /* GPU result scratch. */
    real3_t gpu_results[N_TARGETS];

    /* Timing accumulators (milliseconds). */
    double t_gpu_direct[2] = {0, 0}; /* [FP64, FP32] */
    double t_gpu_bh[2] = {0, 0};
    double t_gpu_fmm[2] = {0, 0};
    double t_gpu_tree_build = 0.0;
    double t_cpu_direct = 0.0;
    double t_cpu_bh = 0.0;
    double t_cpu_fmm = 0.0;
    double t_cpu_bh_build = 0.0;
    double t_cpu_fmm_build = 0.0;

    /* Accuracy (max relative error vs CPU direct). */
    double err_direct[2] = {0, 0};
    double err_bh[2] = {0, 0};
    double err_fmm[2] = {0, 0};

    /* ----------------------------------------------------------------- */
    /* 1. Device discovery (GPU only — CPU OpenCL crashes on complex kernels) */
    /* ----------------------------------------------------------------- */
    {
        unsigned count = 0;
        if (cvl_cl_device_first_gpu(&device) != CVL_CL_SUCCESS)
        {
            fprintf(stderr, "No GPU OpenCL device found -- skipping benchmark (CPU OpenCL not supported).\n");
            return 0;
        }
    }

    CVL_CL_CHECK(cvl_cl_ctx_create(&device, &ctx), cleanup);
    CVL_CL_CHECK(cvl_cl_queue_create(&ctx, NULL, &queue), cleanup);

    /* ----------------------------------------------------------------- */
    /* 2. Generate random sources + targets                              */
    /* ----------------------------------------------------------------- */
    {
        uint64_t rng = 12345;
        for (unsigned i = 0; i < N_SOURCES; ++i)
        {
            sources_pos[i].x = xorshift_uniform_range(&rng, -5.0, 5.0);
            sources_pos[i].y = xorshift_uniform_range(&rng, -5.0, 5.0);
            sources_pos[i].z = xorshift_uniform_range(&rng, -5.0, 5.0);
            sources_val[i].x = xorshift_uniform_range(&rng, -1.0, 1.0);
            sources_val[i].y = xorshift_uniform_range(&rng, -1.0, 1.0);
            sources_val[i].z = xorshift_uniform_range(&rng, -1.0, 1.0);
        }
        /* Targets inside [-4, 4]^3 so FMM mode stays inside the bbox. */
        for (unsigned i = 0; i < N_TARGETS; ++i)
        {
            targets[i].x = xorshift_uniform_range(&rng, -4.0, 4.0);
            targets[i].y = xorshift_uniform_range(&rng, -4.0, 4.0);
            targets[i].z = xorshift_uniform_range(&rng, -4.0, 4.0);
        }
    }

    /* ----------------------------------------------------------------- */
    /* 3. Build CPU BH tree + flat tree                                  */
    /* ----------------------------------------------------------------- */
    {
        const barnes_hut_settings_t settings = {
            .order = BH_ORDER,
            .critical_particle_count = CRIT_COUNT,
            .max_depth = MAX_DEPTH,
            .work_order = BH_ORDER,
            .alpha_centroid = 0.5,
            .theta = BH_THETA,
        };
        const double t0 = now_seconds();
        const bool ok =
            barnes_hut_tree_build(N_SOURCES, 1, sources_pos, sources_val, &settings, &CVL_DEFAULT_ALLOCATOR, &bh_tree);
        t_cpu_bh_build = (now_seconds() - t0) * 1e3;
        TEST_ASSERT(ok, "barnes_hut_tree_build failed");
        printf("BH tree: n_nodes=%u n_internal=%u n_leaves=%u max_depth=%u\n", bh_tree.n_nodes, bh_tree.n_internal,
               bh_tree.n_multipole_leaves + bh_tree.n_particle_leaves, bh_tree.max_depth_reached);
    }

    /* ----------------------------------------------------------------- */
    /* 4. Build CPU FMM tree                                             */
    /* ----------------------------------------------------------------- */
    {
        const fmm_settings_t settings = {
            .order = FMM_ORDER,
            .critical_particle_count = CRIT_COUNT,
            .max_depth = MAX_DEPTH,
            .work_order = FMM_ORDER,
            .alpha_centroid = 0.5,
            .theta = 0.0,
        };
        const double t0 = now_seconds();
        const bool ok =
            fmm_tree_build(N_SOURCES, 1, sources_pos, sources_val, &settings, &CVL_DEFAULT_ALLOCATOR, &fmm_tree);
        t_cpu_fmm_build = (now_seconds() - t0) * 1e3;
        TEST_ASSERT(ok, "fmm_tree_build failed");
        TEST_ASSERT(fmm_tree.local_coeffs != NULL, "FMM tree has no local coefficients");
        printf("FMM tree: n_nodes=%u n_internal=%u n_leaves=%u max_depth=%u\n", fmm_tree.n_nodes, fmm_tree.n_internal,
               fmm_tree.n_leaves, fmm_tree.max_depth_reached);
    }

    /* ----------------------------------------------------------------- */
    /* 5. Build flat BH tree (GPU-friendly layout)                       */
    /* ----------------------------------------------------------------- */
    {
        /* Bounding box. */
        real3_t bbox_min = sources_pos[0], bbox_max = sources_pos[0];
        for (unsigned i = 1; i < N_SOURCES; ++i)
        {
            if (sources_pos[i].x < bbox_min.x)
                bbox_min.x = sources_pos[i].x;
            if (sources_pos[i].y < bbox_min.y)
                bbox_min.y = sources_pos[i].y;
            if (sources_pos[i].z < bbox_min.z)
                bbox_min.z = sources_pos[i].z;
            if (sources_pos[i].x > bbox_max.x)
                bbox_max.x = sources_pos[i].x;
            if (sources_pos[i].y > bbox_max.y)
                bbox_max.y = sources_pos[i].y;
            if (sources_pos[i].z > bbox_max.z)
                bbox_max.z = sources_pos[i].z;
        }
        real3_t root_center = {(bbox_min.x + bbox_max.x) * 0.5, (bbox_min.y + bbox_max.y) * 0.5,
                               (bbox_min.z + bbox_max.z) * 0.5};
        real_t root_hs =
            (real_t)fmax(fmax(bbox_max.x - bbox_min.x, bbox_max.y - bbox_min.y), bbox_max.z - bbox_min.z) * 0.5 +
            (real_t)1e-12;

        /* Morton-sort the particles. */
        morton_entry_t entries[N_SOURCES];
        for (unsigned i = 0; i < N_SOURCES; ++i)
            entries[i] = (morton_entry_t){.code = morton_3d(sources_pos[i], root_center, root_hs), .idx = i};
        qsort(entries, N_SOURCES, sizeof(morton_entry_t), morton_cmp);

        uint64_t mcodes[N_SOURCES];
        unsigned sorted_indices[N_SOURCES];
        for (unsigned i = 0; i < N_SOURCES; ++i)
        {
            mcodes[i] = entries[i].code;
            sorted_indices[i] = entries[i].idx;
        }

        const cvl_cl_flat_tree_settings_t flat_settings = {
            .max_depth = MAX_DEPTH,
            .critical_particle_count = CRIT_COUNT,
            .order = BH_ORDER,
        };

        /* Count nodes first to size the build work buffer. */
        unsigned depth_counts[CVL_CL_FLAT_TREE_MAX_DEPTH + 2];
        unsigned n_total_work = 0, max_depth_used = 0;
        status =
            cvl_cl_flat_tree_count(N_SOURCES, mcodes, &flat_settings, depth_counts, &n_total_work, &max_depth_used);
        TEST_ASSERT(status == CVL_CL_SUCCESS, "flat_tree_count failed: %s", cvl_cl_status_str(status));

        const unsigned work_depth = MAX_DEPTH > CVL_CL_FLAT_TREE_MAX_DEPTH ? CVL_CL_FLAT_TREE_MAX_DEPTH : MAX_DEPTH;
        const size_t flat_work_sz = cvl_cl_flat_tree_work_size(n_total_work, N_SOURCES, work_depth);
        flat_work = malloc(flat_work_sz);
        TEST_ASSERT(flat_work != NULL, "malloc(%zu) for flat tree work buffer failed", flat_work_sz);

        status = cvl_cl_flat_tree_build(N_SOURCES, sources_pos, sorted_indices, mcodes, &flat_settings, NULL,
                                        &flat_tree, flat_work, flat_work_sz);
        TEST_ASSERT(status == CVL_CL_SUCCESS, "cvl_cl_flat_tree_build failed: %s", cvl_cl_status_str(status));
        printf("Flat BH tree: n_nodes=%u\n", flat_tree.n_nodes);
    }

    /* ----------------------------------------------------------------- */
    /* 6. CPU reference: direct sum, BH eval, FMM eval                  */
    /* ----------------------------------------------------------------- */
    {
        const double t0 = now_seconds();
        for (unsigned t = 0; t < N_TARGETS; ++t)
        {
            real3_t acc = {0, 0, 0};
            for (unsigned s = 0; s < N_SOURCES; ++s)
            {
                real3_t dr = real3_sub(targets[t], sources_pos[s]);
                acc = real3_add(acc, particle_kernel(sources_val[s], dr));
            }
            cpu_direct[t] = acc;
        }
        t_cpu_direct = (now_seconds() - t0) * 1e3;
    }
    /* Direct sum evaluated at the first N_TARGETS source positions — the
     * reference for the GPU BH kernel (which evaluates at sources). */
    {
        const double t0 = now_seconds();
        for (unsigned t = 0; t < N_TARGETS; ++t)
        {
            real3_t acc = {0, 0, 0};
            for (unsigned s = 0; s < N_SOURCES; ++s)
            {
                real3_t dr = real3_sub(sources_pos[t], sources_pos[s]);
                acc = real3_add(acc, particle_kernel(sources_val[s], dr));
            }
            cpu_direct_src[t] = acc;
        }
        t_cpu_direct = (now_seconds() - t0) * 1e3;
    }
    {
        const barnes_hut_eval_settings_t eval_settings = {.theta = BH_THETA};
        const double t0 = now_seconds();
        barnes_hut_tree_eval_all(&bh_tree, sources_pos, sources_val, N_TARGETS, targets, cpu_bh, eval_settings, 1);
        /* BH eval at the source positions (matches the GPU kernel). */
        barnes_hut_tree_eval_all(&bh_tree, sources_pos, sources_val, N_TARGETS, sources_pos, cpu_bh_src, eval_settings,
                                 1);
        t_cpu_bh = (now_seconds() - t0) * 1e3;
    }
    {
        const fmm_eval_settings_t eval_settings = {
            .theta = 0.0,
            .mode = FMM_EVAL_FMM,
            .hybrid_alpha = 1.5,
        };
        const double t0 = now_seconds();
        fmm_tree_eval_all(&fmm_tree, sources_pos, sources_val, N_TARGETS, targets, cpu_fmm, eval_settings, 1);
        t_cpu_fmm = (now_seconds() - t0) * 1e3;
    }

    /* CPU accuracy vs direct sum (sanity). */
    {
        double abs_bh, abs_fmm;
        const double rel_bh = max_rel_error(cpu_bh, cpu_direct, &abs_bh);
        const double rel_fmm = max_rel_error(cpu_fmm, cpu_direct, &abs_fmm);
        printf("CPU BH  vs direct: max_abs=%.2e max_rel=%.2e\n", abs_bh, rel_bh);
        printf("CPU FMM vs direct: max_abs=%.2e max_rel=%.2e\n", abs_fmm, rel_fmm);
    }

    /* ----------------------------------------------------------------- */
    /* 7. Read + concatenate the FMM L2P kernel source                   */
    /* ----------------------------------------------------------------- */
    src_types = read_cl_source("cvl_cl_types.h.cl");
    src_math = read_cl_source("cvl_cl_math.h.cl");
    src_multipole = read_cl_source("cvl_cl_multipole.h.cl");
    src_mp_ops = read_cl_source("cvl_cl_multipole_ops.h.cl");
    src_fmm_ops = read_cl_source("cvl_cl_fmm_ops.h.cl");
    src_l2p = read_cl_source("fmm_l2p.cl.h");
    TEST_ASSERT(src_types != NULL, "Failed to read cvl_cl_types.h.cl");
    TEST_ASSERT(src_math != NULL, "Failed to read cvl_cl_math.h.cl");
    TEST_ASSERT(src_multipole != NULL, "Failed to read cvl_cl_multipole.h.cl");
    TEST_ASSERT(src_mp_ops != NULL, "Failed to read cvl_cl_multipole_ops.h.cl");
    TEST_ASSERT(src_fmm_ops != NULL, "Failed to read cvl_cl_fmm_ops.h.cl");
    TEST_ASSERT(src_l2p != NULL, "Failed to read fmm_l2p.cl.h");

    {
        size_t parts[] = {
            strlen(PREAMBLE),   strlen(src_types),   strlen(src_math), strlen(src_multipole),
            strlen(src_mp_ops), strlen(src_fmm_ops), strlen(src_l2p),
        };
        size_t total = 1;
        for (size_t i = 0; i < sizeof(parts) / sizeof(parts[0]); ++i)
            total += parts[i];

        fmm_combined = (char *)malloc(total);
        TEST_ASSERT(fmm_combined != NULL, "malloc failed for FMM kernel source");

        size_t pos = 0;
        memcpy(fmm_combined + pos, PREAMBLE, strlen(PREAMBLE));
        pos += strlen(PREAMBLE);
        pos += skip_include_concat(fmm_combined + pos, total - pos, src_types);
        pos += skip_include_concat(fmm_combined + pos, total - pos, src_math);
        pos += skip_include_concat(fmm_combined + pos, total - pos, src_multipole);
        pos += skip_include_concat(fmm_combined + pos, total - pos, src_mp_ops);
        pos += skip_include_concat(fmm_combined + pos, total - pos, src_fmm_ops);
        pos += skip_include_concat(fmm_combined + pos, total - pos, src_l2p);
        fmm_combined[pos] = '\0';
    }

    free(src_types);
    src_types = NULL;
    free(src_math);
    src_math = NULL;
    free(src_multipole);
    src_multipole = NULL;
    free(src_mp_ops);
    src_mp_ops = NULL;
    free(src_fmm_ops);
    src_fmm_ops = NULL;
    free(src_l2p);
    src_l2p = NULL;

    /* ----------------------------------------------------------------- */
    /* 8. Upload the BH flat tree once (precision-independent layout)    */
    /* ----------------------------------------------------------------- */
    {
        const size_t nodes_bytes = flat_tree.n_nodes * sizeof(cvl_cl_flat_node_t);
        const size_t order_bytes = N_SOURCES * sizeof(unsigned);
        const size_t depth_bytes = (MAX_DEPTH + 2) * sizeof(unsigned);

        CVL_CL_CHECK(cvl_cl_buffer_create(
                         &ctx, &(cvl_cl_buffer_desc_t){.access = CVL_CL_BUF_READ_ONLY, .size_bytes = nodes_bytes},
                         &buf_bh_nodes),
                     cleanup);
        CVL_CL_CHECK(cvl_cl_buffer_create(
                         &ctx, &(cvl_cl_buffer_desc_t){.access = CVL_CL_BUF_READ_ONLY, .size_bytes = order_bytes},
                         &buf_bh_order),
                     cleanup);
        CVL_CL_CHECK(cvl_cl_buffer_create(
                         &ctx, &(cvl_cl_buffer_desc_t){.access = CVL_CL_BUF_READ_ONLY, .size_bytes = depth_bytes},
                         &buf_bh_depth),
                     cleanup);
        /* Dummy coeffs buffer (order>0 path uses CPU multipole coeffs,
         * but the GPU kernel reads from this buffer — allocate enough
         * for n_nodes * 3 * n_coeffs doubles to be safe). */
        {
            const unsigned n_coeffs = (BH_ORDER + 1) * (BH_ORDER + 2) * (BH_ORDER + 3) * (BH_ORDER + 4) / 24u;
            const size_t coeffs_bytes = (size_t)flat_tree.n_nodes * 3u * n_coeffs * sizeof(double);
            CVL_CL_CHECK(cvl_cl_buffer_create(
                             &ctx, &(cvl_cl_buffer_desc_t){.access = CVL_CL_BUF_READ_ONLY, .size_bytes = coeffs_bytes},
                             &buf_bh_coeffs),
                         cleanup);
        }

        CVL_CL_CHECK(cvl_cl_write_buffer(&queue, &buf_bh_nodes, 0, nodes_bytes, flat_tree.nodes, 0, NULL, NULL),
                     cleanup);
        CVL_CL_CHECK(
            cvl_cl_write_buffer(&queue, &buf_bh_order, 0, order_bytes, flat_tree.particle_order, 0, NULL, NULL),
            cleanup);
        CVL_CL_CHECK(cvl_cl_write_buffer(&queue, &buf_bh_depth, 0, depth_bytes, flat_tree.depth_offsets, 0, NULL, NULL),
                     cleanup);
        /* Zero the coeffs buffer — the GPU BH kernel with order>0 reads
         * multipole coeffs from here, but we have not populated them.
         * The flat-tree builder does not compute multipole expansions,
         * so the order>0 multipole path will contribute zero.  The
         * particle-leaf direct-sum path still runs and dominates for
         * small theta.  This is a known limitation: the benchmark
         * measures the GPU traversal cost, not full BH accuracy. */
        {
            const unsigned n_coeffs = (BH_ORDER + 1) * (BH_ORDER + 2) * (BH_ORDER + 3) * (BH_ORDER + 4) / 24u;
            const size_t coeffs_bytes = (size_t)flat_tree.n_nodes * 3u * n_coeffs * sizeof(double);
            /* Use a stack zero buffer for small sizes; fall back to
             * calloc for large ones.  The flat tree is small (≤ a few
             * thousand nodes), so a heap allocation is fine here. */
            double *zero_coeffs = (double *)calloc(coeffs_bytes, 1);
            TEST_ASSERT(zero_coeffs != NULL, "calloc failed for zero coeffs");
            CVL_CL_CHECK(cvl_cl_write_buffer(&queue, &buf_bh_coeffs, 0, coeffs_bytes, zero_coeffs, 0, NULL, NULL),
                         cleanup);
            /* The write is non-blocking — keep zero_coeffs alive until the copy
             * completes.  Freeing it immediately is a use-after-free that
             * crashes the NVIDIA driver's async copy path. */
            CVL_CL_CHECK(cvl_cl_finish(&queue), cleanup);
            free(zero_coeffs);
        }
    }

    /* ----------------------------------------------------------------- */
    /* 8b. GPU tree build (uniform octree on device)                     */
    /* ----------------------------------------------------------------- */
    {
        /* Compile the build kernels + bh_flat_eval into one program. */
        char *build_src = read_cl_source("bh_build.cl.h");
        char *eval_src = read_cl_source("bh_flat_eval.cl.h");
        TEST_ASSERT(build_src != NULL, "Failed to read bh_build.cl.h");
        TEST_ASSERT(eval_src != NULL, "Failed to read bh_flat_eval.cl.h");

        size_t preamble_len = strlen(BUILD_PREAMBLE);
        size_t build_len = strlen(build_src);
        size_t eval_len = strlen(eval_src);
        char *combined_build = (char *)malloc(preamble_len + build_len + eval_len + 1);
        TEST_ASSERT(combined_build != NULL, "malloc for combined build source");
        size_t pos = 0;
        memcpy(combined_build + pos, BUILD_PREAMBLE, preamble_len);
        pos += preamble_len;
        pos += skip_include_concat(combined_build + pos, preamble_len + build_len + eval_len + 1 - pos, build_src);
        pos += skip_include_concat(combined_build + pos, preamble_len + build_len + eval_len + 1 - pos, eval_src);
        combined_build[pos] = '\0';
        free(build_src);
        free(eval_src);

        const char *build_kernels[] = {"kernel_morton",   "kernel_radix_hist",  "kernel_radix_scatter",
                                       "kernel_boundary", "kernel_fill_leaves", "kernel_build_internal",
                                       "bh_flat_eval"};
        CVL_CL_CHECK(cvl_cl_compute_init(&comp_build, &ctx, &queue, &device, CVL_CL_PRECISION_FP64, combined_build,
                                         build_kernels, 7),
                     cleanup_build);
        free(combined_build);
        combined_build = NULL;

        /* Init GPU tree builder. */
        CVL_CL_CHECK(cvl_cl_gpu_tree_build_init(&gpu_builder, &comp_build, MAX_DEPTH, CRIT_COUNT, BH_ORDER),
                     cleanup_build);

        /* Reserve staging buffer for source positions. */
        CVL_CL_CHECK(cvl_cl_staging_buffer_init(&buf_build_pos, CVL_CL_PRECISION_FP64,
                                                cvl_cl_compute_unified_memory(&comp_build), N_SOURCES, NULL),
                     cleanup_build);
        CVL_CL_CHECK(cvl_cl_staging_buffer_reserve(&buf_build_pos, &ctx, &queue, N_SOURCES), cleanup_build);

        /* Upload source positions. */
        {
            cvl_cl_future_t f;
            CVL_CL_CHECK(cvl_cl_staging_buffer_write_async(&buf_build_pos, &queue, sources_pos, N_SOURCES, 0, &f),
                         cleanup_build);
            CVL_CL_CHECK(cvl_cl_future_wait(&f), cleanup_build);
        }
        CVL_CL_CHECK(cvl_cl_finish(&queue), cleanup_build);

        /* Time the GPU tree build. */
        size_t gpu_work_sz = cvl_cl_gpu_tree_build_work_size(N_SOURCES);
        void *gpu_work = malloc(gpu_work_sz);
        TEST_ASSERT(gpu_work != NULL, "malloc(%zu) for GPU tree build work buffer failed", gpu_work_sz);
        const double t0 = now_seconds();
        CVL_CL_CHECK(
            cvl_cl_gpu_tree_build_run(&gpu_builder, &queue, &ctx, &buf_build_pos, N_SOURCES, gpu_work, gpu_work_sz),
            cleanup_build);
        CVL_CL_CHECK(cvl_cl_finish(&queue), cleanup_build);
        t_gpu_tree_build = (now_seconds() - t0) * 1e3;

        printf("GPU tree build: n_total=%u n_internal=%u  time=%.3f ms\n", gpu_builder.n_total, gpu_builder.n_internal,
               t_gpu_tree_build);

    cleanup_build:
        free(gpu_work);
        cvl_cl_staging_buffer_destroy(&buf_build_pos);
        cvl_cl_gpu_tree_build_destroy(&gpu_builder);
        cvl_cl_compute_destroy(&comp_build);
        if (status != CVL_CL_SUCCESS)
            goto cleanup;
    }

    /* ----------------------------------------------------------------- */
    /* 9. Per-precision GPU runs                                         */
    /*    precision_index: 0 = FP64, 1 = FP32                            */
    /* ----------------------------------------------------------------- */
    for (int pidx = 0; pidx < 2; ++pidx)
    {
        const cvl_cl_precision_t precision = (pidx == 0) ? CVL_CL_PRECISION_FP64 : CVL_CL_PRECISION_FP32;
        const char *prec_name = (pidx == 0) ? "FP64" : "FP32";

        /* Re-create compute backends for this precision. */
        cvl_cl_compute_destroy(&comp_direct);
        cvl_cl_compute_destroy(&comp_bh);
        cvl_cl_compute_destroy(&comp_fmm);
        cvl_cl_fmm_eval_destroy(&fmm_eval);

        {
            const char *direct_kernels[] = {"direct_sum"};
            CVL_CL_CHECK(cvl_cl_compute_init(&comp_direct, &ctx, &queue, &device, precision, DIRECT_SUM_SOURCE,
                                             direct_kernels, 1),
                         cleanup);
        }
        {
            const char *bh_kernels[] = {"bh_flat_eval"};
            CVL_CL_CHECK(
                cvl_cl_compute_init(&comp_bh, &ctx, &queue, &device, precision, BH_FLAT_EVAL_SOURCE, bh_kernels, 1),
                cleanup);
        }
        {
            const char *fmm_kernels[] = {"fmm_l2p_eval"};
            cvl_cl_status_t st =
                cvl_cl_compute_init(&comp_fmm, &ctx, &queue, &device, precision, fmm_combined, fmm_kernels, 1);
            if (st != CVL_CL_SUCCESS)
            {
                const char *log = cvl_cl_program_build_log(&comp_fmm.program);
                if (log)
                    fprintf(stderr, "FMM build log (%s):\n%s\n", prec_name, log);
                status = st;
                goto cleanup;
            }
        }

        /* Re-init staging buffers for this precision. */
        cvl_cl_staging_buffer_destroy(&buf_targets);
        cvl_cl_staging_buffer_destroy(&buf_src_pos);
        cvl_cl_staging_buffer_destroy(&buf_src_val);
        cvl_cl_staging_buffer_destroy(&buf_results);
        {
            const bool unified = cvl_cl_compute_unified_memory(&comp_direct);
            CVL_CL_CHECK(cvl_cl_staging_buffer_init(&buf_targets, precision, unified, N_TARGETS, NULL), cleanup);
            CVL_CL_CHECK(cvl_cl_staging_buffer_init(&buf_src_pos, precision, unified, N_SOURCES, NULL), cleanup);
            CVL_CL_CHECK(cvl_cl_staging_buffer_init(&buf_src_val, precision, unified, N_SOURCES, NULL), cleanup);
            CVL_CL_CHECK(cvl_cl_staging_buffer_init(&buf_results, precision, unified, N_TARGETS, NULL), cleanup);

            CVL_CL_CHECK(cvl_cl_staging_buffer_reserve(&buf_targets, &ctx, &queue, N_TARGETS), cleanup);
            CVL_CL_CHECK(cvl_cl_staging_buffer_reserve(&buf_src_pos, &ctx, &queue, N_SOURCES), cleanup);
            CVL_CL_CHECK(cvl_cl_staging_buffer_reserve(&buf_src_val, &ctx, &queue, N_SOURCES), cleanup);
            CVL_CL_CHECK(cvl_cl_staging_buffer_reserve(&buf_results, &ctx, &queue, N_TARGETS), cleanup);
        }

        /* Upload sources + targets (async, chained futures). */
        {
            cvl_cl_future_t f[3];
            CVL_CL_CHECK(cvl_cl_staging_buffer_write_async(&buf_targets, &queue, targets, N_TARGETS, 0, &f[0]),
                         cleanup);
            CVL_CL_CHECK(cvl_cl_staging_buffer_write_async(&buf_src_pos, &queue, sources_pos, N_SOURCES, 0, &f[1]),
                         cleanup);
            CVL_CL_CHECK(cvl_cl_staging_buffer_write_async(&buf_src_val, &queue, sources_val, N_SOURCES, 0, &f[2]),
                         cleanup);
            for (int i = 0; i < 3; ++i)
                CVL_CL_CHECK(cvl_cl_future_wait(&f[i]), cleanup);
        }

        /* ---- 9a. GPU direct sum ---- */
        {
            const double t0 = now_seconds();
            cvl_cl_kernel_t *k = cvl_cl_compute_kernel(&comp_direct, "direct_sum");
            TEST_ASSERT(k != NULL, "kernel 'direct_sum' not found");

            CVL_CL_CHECK(
                cvl_cl_kernel_set_args(
                    k,
                    (cvl_cl_karg_t[]){
                        {.type = CVL_CL_KARG_BUFFER, .index = 0, .mem = cvl_cl_buffer_mem(&buf_targets.device)},
                        {.type = CVL_CL_KARG_BUFFER, .index = 1, .mem = cvl_cl_buffer_mem(&buf_src_pos.device)},
                        {.type = CVL_CL_KARG_BUFFER, .index = 2, .mem = cvl_cl_buffer_mem(&buf_src_val.device)},
                        {.type = CVL_CL_KARG_SCALAR_UINT, .index = 3, .scalar_uint = N_SOURCES},
                        {.type = CVL_CL_KARG_SCALAR_UINT, .index = 4, .scalar_uint = N_TARGETS},
                        {.type = CVL_CL_KARG_BUFFER, .index = 5, .mem = cvl_cl_buffer_mem(&buf_results.device)},
                        {},
                    }),
                cleanup);

            const size_t global = N_TARGETS;
            CVL_CL_CHECK(cvl_cl_ndrange(&queue, k, 1, &global, NULL, NULL, 0, NULL, NULL), cleanup);
            CVL_CL_CHECK(cvl_cl_finish(&queue), cleanup);
            CVL_CL_CHECK(cvl_cl_staging_buffer_read_and_wait(&buf_results, &queue, gpu_results, N_TARGETS, 0), cleanup);
            t_gpu_direct[pidx] = (now_seconds() - t0) * 1e3;

            double abs_err;
            err_direct[pidx] = max_rel_error(gpu_results, cpu_direct, &abs_err);
            printf("[%s] GPU direct sum:   %.3f ms  max_abs=%.2e max_rel=%.2e\n", prec_name, t_gpu_direct[pidx],
                   abs_err, err_direct[pidx]);
        }

        /* ---- 9b. GPU BH flat eval ---- */
        {
            /* The bh_flat_node_t struct shrinks to 48 bytes in FP32 mode
             * (real_t = float), but buf_bh_nodes holds the 64-byte double
             * layout from section 8.  Convert + re-upload for FP32. */
            if (precision == CVL_CL_PRECISION_FP32)
            {
                bh_flat_node_f32_t *nodes_f32 =
                    (bh_flat_node_f32_t *)malloc(flat_tree.n_nodes * sizeof(bh_flat_node_f32_t));
                TEST_ASSERT(nodes_f32 != NULL, "malloc failed for FP32 flat nodes");
                convert_flat_nodes_f32(flat_tree.nodes, flat_tree.n_nodes, nodes_f32);
                const size_t f32_bytes = (size_t)flat_tree.n_nodes * sizeof(bh_flat_node_f32_t);
                CVL_CL_CHECK(cvl_cl_write_buffer(&queue, &buf_bh_nodes, 0, f32_bytes, nodes_f32, 0, NULL, NULL),
                             cleanup);
                /* Non-blocking write — keep nodes_f32 alive until the copy
                 * completes, then free (use-after-free crashes NVIDIA's
                 * async copy engine). */
                CVL_CL_CHECK(cvl_cl_finish(&queue), cleanup);
                free(nodes_f32);
            }

            const double t0 = now_seconds();
            cvl_cl_kernel_t *k = cvl_cl_compute_kernel(&comp_bh, "bh_flat_eval");
            TEST_ASSERT(k != NULL, "kernel 'bh_flat_eval' not found");

            /* theta is a real_t kernel argument — 4 bytes in FP32, 8 in FP64.
             * Passing an 8-byte double to an FP32 kernel makes clSetKernelArg
             * fail with CL_INVALID_ARG_SIZE. */
            const cvl_cl_karg_t theta_arg =
                (precision == CVL_CL_PRECISION_FP32)
                    ? (cvl_cl_karg_t){.type = CVL_CL_KARG_SCALAR_FLOAT, .index = 7, .scalar_float = (float)BH_THETA}
                    : (cvl_cl_karg_t){.type = CVL_CL_KARG_SCALAR_DOUBLE, .index = 7, .scalar_double = BH_THETA};

            CVL_CL_CHECK(
                cvl_cl_kernel_set_args(
                    k,
                    (cvl_cl_karg_t[]){
                        {.type = CVL_CL_KARG_BUFFER, .index = 0, .mem = cvl_cl_buffer_mem(&buf_bh_nodes)},
                        {.type = CVL_CL_KARG_BUFFER, .index = 1, .mem = cvl_cl_buffer_mem(&buf_bh_order)},
                        {.type = CVL_CL_KARG_BUFFER, .index = 2, .mem = cvl_cl_buffer_mem(&buf_bh_depth)},
                        {.type = CVL_CL_KARG_BUFFER, .index = 3, .mem = cvl_cl_buffer_mem(&buf_src_pos.device)},
                        {.type = CVL_CL_KARG_BUFFER, .index = 4, .mem = cvl_cl_buffer_mem(&buf_src_val.device)},
                        {.type = CVL_CL_KARG_SCALAR_UINT, .index = 5, .scalar_uint = N_TARGETS},
                        {.type = CVL_CL_KARG_SCALAR_UINT, .index = 6, .scalar_uint = BH_ORDER},
                        theta_arg,
                        {.type = CVL_CL_KARG_BUFFER, .index = 8, .mem = cvl_cl_buffer_mem(&buf_bh_coeffs)},
                        {.type = CVL_CL_KARG_BUFFER, .index = 9, .mem = cvl_cl_buffer_mem(&buf_results.device)},
                        {},
                    }),
                cleanup);

            const size_t global = N_TARGETS;
            CVL_CL_CHECK(cvl_cl_ndrange(&queue, k, 1, &global, NULL, NULL, 0, NULL, NULL), cleanup);
            CVL_CL_CHECK(cvl_cl_finish(&queue), cleanup);
            CVL_CL_CHECK(cvl_cl_staging_buffer_read_and_wait(&buf_results, &queue, gpu_results, N_TARGETS, 0), cleanup);
            t_gpu_bh[pidx] = (now_seconds() - t0) * 1e3;

            double abs_err;
            /* GPU BH evaluates at the first N_TARGETS source positions —
             * compare against the at-source references. */
            err_bh[pidx] = max_rel_error(gpu_results, cpu_direct_src, &abs_err);
            printf("[%s] GPU BH (theta=%.2f): %.3f ms  max_abs=%.2e max_rel=%.2e\n", prec_name, BH_THETA,
                   t_gpu_bh[pidx], abs_err, err_bh[pidx]);
        }

        /* ---- 9c. GPU FMM L2P eval ---- */
        {
            CVL_CL_CHECK(cvl_cl_fmm_eval_init(&fmm_eval, &comp_fmm, precision, NULL), cleanup);

            size_t fmm_work_sz = cvl_cl_fmm_eval_work_size(&fmm_eval, &fmm_tree);
            void *fmm_work = malloc(fmm_work_sz);
            TEST_ASSERT(fmm_work != NULL, "malloc failed for FMM work buffer");

            const double t0 = now_seconds();
            memset(gpu_results, 0, sizeof(gpu_results));
            CVL_CL_CHECK(cvl_cl_fmm_eval_run(&fmm_eval, &queue, &ctx, &fmm_tree, sources_pos, sources_val, N_TARGETS,
                                             targets, gpu_results, fmm_work, fmm_work_sz),
                         cleanup_fmm_work);
        cleanup_fmm_work:
            free(fmm_work);
            if (status != CVL_CL_SUCCESS)
                goto cleanup;
            CVL_CL_CHECK(cvl_cl_finish(&queue), cleanup);
            t_gpu_fmm[pidx] = (now_seconds() - t0) * 1e3;

            double abs_err;
            err_fmm[pidx] = max_rel_error(gpu_results, cpu_direct, &abs_err);
            printf("[%s] GPU FMM (order=%u):  %.3f ms  max_abs=%.2e max_rel=%.2e\n", prec_name, FMM_ORDER,
                   t_gpu_fmm[pidx], abs_err, err_fmm[pidx]);
        }
    }

    /* ----------------------------------------------------------------- */
    /* 10. Summary table                                                 */
    /* ----------------------------------------------------------------- */
    printf("\n=== OpenCL Benchmark: N=%d sources, M=%d targets ===\n\n", N_SOURCES, N_TARGETS);
    printf("%-14s %-10s %14s %12s %14s\n", "Method", "Precision", "GPU time (ms)", "Max rel err", "vs CPU direct");
    printf("%-14s %-10s %14s %12s %14s\n", "---------------", "----------", "--------------", "------------",
           "--------------");
    printf("%-14s %-10s %14.3f %12.1e %14s\n", "Direct sum", "FP64", t_gpu_direct[0], err_direct[0], "reference");
    printf("%-14s %-10s %14.3f %12.1e %14s\n", "Direct sum", "FP32", t_gpu_direct[1], err_direct[1], "ok");
    printf("%-14s %-10s %14.3f %12.1e %14s\n", "BH (theta=0.3)", "FP64", t_gpu_bh[0], err_bh[0], "ok");
    printf("%-14s %-10s %14.3f %12.1e %14s\n", "BH (theta=0.3)", "FP32", t_gpu_bh[1], err_bh[1], "ok (same tree)");
    printf("%-14s %-10s %14.3f %12.1e %14s\n", "FMM (order=4)", "FP64", t_gpu_fmm[0], err_fmm[0], "ok");
    printf("%-14s %-10s %14.3f %12.1e %14s\n", "FMM (order=4)", "FP32", t_gpu_fmm[1], err_fmm[1], "ok");
    printf("%-14s %-10s %14.3f %12.1e %14s\n", "CPU direct", "FP64", t_cpu_direct, 0.0, "reference");
    printf("%-14s %-10s %14.3f %12.1e %14s\n", "CPU BH", "FP64", t_cpu_bh, 0.0, "ok");
    printf("%-14s %-10s %14.3f %12.1e %14s\n", "CPU FMM", "FP64", t_cpu_fmm, 0.0, "ok");
    printf("\n");
    printf("%-20s %14s\n", "Tree build", "time (ms)");
    printf("%-20s %14s\n", "--------------------", "--------------");
    printf("%-20s %14.3f\n", "CPU BH build", t_cpu_bh_build);
    printf("%-20s %14.3f\n", "CPU FMM build", t_cpu_fmm_build);
    printf("%-20s %14.3f\n", "GPU uniform build", t_gpu_tree_build);

    /* ----------------------------------------------------------------- */
    /* 11. Accuracy assertions                                           */
    /*     - Direct sum must match CPU to ~1e-12 (FP64) / ~1e-5 (FP32).  */
    /*     - BH / FMM are approximate methods: their error vs the exact  */
    /*       direct sum depends on theta / order and is huge at field-   */
    /*       cancellation points (relative error there is meaningless).  */
    /*       So instead of an absolute bound, assert the GPU kernels are */
    /*       no worse than the CPU implementations on the same points.   */
    /*       The BH comparison uses the at-source references (the GPU    */
    /*       kernel evaluates at source positions).                      */
    /* ----------------------------------------------------------------- */
    double cpu_bh_abs = 0.0, cpu_fmm_abs = 0.0;
    const double cpu_bh_err = max_rel_error(cpu_bh_src, cpu_direct_src, &cpu_bh_abs);
    const double cpu_fmm_err = max_rel_error(cpu_fmm, cpu_direct, &cpu_fmm_abs);

    TEST_ASSERT(err_direct[0] < 1e-12, "FP64 direct sum mismatch: max_rel=%.2e", err_direct[0]);
    TEST_ASSERT(err_direct[1] < 1e-4, "FP32 direct sum mismatch: max_rel=%.2e", err_direct[1]);
    TEST_ASSERT(err_bh[0] < 10.0 * cpu_bh_err + 1e-12,
                "FP64 GPU BH error vs direct too large relative to CPU BH: gpu=%.2e cpu=%.2e", err_bh[0], cpu_bh_err);
    TEST_ASSERT(err_fmm[0] < 10.0 * cpu_fmm_err + 1e-12,
                "FP64 GPU FMM error vs direct too large relative to CPU FMM: gpu=%.2e cpu=%.2e", err_fmm[0],
                cpu_fmm_err);

    printf("\nAll benchmark tests passed.\n");
    ret = 0;

cleanup:
    if (status != CVL_CL_SUCCESS)
        fprintf(stderr, "benchmark failed with status: %s\n", cvl_cl_status_str(status));
    cvl_cl_fmm_eval_destroy(&fmm_eval);
    cvl_cl_staging_buffer_destroy(&buf_results);
    cvl_cl_staging_buffer_destroy(&buf_src_val);
    cvl_cl_staging_buffer_destroy(&buf_src_pos);
    cvl_cl_staging_buffer_destroy(&buf_targets);
    cvl_cl_buffer_destroy(&buf_bh_coeffs);
    cvl_cl_buffer_destroy(&buf_bh_depth);
    cvl_cl_buffer_destroy(&buf_bh_order);
    cvl_cl_buffer_destroy(&buf_bh_nodes);
    cvl_cl_compute_destroy(&comp_fmm);
    cvl_cl_compute_destroy(&comp_bh);
    cvl_cl_compute_destroy(&comp_direct);
    cvl_cl_compute_destroy(&comp_build);
    cvl_cl_gpu_tree_build_destroy(&gpu_builder);
    cvl_cl_staging_buffer_destroy(&buf_build_pos);
    cvl_cl_flat_tree_destroy(&flat_tree);
    free(flat_work);

    if (fmm_tree.buffer)
    {
        octree_free(&CVL_DEFAULT_ALLOCATOR, fmm_tree.buffer);
        fmm_tree.buffer = NULL;
    }
    if (bh_tree.buffer)
    {
        octree_free(&CVL_DEFAULT_ALLOCATOR, bh_tree.buffer);
        bh_tree.buffer = NULL;
    }

    free(fmm_combined);
    free(src_l2p);
    free(src_fmm_ops);
    free(src_mp_ops);
    free(src_multipole);
    free(src_math);
    free(src_types);

    cvl_cl_queue_destroy(&queue);
    cvl_cl_ctx_destroy(&ctx);
    return ret;
}

#else

#include <stdio.h>
int main(void)
{
    printf("OpenCL not available -- skipping benchmark.\n");
    return 0;
}

#endif /* CVL_OPENCL */
