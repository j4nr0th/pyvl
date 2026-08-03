/*
 * test_cvl_cl_device_kernel_compile.c — Verify that all .cl.h kernel
 * headers compile as OpenCL C and produce numerically correct results.
 *
 * Strategy:
 *   1. Read the five .cl.h files at runtime, strip #include lines
 *      (OpenCL C at runtime has no filesystem), and concatenate
 *      in dependency order.
 *   2. Prepend a uint64_t typedef (required by OpenCL C but not
 *      provided by the .cl.h headers).
 *   3. Append a small test kernel that uses functions from every
 *      .cl.h to verify they all compile and work.
 *   4. Build the program, run the kernel, and check the output.
 *
 * The source directory path is injected via CMake -DCVL_CL_SOURCE_DIR.
 */

#ifdef CVL_OPENCL

#include "../test_common.h"
#include "cvl_cl.h"

#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/*  Helpers for building the concatenated kernel source                */
/* ------------------------------------------------------------------ */

/**
 * @brief Concatenate @p src into @p dst, skipping lines that start
 *        with `#include` (to avoid filesystem-dependent includes
 *        that OpenCL C cannot resolve at runtime).
 *
 * @param dst     Output buffer (NUL-terminated on return).
 * @param dst_cap Capacity of @p dst (including trailing NUL).
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

        /* Trim leading whitespace to detect `#include` */
        const char *trimmed = src;
        while (*trimmed == ' ' || *trimmed == '\t')
            ++trimmed;
        int is_include = (trimmed[0] == '#' && strncmp(trimmed + 1, "include", 7) == 0);

        if (!is_include)
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
    return read_file_to_string(path, 4096);
}

/* ------------------------------------------------------------------ */
/*  Test kernel (appended after concatenated .cl.h source)             */
/* ------------------------------------------------------------------ */

/*
 * The kernel wrapper exercises functions from every .cl.h file:
 *   cvl_cl_types.h.cl       — real3_add, real3_dot (used implicitly)
 *   cvl_cl_math.h.cl        — particle_kernel, morton_3d, multipole_num_coeffs
 *   cvl_cl_multipole.h.cl   — build_binomial_expansion, multipole_poly_mul_linear,
 *                             multipole_add_poly_to_order
 *   cvl_cl_multipole_ops.h.cl  — multipole_t, multipole_update
 *   cvl_cl_fmm_ops.h.cl     — local_expansion_t, local_expansion_eval,
 *                             multipole_to_local
 *
 * To avoid VLAs in OpenCL C, all local arrays use compile-time-constant
 * sizes derived from order = 2.
 */
/* ── Prepend uint64_t typedef (needed because #include lines are stripped) ── */
static const char *UINT64_T_PREAMBLE = "typedef unsigned long uint64_t;\n"
                                       "typedef unsigned long size_t;\n"
                                       "typedef unsigned uint;\n"
                                       "\n";

static const char *KERNEL_WRAPPER =
    "\n"
    "__kernel void test_kernel(__global double *out)\n"
    "{\n"
    "    const unsigned ord = 2;\n"
    "    const size_t   nc  = 15;  /* multipole_num_coeffs(2) */\n"
    "    const size_t   scr = 27;  /* multipole_scratch_size(2) */\n"
    "\n"
    "    /* ---- 1. particle_kernel (cvl_cl_math.h.cl) ---- */\n"
    "    {\n"
    "        real3_t gamma = {1.0, 2.0, 3.0};\n"
    "        real3_t r     = {4.0, 5.0, 6.0};\n"
    "        real3_t res   = particle_kernel(gamma, r);\n"
    "        out[0] = res.x;\n"
    "        out[1] = res.y;\n"
    "        out[2] = res.z;\n"
    "    }\n"
    "\n"
    "    /* ---- 2. morton_3d (cvl_cl_math.h.cl) ---- */\n"
    "    {\n"
    "        real3_t  p      = {0.1, 0.2, 0.3};\n"
    "        real3_t  center = {0.0, 0.0, 0.0};\n"
    "        uint64_t mc     = morton_3d(p, center, 1.0);\n"
    "        out[3] = (double)mc;\n"
    "    }\n"
    "\n"
    "    /* ---- 3. multipole_num_coeffs / multipole_coeff_index ---- */\n"
    "    out[4] = (double)multipole_num_coeffs(4);\n"
    "    out[5] = (double)multipole_coeff_index(1, 1, 0, 0);\n"
    "\n"
    "    /* ---- 4. build_binomial_expansion (cvl_cl_multipole.h.cl) ---- */\n"
    "    {\n"
    "        real_t shift_exp[3 * 9];  /* work_order=2 => dim=3, plane=9 */\n"
    "        build_binomial_expansion(shift_exp, 1.0, 2.0, 3.0, 2, 3, 9);\n"
    "        /* (x+1)^1 const term = 1.0 */\n"
    "        out[6] = shift_exp[0*9 + 1*3 + 0];\n"
    "    }\n"
    "\n"
    "    /* ---- 5. multipole_poly_mul_linear (cvl_cl_multipole.h.cl) ---- */\n"
    "    {\n"
    "        real_t a[15];\n"
    "        real_t b[15];\n"
    "        for (int i = 0; i < 15; ++i) a[i] = 0.0;\n"
    "        a[multipole_coeff_index(0, 0, 0, 0)] = 1.0;\n"
    "        multipole_poly_mul_linear(a, b, 2.0, 3.0, 4.0, 1.0, 2);\n"
    "        out[7]  = b[multipole_coeff_index(0, 0, 0, 0)];  /* const = 1 */\n"
    "        out[8]  = b[multipole_coeff_index(1, 1, 0, 0)];  /* x  = 2 */\n"
    "        out[9]  = b[multipole_coeff_index(1, 0, 1, 0)];  /* y  = 3 */\n"
    "        out[10] = b[multipole_coeff_index(1, 0, 0, 1)];  /* z  = 4 */\n"
    "    }\n"
    "\n"
    "    /* ---- 6. multipole_add_poly_to_order (cvl_cl_multipole.h.cl) ---- */\n"
    "    {\n"
    "        real_t coeffs_x[15];\n"
    "        real_t coeffs_y[15];\n"
    "        real_t coeffs_z[15];\n"
    "        real_t poly[15];\n"
    "        for (int i = 0; i < 15; ++i) coeffs_x[i] = coeffs_y[i] = coeffs_z[i] = poly[i] = 0.0;\n"
    "        poly[multipole_coeff_index(0, 0, 0, 0)] = 1.0;\n"
    "        multipole_add_poly_to_order(poly, 2, 2.0, 1.0, 2.0, 3.0,\n"
    "                                     coeffs_x, coeffs_y, coeffs_z);\n"
    "        size_t oidx = multipole_coeff_index(2, 0, 0, 0);\n"
    "        out[11] = coeffs_x[oidx];  /* 2.0 * 1.0 = 2.0 */\n"
    "        out[12] = coeffs_y[oidx];  /* 2.0 * 2.0 = 4.0 */\n"
    "        out[13] = coeffs_z[oidx];  /* 2.0 * 3.0 = 6.0 */\n"
    "    }\n"
    "\n"
    "    /* ---- 7. multipole_update (cvl_cl_multipole_ops.h.cl) ---- */\n"
    "    {\n"
    "        real_t mx[15], my[15], mz[15];\n"
    "        real_t cur[27], nxt[27];\n"
    "        for (int i = 0; i < 15; ++i) mx[i] = my[i] = mz[i] = 0.0;\n"
    "        multipole_t mp = {.order = 2, .center = {0,0,0},\n"
    "                          .coeffs_x = mx, .coeffs_y = my, .coeffs_z = mz};\n"
    "        real3_t src_pos   = {1.0, 0.0, 0.0};\n"
    "        real3_t src_value = {2.0, 0.0, 0.0};\n"
    "        multipole_update(&mp, src_pos, src_value, cur, nxt);\n"
    "        /* After P2M of (1,0,0) with value (2,0,0) at order 2:\n"
    "         * coeffs_x[0] should be 2.0 (source_value.x * 1.0 at order 0) */\n"
    "        out[14] = mx[multipole_coeff_index(0, 0, 0, 0)];  /* 2.0 */\n"
    "    }\n"
    "\n"
    "    /* ---- 8. local_expansion_eval (cvl_cl_fmm_ops.h.cl) ---- */\n"
    "    {\n"
    "        real_t lx[15], ly[15], lz[15];\n"
    "        for (int i = 0; i < 15; ++i) lx[i] = ly[i] = lz[i] = 0.0;\n"
    "        /* Set constant term of local expansion = (1,2,3) */\n"
    "        lx[multipole_coeff_index(0, 0, 0, 0)] = 1.0;\n"
    "        ly[multipole_coeff_index(0, 0, 0, 0)] = 2.0;\n"
    "        lz[multipole_coeff_index(0, 0, 0, 0)] = 3.0;\n"
    "        local_expansion_t loc = {.order = 2, .center = {0,0,0},\n"
    "                                 .coeffs_x = lx, .coeffs_y = ly, .coeffs_z = lz};\n"
    "        real3_t eval_pt = {1.0, 2.0, 3.0};\n"
    "        real3_t ev = local_expansion_eval(&loc, eval_pt);\n"
    "        out[15] = ev.x;  /* 1.0 */\n"
    "        out[16] = ev.y;  /* 2.0 */\n"
    "        out[17] = ev.z;  /* 3.0 */\n"
    "    }\n"
    "}\n";

/* ------------------------------------------------------------------ */
/*  Expected values (computed by the same functions on the host side)  */
/* ------------------------------------------------------------------ */

static const double EXPECTED_OUT[18] = {
    /* 0-2: particle_kernel({1,2,3}, {4,5,6}) */
    1.0 / 77.0, /* x = gamma.x / r2 */
    2.0 / 77.0, /* y = gamma.y / r2 */
    3.0 / 77.0, /* z = gamma.z / r2 */
    /* 3: morton_3d({0.1,0.2,0.3}, {0,0,0}, 1.0) — any positive value */
    0.0, /* verified dynamically */
    /* 4-5: coefficient helpers */
    70.0, /* multipole_num_coeffs(4) */
    4.0,  /* multipole_coeff_index(1,1,0,0) */
    /* 6: build_binomial_expansion — (x+1)^1 const term */
    1.0,
    /* 7-10: multipole_poly_mul_linear(a={1}, (2x+3y+4z+1)) */
    1.0, /* const */
    2.0, /* x */
    3.0, /* y */
    4.0, /* z */
    /* 11-13: multipole_add_poly_to_order(scale=2, cx=1, cy=2, cz=3) */
    2.0, /* coeffs_x at order 2 */
    4.0, /* coeffs_y at order 2 */
    6.0, /* coeffs_z at order 2 */
    /* 14: multipole_update coeffs_x[0] after P2M(1,0,0) with value 2 */
    2.0,
    /* 15-17: local_expansion_eval const=(1,2,3) at (1,2,3) */
    1.0, /* x */
    2.0, /* y */
    3.0, /* z */
};

enum
{
    NUM_OUTPUTS = 18
};

/* ------------------------------------------------------------------ */
/*  Test runner — builds, runs, and verifies the kernel                */
/*  for a given precision mode and tolerance.                          */
/* ------------------------------------------------------------------ */

static int run_precision_test(cvl_cl_ctx_t *ctx, cvl_cl_queue_t *queue, const char *full_source,
                              cvl_cl_precision_t precision, double tolerance, const char *label)
{
    cvl_cl_status_t status = CVL_CL_SUCCESS;
    cvl_cl_program_t program = {0};
    cvl_cl_kernel_t kernel = {0};
    cvl_cl_buffer_t buf_out = {0};
    int ret = 1;

    printf("  [%s] building program ...\n", label);

    const cvl_cl_device_t *dev = cvl_cl_ctx_device(ctx);

    cvl_cl_program_desc_t desc = {
        .source_type = CVL_CL_PROGRAM_SOURCE_STRING,
        .source_string = full_source,
        .precision = precision,
    };
    status = cvl_cl_program_create(ctx, &desc, dev->id, &program, NULL);
    if (status != CVL_CL_SUCCESS)
    {
        const char *log = cvl_cl_program_build_log(&program);
        fprintf(stderr, "  [%s] Program build FAILED with status %s.\n", label, cvl_cl_status_str(status));
        if (log)
            fprintf(stderr, "  Build log:\n%s\n", log);
        goto cleanup;
    }
    TEST_ASSERT(cvl_cl_program_program(&program) != NULL, "  [%s] Program handle NULL", label);
    TEST_ASSERT(cvl_cl_program_build_log(&program) == NULL, "  [%s] Build log not NULL", label);

    CVL_CL_CHECK(cvl_cl_kernel_create(&program, "test_kernel", &kernel), cleanup);

    const size_t out_bytes = NUM_OUTPUTS * sizeof(double);
    CVL_CL_CHECK(cvl_cl_buffer_create(
                     ctx, &(cvl_cl_buffer_desc_t){.access = CVL_CL_BUF_WRITE_ONLY, .size_bytes = out_bytes}, &buf_out),
                 cleanup);

    CVL_CL_CHECK(cvl_cl_kernel_set_args(&kernel,
                                        (cvl_cl_karg_t[]){
                                            {.type = CVL_CL_KARG_BUFFER, .index = 0, .mem = buf_out.mem},
                                            {},
                                        }),
                 cleanup);

    {
        const size_t global_work = 1;
        const size_t local_work = 1;
        CVL_CL_CHECK(cvl_cl_ndrange(queue, &kernel, 1, &global_work, &local_work, NULL, 0, NULL, NULL), cleanup);
    }

    CVL_CL_CHECK(cvl_cl_flush(queue), cleanup);
    CVL_CL_CHECK(cvl_cl_finish(queue), cleanup);

    double host_out[NUM_OUTPUTS];
    memset(host_out, 0, out_bytes);
    CVL_CL_CHECK(cvl_cl_read_buffer(queue, &buf_out, 0, out_bytes, host_out, 0, NULL, NULL), cleanup);
    CVL_CL_CHECK(cvl_cl_finish(queue), cleanup);

    TEST_ASSERT(host_out[3] > 0, "  [%s] morton_3d output positive, got %g", label, host_out[3]);

    for (int i = 0; i < NUM_OUTPUTS; ++i)
    {
        if (i == 3)
            continue;
        double abs_err = fabs(host_out[i] - EXPECTED_OUT[i]);
        double rel_err = abs_err / (fabs(EXPECTED_OUT[i]) + 1e-30);
        TEST_ASSERT(abs_err < tolerance || rel_err < tolerance, "  [%s] out[%d] = %.15g, expected %.15g", label, i,
                    host_out[i], EXPECTED_OUT[i]);
    }

    printf("  [%s] passed.\n", label);
    ret = 0;

cleanup:
    cvl_cl_buffer_destroy(&buf_out);
    cvl_cl_kernel_destroy(&kernel);
    cvl_cl_program_destroy(&program);
    return ret;
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
    unsigned count = 0;
    int ret = 1;

    char *types_src = NULL, *math_src = NULL, *multipole_src = NULL;
    char *mpo_src = NULL, *fmm_src = NULL, *full_source = NULL;

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

    CVL_CL_CHECK(cvl_cl_ctx_create(&device, &ctx), cleanup);
    CVL_CL_CHECK(cvl_cl_queue_create(&ctx, NULL, &queue), cleanup);

    types_src = read_cl_source("cvl_cl_types.h.cl");
    math_src = read_cl_source("cvl_cl_math.h.cl");
    multipole_src = read_cl_source("cvl_cl_multipole.h.cl");
    mpo_src = read_cl_source("cvl_cl_multipole_ops.h.cl");
    fmm_src = read_cl_source("cvl_cl_fmm_ops.h.cl");

    /* Concatenate in dependency order, stripping #include */
    {
        size_t len_p = strlen(UINT64_T_PREAMBLE);
        size_t len_t = strlen(types_src);
        size_t len_m = strlen(math_src);
        size_t len_mp = strlen(multipole_src);
        size_t len_o = strlen(mpo_src);
        size_t len_f = strlen(fmm_src);
        size_t len_w = strlen(KERNEL_WRAPPER);
        size_t total = len_p + len_t + len_m + len_mp + len_o + len_f + len_w + 1024;
        full_source = (char *)calloc(total, 1);
        TEST_ASSERT(full_source != NULL, "calloc for full_source failed");

        size_t pos = 0;
        memcpy(full_source + pos, UINT64_T_PREAMBLE, len_p);
        pos += len_p;
        pos += skip_include_concat(full_source + pos, total - pos, types_src);
        pos += skip_include_concat(full_source + pos, total - pos, math_src);
        pos += skip_include_concat(full_source + pos, total - pos, multipole_src);
        pos += skip_include_concat(full_source + pos, total - pos, mpo_src);
        pos += skip_include_concat(full_source + pos, total - pos, fmm_src);
        TEST_ASSERT(pos + len_w < total, "Concatenated source overflow");
        memcpy(full_source + pos, KERNEL_WRAPPER, len_w + 1);
    }

    printf("Precision test suite\n");
    ret = run_precision_test(&ctx, &queue, full_source, CVL_CL_PRECISION_FP64, 1e-12, "FP64");
    if (ret == 0)
        ret = run_precision_test(&ctx, &queue, full_source, CVL_CL_PRECISION_FP32, 1e-5f, "FP32");

    if (ret == 0)
        printf("All device kernel compile tests passed.\n");

cleanup:
    free(full_source);
    free(types_src);
    free(math_src);
    free(multipole_src);
    free(mpo_src);
    free(fmm_src);
    cvl_cl_queue_destroy(&queue);
    cvl_cl_ctx_destroy(&ctx);
    return ret;
}

#else /* !CVL_OPENCL */

#include <stdio.h>
int main(void)
{
    printf("OpenCL not available -- skipping test.\n");
    return 0;
}

#endif /* CVL_OPENCL */
