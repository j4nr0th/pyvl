#pragma once
/*
 * Shared type definitions for C17 / OpenCL C dual-compilation headers.
 *
 * This file is designed to be `#include`d by `.cl.h` kernel header files.
 * It provides the common types (real_t, real3_t, real3x3_t) and macros
 * needed to write kernel function bodies that compile identically in
 * C17 (host side) and OpenCL C (device side).
 *
 * OpenCL C path: defines real_t controlled by the compile-time define
 *   CVL_CL_REAL_FP32  →  float (32-bit)
 *   otherwise          →  double (64-bit, default)
 * The cvl_cl_program module selects precision via a prepended header
 * block.  See @ref cvl_cl_precision_t.
 *
 * C17 path: includes the project's common.h for the canonical types.
 */

#ifdef __OPENCL_C_VERSION__

/* ---- OpenCL C side ---- */

#ifdef CVL_CL_REAL_FP32
    typedef float real_t;
#else
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
    typedef double real_t;
#endif

typedef struct
{
    real_t x, y, z;
} real3_t;

typedef struct
{
    real3_t row0, row1, row2;
} real3x3_t;

/* OpenCL C has its own math builtins — no includes needed. */

static inline real3_t real3_add(real3_t a, real3_t b)
{
    return (real3_t){a.x + b.x, a.y + b.y, a.z + b.z};
}
static inline real3_t real3_sub(real3_t a, real3_t b)
{
    return (real3_t){a.x - b.x, a.y - b.y, a.z - b.z};
}
static inline real_t real3_dot(real3_t a, real3_t b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
static inline real3_t real3_cross(real3_t a, real3_t b)
{
    return (real3_t){a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
static inline real_t real3_mag(real3_t a)
{
    return sqrt(real3_dot(a, a));
}
static inline real3_t real3_unit(real3_t a)
{
    real_t m = 1.0 / real3_mag(a);
    return (real3_t){a.x * m, a.y * m, a.z * m};
}
static inline real3_t real3_mul1(real3_t a, real_t k)
{
    return (real3_t){a.x * k, a.y * k, a.z * k};
}
static inline real3_t real3_neg(real3_t a)
{
    return (real3_t){-a.x, -a.y, -a.z};
}
static inline real_t real3_max(real3_t a)
{
    return a.x > a.y ? (a.x > a.z ? a.x : a.z) : (a.y > a.z ? a.y : a.z);
}
static inline bool real3_all_zero(real3_t a)
{
    return (a.x == 0 && a.y == 0 && a.z == 0);
}

static inline real3_t real3x3_vecmul(real3x3_t a, real3_t b)
{
    return (real3_t){real3_dot(a.row0, b), real3_dot(a.row1, b), real3_dot(a.row2, b)};
}
static inline real3_t real3x3_vecmul_transpose(real3x3_t a, real3_t b)
{
    return real3_add(real3_mul1(a.row0, b.x), real3_add(real3_mul1(a.row1, b.y), real3_mul1(a.row2, b.z)));
}

#else

/* ---- Host C17 side ---- */

#include "../common.h"

#endif /* __OPENCL_C_VERSION__ */

/* ------------------------------------------------------------------ */
/* Cross-platform macros (common to both sides)                        */
/* ------------------------------------------------------------------ */

#ifndef CVL_CL_RESTRICT
#ifdef __GNUC__
#define CVL_CL_RESTRICT __restrict__
#else
#define CVL_CL_RESTRICT restrict
#endif
#endif
