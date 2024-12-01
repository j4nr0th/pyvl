//
// Created by jan on 15.11.2024.
//

#ifndef COMMON_H
#define COMMON_H

#include <math.h>
#include <stddef.h>
#include <stdint.h>

typedef double real_t;
// typedef uint32_t id_t;

typedef struct
{
    uint32_t orientation : 1;
    uint32_t value : 31;
} geo_id_t;

enum : uint32_t
{
    INVALID_ID = ((~(uint32_t)0) >> 1),                     //  ID that should not correspond to any entry
    REVERSED = ((uint32_t)1) << (8 * sizeof(uint32_t) - 1), // OR-ed with ID to indicate reverse direction
};

static inline bool id_valid(const geo_id_t *id)
{
    return id->value != INVALID_ID;
}

typedef union {
    real_t data[3];
    struct
    {
        real_t x;
        real_t y;
        real_t z;
    };
    struct
    {
        real_t v0;
        real_t v1;
        real_t v2;
    };
} real3_t;

typedef union {
    struct
    {
        real3_t row0;
        real3_t row1;
        real3_t row2;
    };
    struct
    {
        real_t data[9];
    };
    struct
    {
        real_t m00, m01, m02;
        real_t m10, m11, m12;
        real_t m20, m21, m22;
    };
} real3x3_t;

static inline real3_t real3_add(const real3_t a, const real3_t b)
{
    return (real3_t){{a.v0 + b.v0, a.v1 + b.v1, a.v2 + b.v2}};
}
static inline real3_t real3_sub(const real3_t a, const real3_t b)
{
    return (real3_t){{a.v0 - b.v0, a.v1 - b.v1, a.v2 - b.v2}};
}
static inline real_t real3_dot(const real3_t a, const real3_t b)
{
    return a.v0 * b.v0 + a.v1 * b.v1 + a.v2 * b.v2;
}
static inline real3_t real3_cross(const real3_t a, const real3_t b)
{
    return (real3_t){{
        a.v1 * b.v2 - a.v2 * b.v1,
        a.v2 * b.v0 - a.v0 * b.v2,
        a.v0 * b.v1 - a.v1 * b.v0,
    }};
}
static inline real_t real3_mag(const real3_t a)
{
    return hypot(hypot(a.v0, a.v1), a.v2);
}
static inline real3_t real3_unit(const real3_t a)
{
    const real_t mag = 1.0 / real3_mag(a);
    return (real3_t){{a.v0 * mag, a.v1 * mag, a.v2 * mag}};
}
static inline real3_t real3_mul1(const real3_t a, real_t k)
{
    return (real3_t){{a.v0 * k, a.v1 * k, a.v2 * k}};
}
static inline real3_t real3_neg(const real3_t a)
{
    return (real3_t){{-a.v0, -a.v1, -a.v2}};
}
static inline real_t real3_max(const real3_t a)
{
    return a.v0 > a.v1 ? (a.v0 > a.v2 ? a.v0 : a.v2) : (a.v1 > a.v2 ? a.v1 : a.v2);
}

static inline real3_t real3x3_vecmul(const real3x3_t a, const real3_t b)
{
    return (real3_t){
        .v0 = real3_dot(a.row0, b),
        .v1 = real3_dot(a.row1, b),
        .v2 = real3_dot(a.row2, b),
    };
}
static inline real3x3_t real3x3_matmul(const real3x3_t a, const real3x3_t b)
{
    const real3_t col_b0 = {{b.m00, b.m10, b.m20}};
    const real3_t col_b1 = {{b.m01, b.m11, b.m21}};
    const real3_t col_b2 = {{b.m02, b.m12, b.m22}};
    return (real3x3_t){
        .m00 = real3_dot(a.row0, col_b0),
        .m01 = real3_dot(a.row0, col_b1),
        .m02 = real3_dot(a.row0, col_b2),
        .m10 = real3_dot(a.row1, col_b0),
        .m11 = real3_dot(a.row1, col_b1),
        .m12 = real3_dot(a.row1, col_b2),
        .m20 = real3_dot(a.row2, col_b0),
        .m21 = real3_dot(a.row2, col_b1),
        .m22 = real3_dot(a.row2, col_b2),
    };
}
static inline real3x3_t real3x3_from_angles(const real3_t angles)
{
    const real_t cx = cos(angles.x);
    const real_t sx = sin(angles.x);
    const real_t cy = cos(angles.y);
    const real_t sy = sin(angles.y);
    const real_t cz = cos(angles.z);
    const real_t sz = sin(angles.z);

    return (real3x3_t){
        .m00 = cy * cz,
        .m01 = sx * sy * cz - cx * sz,
        .m02 = cx * sy * cz + sx * sz,
        .m10 = cy * sz,
        .m11 = sx * sy * sz + cx * cz,
        .m12 = cx * sy * sz - sx * cz,
        .m20 = -sy,
        .m21 = sx * cy,
        .m22 = cx * cy,
    };
}
static inline real3x3_t real3x3_inverse_from_angles(const real3_t angles)
{
    const real_t cx = cos(angles.x);
    const real_t sx = sin(angles.x);
    const real_t cy = cos(angles.y);
    const real_t sy = sin(angles.y);
    const real_t cz = cos(angles.z);
    const real_t sz = sin(angles.z);

    return (real3x3_t){
        .m00 = cy * cz,
        .m10 = sx * sy * cz - cx * sz,
        .m20 = cx * sy * cz + sx * sz,
        .m01 = cy * sz,
        .m11 = sx * sy * sz + cx * cz,
        .m21 = cx * sy * sz - sx * cz,
        .m02 = -sy,
        .m12 = sx * cy,
        .m22 = cx * cy,
    };
}
static inline real3_t angles_from_real3x3(const real3x3_t a)
{
    return (real3_t){.v0 = atan2(a.m21, a.m22), .v1 = atan2(-a.m20, hypot(a.m00, a.m01)), .v2 = atan2(a.m10, a.m00)};
}

static inline real_t clamp_angle_to_range(const real_t a)
{
    const double rem = remainder((double)a, 2 * M_PI);
    if (rem < 0)
        return 2 * M_PI + rem;
    return rem;
}

static inline int geo_id_compare(const geo_id_t id1, const geo_id_t id2)
{
    if (id1.value != id2.value)
        return 0;
    if (id1.orientation == id2.orientation)
        return 1;
    return -1;
}

typedef struct
{
    void *(*allocate)(void *state, size_t size);
    void (*deallocate)(void *state, void *ptr);
    void *(*reallocate)(void *state, void *ptr, size_t new_size);
    void *state;
} allocator_t;

#endif // COMMON_H
