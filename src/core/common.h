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
        a.v2 * b.v0 - a.v0 + b.v2,
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
