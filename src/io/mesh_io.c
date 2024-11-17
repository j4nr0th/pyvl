//
// Created by jan on 16.11.2024.
//

#include "mesh_io.h"

#include <ctype.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct
{
    char *buffer;
    const allocator_t *allocator;
    unsigned used;
    unsigned capacity;
    unsigned increment;
} string_stream;

#ifdef __GNUC__
[[gnu::format(printf, 2, 3)]]
#endif
static int string_stream_write_fmt(string_stream *stream, const char *fmt, ...)
{
    va_list args, cpy;
    va_start(args, fmt);
    va_copy(cpy, args);
    int cnt = vsnprintf(nullptr, 0, fmt, cpy);
    va_end(cpy);
    if (cnt < 0)
    {
        va_end(args);
        return 0;
    }

    const unsigned new_usage = stream->used + (unsigned)cnt;
    if (new_usage >= stream->capacity)
    {
        const unsigned new_capacity = stream->capacity + stream->increment * (1 + (new_usage - stream->capacity) / stream->increment);
        char *const new_buffer = stream->allocator->reallocate(stream->allocator->state, stream->buffer, sizeof(*new_buffer) * new_capacity);
        if (!new_buffer)
        {
            return -1;
        }
        stream->buffer = new_buffer;
        stream->capacity = new_capacity;
    }
    cnt = vsnprintf(stream->buffer + stream->used, stream->capacity - stream->used - 1, fmt, args);
    va_end(args);
    if (cnt < 0)
    {
        return cnt;
    }
    stream->used += cnt;
    return 0;
}


static int unpack_id(const geo_id_t id)
{
    const int v = id.value == INVALID_ID ? 0 : (int)id.value + 1;

    if (id.orientation)
    {
        return -v;
    }
    return v;
}

char* serialize_mesh(const mesh_t* this, const allocator_t* allocator)
{
    string_stream out_stream = {
        .buffer = nullptr, .allocator = allocator, .used = 0, .capacity = 0, .increment = 1 << 12,
    };

    if (string_stream_write_fmt(&out_stream, u8"0\n%6u %5u %8u\n", this->n_points, this->n_lines, this->n_surfaces) < 0) goto failed;
    for (unsigned ipt = 0; ipt < this->n_points; ++ipt)
    {
        const real3_t *pos = this->positions + ipt;
        if (string_stream_write_fmt(&out_stream, u8"%.15g %.15g %.15g\n", pos->v0, pos->v1, pos->v2) < 0) goto failed;
    }
    for (unsigned iln = 0; iln < this->n_lines; ++iln)
    {
        const line_t *ln = this->lines + iln;
        const int ids[2] = {unpack_id(ln->p1), unpack_id(ln->p2)};
        if (string_stream_write_fmt(&out_stream, u8"%d %d\n", abs(ids[0]), abs(ids[1])) < 0) goto failed;
    }
    for (unsigned is = 0; is < this->n_surfaces; ++is)
    {
        const surface_t *s = this->surfaces[is];
        if (string_stream_write_fmt(&out_stream, u8"%u", s->n_lines) < 0) goto failed;
        for (unsigned iln = 0; iln < s->n_lines; ++iln)
        {
            if (string_stream_write_fmt(&out_stream, u8" %d", unpack_id(s->lines[iln])) < 0) goto failed;
        }
        if (string_stream_write_fmt(&out_stream, u8"\n") < 0) goto failed;
    }

    char *out = allocator->reallocate(allocator->state, out_stream.buffer, (out_stream.used + 1) * sizeof(*out_stream.buffer));
    if (!out) goto failed;

    return out;

failed:
    allocator->deallocate(allocator->state, out_stream.buffer);
    return nullptr;
}

static geo_id_t pack_id(const int i)
{
    return (geo_id_t){.orientation = i < 0, .value = abs(i) - 1};
}


mesh_t* deserialize_mesh(const char* str, const allocator_t* allocator)
{
    mesh_t *this = allocator->allocate(allocator->state, sizeof(*this));
    if (!this) return nullptr;

    this->positions = nullptr;
    this->lines = nullptr;
    this->surfaces = nullptr;

    char* ptr;
    //  Parse version of file
    const unsigned version = strtoul(str, &ptr, 10);
    if (ptr == str || version > 0) goto failed;
    str = ptr;

    //  Parse point, line, and surface counts
    this->n_points = (unsigned)strtoul(str, &ptr, 10);
    if (ptr == str) goto failed;
    str = ptr;
    this->n_lines = (unsigned)strtoul(str, &ptr, 10);
    if (ptr == str) goto failed;
    str = ptr;
    this->n_surfaces = (unsigned)strtoul(str, &ptr, 10);
    if (ptr == str) goto failed;
    str = ptr;

    this->positions = allocator->allocate(allocator->state, sizeof(*this->positions) * this->n_points);
    if (!this->positions) goto failed;
    //  Parse positions
    for (unsigned ipos = 0; ipos < this->n_points; ++ipos)
    {
        real3_t* p = this->positions + ipos;
        for (unsigned k = 0; k < 3; ++k)
        {
            p->data[k] = strtod(str, &ptr);
            if (str == ptr) goto failed;
            str = ptr;
        }
    }

    this->lines = allocator->allocate(allocator->state, sizeof(*this->lines) * this->n_lines);
    if (!this->lines) goto failed;
    //  Parse lines
    for (unsigned iln = 0; iln < this->n_lines; ++iln)
    {
        line_t* ln = this->lines + iln;
        ln->p1 = pack_id((int)strtol(str, &ptr, 10));
        if (str == ptr) goto failed;
        str = ptr;
        ln->p2 = pack_id((int)strtol(str, &ptr, 10));
        if (str == ptr) goto failed;
        str = ptr;
    }
    // save place where surfaces begin
    const char* const p_begin = str;

    // read surface data, but only focus on getting number of surfaces themselves
    unsigned total_lines = 0;
    for (unsigned is = 0; is < this->n_surfaces; ++is)
    {
        const unsigned n = strtoul(str, &ptr, 10);
        if (str == ptr) goto failed;
        str = ptr;
        total_lines += n;
        for (unsigned i = 0; i < n; ++i)
        {
            (void)strtol(str, &ptr, 10);
            if (str == ptr) goto failed;
            str = ptr;
        }
    }
    //  restore parsing state
    str = p_begin;

    this->surfaces = allocator->allocate(allocator->state, sizeof(*this->surfaces) * this->n_surfaces + (total_lines * sizeof(geo_id_t)) + (this->n_surfaces * sizeof(surface_t)));
    if (!this->surfaces) goto failed;

    geo_id_t *id_ptr = (geo_id_t *)(this->surfaces + this->n_surfaces);

    for (unsigned is = 0; is < this->n_surfaces; ++is)
    {
        const unsigned n = strtoul(str, &ptr, 10);
        if (str == ptr) goto failed;
        str = ptr;
        *(uint32_t *)id_ptr = (uint32_t)n;
        for (unsigned i = 0; i < n; ++i)
        {
            id_ptr[i + 1] = pack_id((int)strtol(str, &ptr, 10));
            if (str == ptr) goto failed;
            str = ptr;
        }
        this->surfaces[is] = (surface_t *)id_ptr;
        id_ptr += n + 1;
    }

    return this;

failed:
    allocator->deallocate(allocator->state, this->surfaces);
    allocator->deallocate(allocator->state, this->lines);
    allocator->deallocate(allocator->state, this->positions);
    allocator->deallocate(allocator->state, this);
    return nullptr;
}
