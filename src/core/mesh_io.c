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
    int cnt = vsnprintf(NULL, 0, fmt, cpy);
    va_end(cpy);
    if (cnt < 0)
    {
        va_end(args);
        return 0;
    }

    const unsigned new_usage = stream->used + (unsigned)cnt;
    if (new_usage >= stream->capacity)
    {
        const unsigned new_capacity =
            stream->capacity + stream->increment * (1 + (new_usage - stream->capacity) / stream->increment);
        char *const new_buffer =
            stream->allocator->reallocate(stream->allocator->state, stream->buffer, sizeof(*new_buffer) * new_capacity);
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

char *serialize_mesh(const mesh_t *this, const real3_t *positions, const allocator_t *allocator)
{
    string_stream out_stream = {
        .buffer = NULL,
        .allocator = allocator,
        .used = 0,
        .capacity = 0,
        .increment = 1 << 12,
    };

    if (string_stream_write_fmt(&out_stream, "0\n/* Points Lines Surfaces */\n   %6u %5u %8u\n/* Positions */\n",
                                this->n_points, this->n_lines, this->n_surfaces) < 0)
        goto failed;
    for (unsigned ipt = 0; ipt < this->n_points; ++ipt)
    {
        const real3_t *pos = positions + ipt;
        if (string_stream_write_fmt(&out_stream, "%.15g %.15g %.15g\n", pos->v0, pos->v1, pos->v2) < 0)
            goto failed;
    }

    if (string_stream_write_fmt(&out_stream, "/* Line connectivity */\n") < 0)
        goto failed;
    for (unsigned iln = 0; iln < this->n_lines; ++iln)
    {
        const line_t *ln = this->lines + iln;
        const int ids[2] = {unpack_id(ln->p1), unpack_id(ln->p2)};
        if (string_stream_write_fmt(&out_stream, "%d %d\n", abs(ids[0]), abs(ids[1])) < 0)
            goto failed;
    }

    if (string_stream_write_fmt(&out_stream, "/* Surface connectivity */\n") < 0)
        goto failed;

    for (unsigned is = 0; is < this->n_surfaces; ++is)
    {
        if (string_stream_write_fmt(&out_stream, "%u", this->surface_offsets[is + 1] - this->surface_offsets[is]) < 0)
            goto failed;
        for (unsigned iln = this->surface_offsets[is]; iln < this->surface_offsets[is + 1]; ++iln)
        {
            if (string_stream_write_fmt(&out_stream, " %d", unpack_id(this->surface_lines[iln])) < 0)
                goto failed;
        }
        if (string_stream_write_fmt(&out_stream, "\n") < 0)
            goto failed;
    }

    char *out =
        allocator->reallocate(allocator->state, out_stream.buffer, (out_stream.used + 1) * sizeof(*out_stream.buffer));
    if (!out)
        goto failed;

    return out;

failed:
    allocator->deallocate(allocator->state, out_stream.buffer);
    return NULL;
}

static geo_id_t pack_id(const int i)
{
    return (geo_id_t){.orientation = i < 0, .value = abs(i) - 1};
}

static const char *skip_forward(const char *str)
{
    for (;;)
    {
        if (*str == '/')
        {
            if (*(str + 1) == '/')
                while (*str && *str != '\n')
                    ++str;
            else if (*(str + 1) == '*')
            {
                str += 2;
                while (*str && *str != '*' && *(str + 1) != '/')
                    ++str;
                str += 2;
            }
            continue;
        }
        if (!isspace(*str))
            break;
        ++str;
    }
    return str;
}

int deserialize_mesh(mesh_t *p_out, real3_t **p_positions, const char *str, const allocator_t *allocator)
{
    mesh_t this;
    real3_t *positions = NULL;
    this.lines = NULL;
    this.surface_offsets = NULL;
    this.surface_lines = NULL;

    char *ptr = (char *)str;
    //  Parse version of file
    str = skip_forward(ptr);
    const unsigned version = strtoul(str, &ptr, 10);
    if (ptr == str || version > 0)
        goto failed;
    str = skip_forward(ptr);

    //  Parse point, line, and surface counts
    this.n_points = (unsigned)strtoul(str, &ptr, 10);
    if (ptr == str)
        goto failed;
    str = skip_forward(ptr);
    this.n_lines = (unsigned)strtoul(str, &ptr, 10);
    if (ptr == str)
        goto failed;
    str = skip_forward(ptr);
    this.n_surfaces = (unsigned)strtoul(str, &ptr, 10);
    if (ptr == str)
        goto failed;
    str = skip_forward(ptr);

    positions = allocator->allocate(allocator->state, sizeof(*positions) * this.n_points);
    if (!positions)
        goto failed;
    //  Parse positions
    for (unsigned ipos = 0; ipos < this.n_points; ++ipos)
    {
        real3_t *p = positions + ipos;
        for (unsigned k = 0; k < 3; ++k)
        {
            p->data[k] = strtod(str, &ptr);
            if (str == ptr)
                goto failed;
            str = skip_forward(ptr);
        }
    }
    this.lines = allocator->allocate(allocator->state, sizeof(*this.lines) * this.n_lines);
    if (!this.lines)
        goto failed;
    //  Parse lines
    for (unsigned iln = 0; iln < this.n_lines; ++iln)
    {
        line_t *ln = this.lines + iln;
        ln->p1 = pack_id((int)strtol(str, &ptr, 10));
        if (str == ptr)
            goto failed;
        if (ln->p1.value != INVALID_ID && ln->p1.value > this.n_points)
            goto failed;
        str = skip_forward(ptr);
        ln->p2 = pack_id((int)strtol(str, &ptr, 10));
        if (str == ptr)
            goto failed;
        if (ln->p2.value != INVALID_ID && ln->p2.value > this.n_points)
            goto failed;
        str = skip_forward(ptr);
    }
    // save place where surfaces begin
    const char *const p_begin = str;

    // read surface data, but only focus on getting number of surfaces themselves
    this.surface_offsets = allocator->allocate(allocator->state, sizeof *this.surface_offsets * (this.n_surfaces + 1));
    if (!this.surface_offsets)
        goto failed;
    this.surface_offsets[0] = 0;
    unsigned total_lines = 0;
    for (unsigned is = 0; is < this.n_surfaces; ++is)
    {
        const unsigned n = strtoul(str, &ptr, 10);
        if (str == ptr)
            goto failed;
        str = skip_forward(ptr);
        total_lines += n;
        for (unsigned i = 0; i < n; ++i)
        {
            (void)strtol(str, &ptr, 10);
            if (str == ptr)
                goto failed;
            str = skip_forward(ptr);
        }
        this.surface_offsets[is + 1] = total_lines;
    }
    //  restore parsing state
    str = p_begin;

    this.surface_lines = allocator->allocate(allocator->state, sizeof(*this.surface_lines) * total_lines);
    if (!this.surface_lines)
        goto failed;

    for (unsigned is = 0; is < this.n_surfaces; ++is)
    {
        // Have to skip it, so just parse whatever is there
        (void)strtoul(str, &ptr, 10);
        if (str == ptr)
            goto failed;
        str = skip_forward(ptr);
        for (unsigned i = this.surface_offsets[is]; i < this.surface_offsets[is + 1]; ++i)
        {
            this.surface_lines[i] = pack_id((int)strtol(str, &ptr, 10));
            if (str == ptr)
                goto failed;
            str = skip_forward(ptr);
        }
    }
    *p_positions = positions;
    *p_out = this;
    return 0;

failed:
    allocator->deallocate(allocator->state, this.surface_offsets);
    allocator->deallocate(allocator->state, this.surface_lines);
    allocator->deallocate(allocator->state, this.lines);
    allocator->deallocate(allocator->state, positions);
    return -1;
}
