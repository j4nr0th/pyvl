//
// Created by jan on 16.11.2024.
//

#include "mesh.h"

void mesh_free(mesh_t *this, const allocator_t *allocator)
{
    allocator->deallocate(allocator->state, this->surface_lines);
    allocator->deallocate(allocator->state, this->surface_offsets);
    allocator->deallocate(allocator->state, this->lines);
}

real3_t line_direction(const real3_t *positions, const mesh_t *mesh, geo_id_t line_id)
{
    const line_t *ln = mesh->lines + line_id.value;
    if (!line_id.orientation)
    {
        return real3_sub(positions[ln->p1.value], positions[ln->p2.value]);
    }
    return real3_sub(positions[ln->p2.value], positions[ln->p1.value]);
}

real3_t surface_center(const real3_t *positions, const mesh_t *mesh, geo_id_t surface_id)
{
    real3_t out = {};
    for (unsigned i_line = mesh->surface_offsets[surface_id.value];
         i_line < mesh->surface_offsets[surface_id.value + 1]; ++i_line)
    {
        real3_t p;
        const geo_id_t ln_id = mesh->surface_lines[i_line];
        const line_t *ln = mesh->lines + ln_id.value;
        if (!ln_id.orientation)
        {
            p = positions[ln->p1.value];
        }
        else
        {
            p = positions[ln->p2.value];
        }
        out = real3_add(out, p);
    }
    const real_t div =
        1.0 / (double)(mesh->surface_offsets[surface_id.value + 1] - mesh->surface_offsets[surface_id.value]);
    return (real3_t){{out.v0 * div, out.v1 * div, out.v2 * div}};
}

real3_t surface_normal(const real3_t *positions, const mesh_t *mesh, geo_id_t surface_id)
{
    real3_t out = {};
    const unsigned i0 = mesh->surface_offsets[surface_id.value];
    const unsigned i1 = mesh->surface_offsets[surface_id.value + 1];
    real3_t d1 = line_direction(positions, mesh, mesh->surface_lines[i0]);
    const real3_t dr0 = d1;
    for (unsigned i_line = i0 + 1; i_line < i1; ++i_line)
    {
        const real3_t d2 = line_direction(positions, mesh, mesh->surface_lines[i_line]);
        const real3_t normal = real3_cross(d2, d1);
        out = real3_add(out, normal);
        d1 = d2;
    }
    //  Add the last one
    const real3_t normal = real3_cross(dr0, d1);
    out = real3_unit(real3_add(out, normal));
    if (!surface_id.orientation)
        return out;
    return (real3_t){{-out.v0, -out.v1, -out.v2}};
}

int mesh_dual_from_primal(mesh_t *p_out, const mesh_t *primal, const allocator_t *allocator)
{
    mesh_t dual;

    dual.n_points = primal->n_surfaces;
    dual.n_lines = primal->n_lines;
    dual.n_surfaces = primal->n_points;

    dual.surface_offsets = nullptr;
    dual.surface_lines = nullptr;

    dual.lines = allocator->allocate(allocator->state, sizeof(*dual.lines) * dual.n_lines);

    if (!dual.lines)
    {
        return -1;
    }

    /* Dual line `i` contains ids of surfaces which contain line `i`. */
    for (unsigned i_line = 0; i_line < primal->n_lines; ++i_line)
    {
        geo_id_t surf_ids[2] = {{.orientation = 0, .value = INVALID_ID}, {.orientation = 0, .value = INVALID_ID}};
        unsigned cnt = 0;

        /* Check each surface, until two with the edge are found. */
        for (unsigned i_surf = 0; i_surf < primal->n_surfaces && cnt < 2; ++i_surf)
        {
            for (unsigned i_surf_line = primal->surface_offsets[i_surf];
                 i_surf_line < primal->surface_offsets[i_surf + 1]; ++i_surf_line)
            {
                const geo_id_t line_id = primal->surface_lines[i_surf_line];
                if (line_id.value == i_line)
                {
                    surf_ids[!(line_id.orientation)].value = i_surf;
                    cnt += 1;
                    break;
                }
            }
        }

        dual.lines[i_line] = (line_t){.p1 = surf_ids[0], .p2 = surf_ids[1]};
    }

    /*
     * Dual surfaces map to primal points and each consists of lines, which indicate what primal lines contain
     * these points.
     */

    unsigned line_count = 0;
    for (unsigned i_pt = 0; i_pt < primal->n_points; ++i_pt)
    {
        unsigned cnt = 0;

        for (unsigned i_line = 0; i_line < primal->n_lines; ++i_line)
        {
            const line_t *ln = primal->lines + i_line;
            cnt += (ln->p1.value == i_pt) + (ln->p2.value == i_pt);
        }
        line_count += cnt;
    }

    /* Allocate memory for dual surface offsets and lines */
    geo_id_t *const surface_lines = allocator->allocate(allocator->state, line_count * sizeof(*surface_lines));
    unsigned *const surface_offsets =
        allocator->allocate(allocator->state, (dual.n_surfaces + 1) * sizeof(*surface_offsets));

    if (!surface_lines || !surface_offsets)
    {
        allocator->deallocate(allocator->state, dual.lines);
        allocator->deallocate(allocator->state, surface_offsets);
        allocator->deallocate(allocator->state, surface_lines);

        return -1;
    }
    dual.surface_offsets = surface_offsets;
    dual.surface_lines = surface_lines;
    unsigned offset = 0;
    surface_offsets[0] = 0;
    for (unsigned i_pt = 0; i_pt < primal->n_points; ++i_pt)
    {
        unsigned cnt = 0;

        for (unsigned i_line = 0; i_line < primal->n_lines; ++i_line)
        {
            const line_t *ln = primal->lines + i_line;
            if (ln->p1.value == i_pt)
            {
                surface_lines[offset + cnt] = (geo_id_t){.orientation = 0, .value = i_line};
                cnt += 1;
            }
            if (ln->p2.value == i_pt)
            {
                surface_lines[offset + cnt] = (geo_id_t){.orientation = 1, .value = i_line};
                cnt += 1;
            }
        }

        offset += cnt;
        surface_offsets[i_pt + 1] = offset;
    }

    *p_out = dual;
    return 0;
}

int mesh_from_elements(mesh_t *p_out, unsigned n_elements, const unsigned point_counts[static restrict n_elements],
                       const unsigned flat_points[restrict], const allocator_t *allocator)
{
    mesh_t this;

    this.n_points = 0;
    this.n_surfaces = n_elements;
    this.n_lines = 0;
    unsigned all_points = 0;
    for (unsigned i_elm = 0; i_elm < n_elements; ++i_elm)
    {
        all_points += point_counts[i_elm]; // count points and non-unique lines, so maximum possible line count
    }

    this.lines = allocator->allocate(allocator->state, sizeof(*this.lines) * all_points);
    geo_id_t *const surface_lines = allocator->allocate(allocator->state, all_points * sizeof(*surface_lines));
    unsigned *const surface_offsets =
        allocator->allocate(allocator->state, (this.n_surfaces + 1) * sizeof(*surface_offsets));

    if (!this.lines || !surface_lines || !surface_offsets)
    {
        allocator->deallocate(allocator->state, surface_offsets);
        allocator->deallocate(allocator->state, surface_lines);
        allocator->deallocate(allocator->state, this.lines);
        return -1;
    }

    this.surface_offsets = surface_offsets;
    this.surface_lines = surface_lines;

    unsigned offset = 0;
    surface_offsets[0] = 0;
    for (unsigned i_elm = 0; i_elm < n_elements; ++i_elm)
    {
        const unsigned n = point_counts[i_elm];
        unsigned left = flat_points[offset + n - 1];
        for (unsigned i_pt = 0; i_pt < n; ++i_pt)
        {
            unsigned orient = 0;
            const unsigned right = flat_points[offset + i_pt];
            //  Check if a line with these points already exists.
            unsigned i_line;
            for (i_line = 0; i_line < this.n_lines; ++i_line)
            {
                const line_t *ln = this.lines + i_line;
                if (ln->p1.value == left && ln->p2.value == right)
                {
                    orient = 0;
                    break;
                }
                if (ln->p2.value == left && ln->p1.value == right)
                {
                    orient = 1;
                    break;
                }
            }
            surface_lines[offset + i_pt] = (geo_id_t){.orientation = orient, .value = i_line};
            if (i_line == this.n_lines)
            {
                // no other line contains it yet, so make a new one
                this.lines[this.n_lines] = (line_t){.p1 = {.value = left}, .p2 = {.value = right}};
                this.n_lines += 1;
            }
            left = right;
        }
        offset += n;
        surface_offsets[i_elm + 1] = offset;
    }

    *p_out = this;
    return 0;
}

unsigned mesh_to_elements(const mesh_t *mesh, unsigned **p_point_counts, unsigned **p_flat_points,
                          const allocator_t *allocator)
{
    const unsigned n_surfaces = mesh->n_surfaces;
    if (n_surfaces == 0)
    {
        return 0;
    }
    unsigned *const point_counts = allocator->allocate(allocator->state, sizeof(*point_counts) * n_surfaces);
    if (!point_counts)
    {
        return 0;
    }
    unsigned total_points = 0;
    for (unsigned i = 0; i < n_surfaces; ++i)
    {
        const unsigned n_lines = (mesh->surface_offsets[i + 1] - mesh->surface_offsets[i]);
        point_counts[i] = n_lines;
        total_points += n_lines;
    }

    unsigned *const flat_points = allocator->allocate(allocator->state, sizeof(*flat_points) * total_points);
    if (!flat_points)
    {
        allocator->deallocate(allocator->state, point_counts);
        return 0;
    }

    unsigned *p = flat_points;
    for (unsigned i = 0; i < n_surfaces; ++i)
    {
        for (unsigned j = mesh->surface_offsets[i]; j < mesh->surface_offsets[i + 1]; ++j)
        {
            const geo_id_t line_id = mesh->surface_lines[j];
            unsigned pt;
            if (line_id.orientation)
            {
                pt = mesh->lines[line_id.value].p1.value;
            }
            else
            {
                pt = mesh->lines[line_id.value].p2.value;
            }
            *p = pt;
            p += 1;
        }
    }

    *p_point_counts = point_counts;
    *p_flat_points = flat_points;

    return n_surfaces;
}
