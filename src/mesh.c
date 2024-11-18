//
// Created by jan on 16.11.2024.
//

#include "mesh.h"


void mesh_free(mesh_t* this, const allocator_t* allocator)
{
    // for (unsigned i = 0; i < this->n_surfaces; ++i)
    // {
    //     allocator->deallocate(allocator->state, this->surfaces[i]);
    // }
    allocator->deallocate(allocator->state, this->surfaces);
    allocator->deallocate(allocator->state, this->lines);
    allocator->deallocate(allocator->state, this->positions);
    allocator->deallocate(allocator->state, this);
}

real3_t line_direction(const mesh_t* mesh, geo_id_t line_id)
{
    const line_t *ln = mesh->lines + line_id.value;
    if (!line_id.orientation)
    {
        return real3_sub(mesh->positions[ln->p1.value], mesh->positions[ln->p2.value]);
    }
    return real3_sub(mesh->positions[ln->p2.value], mesh->positions[ln->p1.value]);
}

real3_t surface_center(const mesh_t *mesh, geo_id_t surface_id)
{
    const surface_t *surf = mesh->surfaces[surface_id.value];
    real3_t out = {};
    for (unsigned i_line = 0; i_line < surf->n_lines; ++i_line)
    {
        real3_t p;
        const geo_id_t ln_id = surf->lines[i_line];
        const line_t *ln = mesh->lines + ln_id.value;
        if (!ln_id.orientation)
        {
            p = mesh->positions[ln->p1.value];
        }
        else
        {
            p = mesh->positions[ln->p2.value];
        }
        out = real3_add(out, p);
    }
    const real_t div = 1.0 / (double)surf->n_lines;
    return (real3_t){{out.v0 * div, out.v1 * div, out.v2 * div}};
}

real3_t surface_normal(const mesh_t* mesh, geo_id_t surface_id)
{
    real3_t out = {};
    const surface_t *surf = mesh->surfaces[surface_id.value];
    real3_t d1 = line_direction(mesh, surf->lines[0]);
    for (unsigned i_line = 1; i_line < surf->n_lines; ++i_line)
    {
        const real3_t d2 = line_direction(mesh, surf->lines[i_line]);
        const real3_t normal = real3_unit(real3_cross(d1, d2));
        out = real3_add(out, normal);
        d1 = d2;
    }
    //  Add the last one
    const real3_t d2 = line_direction(mesh, surf->lines[0]);
    const real3_t normal = real3_unit(real3_cross(d1, d2));
    out = real3_unit(real3_add(out, normal));
    if (!surface_id.orientation)
        return out;
    return (real3_t){{-out.v0, -out.v1, -out.v2}};
}


mesh_t* mesh_dual_from_primal(const mesh_t* primal, const allocator_t* allocator)
{
    mesh_t *const dual = allocator->allocate(allocator->state, sizeof(*dual));
    if (!dual)
    {
        return nullptr;
    }

    dual->n_points   = primal->n_surfaces;
    dual->n_lines    = primal->n_lines;
    dual->n_surfaces = primal->n_points;

    dual->surfaces = nullptr;

    dual->positions = allocator->allocate(allocator->state, sizeof(*dual->positions) * dual->n_points);
    dual->lines = allocator->allocate(allocator->state, sizeof(*dual->lines) * dual->n_lines);

    if (!dual->positions || !dual->lines)
    {
        goto failed;
    }

    /*
     * Dual positions are arbitrary. In this case, they are chosen to be normal vectors of primal surfaces,
     * since it is more expensive to compute those than centers of elements.
     */

    for (unsigned i_surf = 0; i_surf < primal->n_surfaces; ++i_surf)
    {
        dual->positions[i_surf] = surface_normal(primal, (geo_id_t){.value = i_surf});
    }

    /* Dual line `i` contains ids of surfaces which contain line `i`. */
    for (unsigned i_line = 0; i_line < primal->n_lines; ++i_line)
    {
        geo_id_t surf_ids[2] = {{.orientation = 0, .value = INVALID_ID}, {.orientation = 0, .value = INVALID_ID}};
        unsigned cnt = 0;

        /* Check each surface, until two with the edge are found. */
        for (unsigned i_surf = 0; i_surf < primal->n_surfaces && cnt < 2; ++i_surf)
        {
            const surface_t *s = primal->surfaces[i_surf];
            for (unsigned i_surf_line = 0; i_surf_line < s->n_lines; ++i_surf_line)
            {
                if (s->lines[i_surf_line].value == i_line)
                {
                    surf_ids[cnt].value = i_surf;
                    cnt += 1;
                    break;
                }
            }
        }

        dual->lines[i_line] = (line_t){.p1 = surf_ids[0], .p2 = surf_ids[1]};
    }

    /*
     * Dual surfaces map to primal points and each consists of lines, which indicate what primal lines contain
     * these points.
     */

    size_t mem_sz = 0;
    for (unsigned i_pt = 0; i_pt < primal->n_points; ++i_pt)
    {
        unsigned cnt = 0;

        for (unsigned i_line = 0; i_line < primal->n_lines; ++i_line)
        {
            const line_t *ln = primal->lines + i_line;
            if (ln->p1.value == i_pt || ln->p2.value == i_pt)
            {
                cnt += 1;
            }
        }
        mem_sz += sizeof(surface_t) + cnt * sizeof(geo_id_t);
    }


    /* Allocate memory for dual surface behind their pointers (which are more like offsets) */
    surface_t **const surf_ptr = allocator->allocate(allocator->state, sizeof(*dual->surfaces) * dual->n_surfaces + mem_sz);
    if (!surf_ptr)
    {
        goto failed;
    }
    dual->surfaces = (const surface_t **)surf_ptr;
    geo_id_t *id_ptr = (geo_id_t *)(surf_ptr + dual->n_surfaces);
    for (unsigned i_pt = 0; i_pt < primal->n_points; ++i_pt)
    {
        unsigned cnt = 0;

        for (unsigned i_line = 0; i_line < primal->n_lines; ++i_line)
        {
            const line_t *ln = primal->lines + i_line;
            if (ln->p1.value == i_pt)
            {
                id_ptr[cnt + 1] = (geo_id_t){.orientation = 0, .value = i_line};
                cnt += 1;
            }
            else if(ln->p2.value == i_pt)
            {
                id_ptr[cnt + 1] = (geo_id_t){.orientation = 1, .value = i_line};
                cnt += 1;
            }
        }
        surf_ptr[i_pt] = (surface_t *)id_ptr;
        *(uint32_t *)id_ptr = cnt;
        id_ptr += cnt + 1;
    }

    return dual;

failed:

    allocator->deallocate(allocator->state, dual->positions);
    allocator->deallocate(allocator->state, dual->lines);
    allocator->deallocate(allocator->state, dual->surfaces);
    allocator->deallocate(allocator->state, dual);

    return nullptr;
}

mesh_t *mesh_from_elements(unsigned n_elements, const unsigned point_counts[static restrict n_elements],
    const unsigned flat_points[static restrict n_elements], const allocator_t *allocator)
{
    mesh_t *const this = allocator->allocate(allocator->state, sizeof(*this));
    if (!this)
    {
        return nullptr;
    }

    this->positions = nullptr;
    this->n_points = 0;
    this->n_surfaces = n_elements;
    this->n_lines = 0;
    unsigned all_points = 0;
    for (unsigned i_elm = 0; i_elm < n_elements; ++i_elm)
    {
        all_points += point_counts[i_elm]; // count points and non-unique lines, so maximum possible line count
    }

    this->lines = allocator->allocate(allocator->state, sizeof(*this->lines) * all_points);
    if (!this->lines)
    {
        allocator->deallocate(allocator->state, this);
        return nullptr;
    }

    surface_t **const surf_ptr = allocator->allocate(allocator->state,  n_elements * sizeof(*this->surfaces)+ sizeof(geo_id_t) * (n_elements + all_points));
    if (!surf_ptr)
    {
        allocator->deallocate(allocator->state, this->lines);
        allocator->deallocate(allocator->state, this);
        return nullptr;
    }
    this->surfaces = (const surface_t **)surf_ptr;
    geo_id_t *id_ptr = (geo_id_t *)(surf_ptr + n_elements);
    unsigned offset = 0;
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
            for (i_line = 0; i_line < this->n_lines; ++i_line)
            {
                const line_t *ln = this->lines + i_line;
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
            id_ptr[i_pt + 1] = (geo_id_t){.orientation = orient, .value = i_line};
            if (i_line == this->n_lines)
            {
                // no other line contains it yet, so make a new one
                this->lines[this->n_lines] = (line_t){.p1 = {.value=left}, .p2 =  {.value=right}};
                this->n_lines += 1;
            }
            left = right;
        }

        *(uint32_t *)id_ptr = n;
        offset += n;
    }

    return this;
}

