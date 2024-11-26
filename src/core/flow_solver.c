//
// Created by jan on 19.11.2024.
//

#include "flow_solver.h"

real3_t compute_line_induction(const real3_t control_point, const geo_id_t i_line, const mesh_t *mesh, const real_t tol)
{
    const unsigned pt1 = mesh->lines[i_line.value].p1.value, pt2 = mesh->lines[i_line.value].p2.value;
    const real3_t r1 = mesh->positions[pt1];
    const real3_t r2 = mesh->positions[pt2];
    real3_t dr1 = real3_sub(control_point, r1);
    real3_t dr2 = real3_sub(control_point, r2);
    real3_t direction = real3_sub(r2, r1);
    real_t len = real3_mag(direction);
    direction.v0 /= len;
    direction.v1 /= len;
    direction.v2 /= len;

    if (len < tol)
    {
        //  Filament is too short
        return (real3_t){};
    }

    real_t tan_dist1 = real3_dot(direction, dr1);
    real_t tan_dist2 = real3_dot(direction, dr2);

    real_t norm_dist1 = sqrt(real3_dot(dr1, dr1) - (tan_dist1 * tan_dist1));
    real_t norm_dist2 = sqrt(real3_dot(dr2, dr2) - (tan_dist2 * tan_dist2));

    real_t norm_dist = (norm_dist1 + norm_dist2) / 2.0;

    if (norm_dist < tol)
    {
        //  Normal distance is too small
        return (real3_t){};
        // continue
    }

    real_t vel_mag_half = 0.5 * M_1_PI / norm_dist * (atan2(tan_dist2, norm_dist) - atan2(tan_dist1, norm_dist));
    real3_t dr_avg = real3_mul1(real3_add(dr1, dr2), 0.5);
    real3_t vel_dir = real3_mul1(real3_cross(dr_avg, direction), vel_mag_half);

    if (i_line.orientation)
    {
        return real3_neg(vel_dir);
    }
    return vel_dir;
}

real3_t compute_surface_induction(const real3_t control_point, const geo_id_t i_surf, const mesh_t *mesh,
                                  const real_t tol)
{
    const surface_t *s = mesh->surfaces[i_surf.value];
    real3_t total = {0};
    for (unsigned i_ln = 0; i_ln < s->n_lines; ++i_ln)
    {
        const geo_id_t line = s->lines[i_ln];
        const real3_t v_line = compute_line_induction(control_point, line, mesh, tol);
        total = real3_add(total, v_line);
    }
    if (i_surf.orientation)
    {
        return real3_neg(total);
    }
    return total;
}

void compute_self_matrix(const mesh_t *mesh, const real_t tol, real_t *const mtx)
{
    const unsigned n = mesh->n_surfaces;
    for (unsigned i = 0; i < n; ++i)
    {
        const real3_t cp = surface_center(mesh, (geo_id_t){.value = i});
        const real3_t norm = surface_normal(mesh, (geo_id_t){.value = i});

        for (unsigned j = 0; j < n; ++j)
        {
            const real3_t v_ind = compute_surface_induction(cp, (geo_id_t){.value = j}, mesh, tol);
            mtx[i * n + j] = real3_dot(v_ind, norm);
        }
    }
}
