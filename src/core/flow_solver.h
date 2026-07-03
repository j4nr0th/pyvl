//
// Created by jan on 19.11.2024.
//

#ifndef FLOW_SOLVER_H
#define FLOW_SOLVER_H

#include "solver_state.h"
#include "transformation.h"

real3_t compute_mesh_line_induction(const real3_t *restrict positions, real3_t control_point, geo_id_t i_line,
                                    const mesh_t *mesh, real_t vortex_cutoff, real_t vortex_smallest_size);

real3_t compute_mesh_surface_induction(const real3_t *restrict positions, real3_t control_point, geo_id_t i_surf,
                                       const mesh_t *mesh, real_t vortex_cutoff, real_t vortex_smallest_size);

void compute_mesh_self_matrix(const real3_t *restrict positions, const mesh_t *mesh, real_t vortex_cutoff,
                              real_t vortex_smallest_size, real_t *mtx);

/**
 * Compute the velocity induced by the vorticity filament.
 *
 * @param vortex_cutoff Minimum distance before we clamp the result to zero.
 * @param vortex_far_approximation Threshold under which approximation to arctan is used.
 * @param r1 Starting point of the filament.
 * @param r2 End point of the filament.
 * @param direction Unit vector from r1 to r2.
 * @param control_point Position where the effect of filament should be computed.
 * @return Velocity induced by the vorticity filament at the control point.
 */
real3_t compute_filament_induction(real_t vortex_cutoff, real_t vortex_far_approximation, real3_t r1, real3_t r2,
                                   real3_t direction, real3_t control_point);

void compute_line_induction(unsigned n_lines, const line_t CVL_ARRAY_ARG(lines, static restrict n_lines),
                            unsigned n_positions, const real3_t CVL_ARRAY_ARG(positions, static restrict n_positions),
                            unsigned n_cpts, const real3_t CVL_ARRAY_ARG(cpts, static restrict n_cpts),
                            real3_t CVL_ARRAY_ARG(out, restrict n_lines *n_cpts), real_t vortex_cutoff,
                            real_t vortex_far_approximation, real_t vortex_smallest_size, unsigned n_threads);

void compute_line_induction_symmetry(unsigned n_lines, const line_t CVL_ARRAY_ARG(lines, static restrict n_lines),
                                     unsigned n_positions,
                                     const real3_t CVL_ARRAY_ARG(positions, static restrict n_positions),
                                     unsigned n_cpts, const real3_t CVL_ARRAY_ARG(cpts, static restrict n_cpts),
                                     real3_t CVL_ARRAY_ARG(out, restrict n_lines *n_cpts), real_t vortex_cutoff,
                                     real_t vortex_far_approximation, real_t vortex_smallest_size,
                                     const transformation_plane_t *symmetry_plane, unsigned n_threads);

void line_induction_to_surface_induction(unsigned n_surfaces,
                                         const unsigned CVL_ARRAY_ARG(surface_offsets, static restrict n_surfaces + 1),
                                         const geo_id_t CVL_ARRAY_ARG(surface_lines, restrict), unsigned n_lines,
                                         unsigned n_cpts,
                                         const real3_t CVL_ARRAY_ARG(line_inductions, static restrict n_lines *n_cpts),
                                         real3_t CVL_ARRAY_ARG(out, restrict n_surfaces *n_cpts), unsigned n_threads);

void line_induction_to_normal_surface_induction(
    unsigned n_surfaces, const unsigned CVL_ARRAY_ARG(surface_offsets, static restrict n_surfaces + 1),
    const geo_id_t CVL_ARRAY_ARG(surface_lines, restrict), unsigned n_lines, unsigned n_cpts,
    const real3_t CVL_ARRAY_ARG(normal_vectors, static restrict n_cpts),
    const real3_t CVL_ARRAY_ARG(line_inductions, static restrict n_lines *n_cpts),
    real_t CVL_ARRAY_ARG(out, restrict n_surfaces *n_cpts), unsigned n_threads);

void line_forces_from_surface_circulation(const real3_t CVL_ARRAY_ARG(positions, restrict), const mesh_t *primal,
                                          const mesh_t *dual,
                                          const real_t CVL_ARRAY_ARG(surface_circulations, restrict),
                                          real3_t CVL_ARRAY_ARG(line_forces, restrict));

#endif // FLOW_SOLVER_H
