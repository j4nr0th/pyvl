//
// Created by jan on 19.11.2024.
//

#ifndef FLOW_SOLVER_H
#define FLOW_SOLVER_H

#include "solver_state.h"

real3_t compute_mesh_line_induction(const real3_t *restrict positions, real3_t control_point, geo_id_t i_line,
                                    const mesh_t *mesh, real_t tol);

real3_t compute_mesh_surface_induction(const real3_t *restrict positions, real3_t control_point, geo_id_t i_surf,
                                       const mesh_t *mesh, real_t tol);

void compute_mesh_self_matrix(const real3_t *restrict positions, const mesh_t *mesh, real_t tol, real_t *mtx);

void compute_line_induction(unsigned n_lines, const line_t lines[static restrict n_lines], unsigned n_positions,
                            const real3_t positions[static restrict n_positions], unsigned n_cpts,
                            const real3_t cpts[static restrict n_cpts], real3_t out[restrict n_lines * n_cpts],
                            real_t tol);

void line_induction_to_surface_induction(unsigned n_surfaces,
                                         const unsigned surface_offsets[static restrict n_surfaces + 1],
                                         const geo_id_t surface_lines[restrict], unsigned n_lines, unsigned n_cpts,
                                         const real3_t line_inductions[static restrict n_lines * n_cpts],
                                         real3_t out[restrict n_surfaces * n_cpts]);

void line_induction_to_normal_surface_induction(unsigned n_surfaces,
                                                const unsigned surface_offsets[static restrict n_surfaces + 1],
                                                const geo_id_t surface_lines[restrict], unsigned n_lines,
                                                unsigned n_cpts, const real3_t normal_vectors[static restrict n_cpts],
                                                const real3_t line_inductions[static restrict n_lines * n_cpts],
                                                real_t out[restrict n_surfaces * n_cpts]);

void line_forces_from_surface_circulation(const real3_t positions[restrict], const mesh_t *primal, const mesh_t *dual,
                                          const real_t surface_circulations[restrict], real3_t line_forces[restrict]);

#endif // FLOW_SOLVER_H
