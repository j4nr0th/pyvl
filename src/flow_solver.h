//
// Created by jan on 19.11.2024.
//

#ifndef FLOW_SOLVER_H
#define FLOW_SOLVER_H

#include "solver_state.h"

real3_t compute_line_induction(real3_t control_point, geo_id_t i_line, const mesh_t* mesh, real_t tol);

real3_t compute_surface_induction(real3_t control_point, geo_id_t i_surf, const mesh_t* mesh, real_t tol);

void compute_self_matrix(const mesh_t *mesh, real_t tol, real_t *mtx);

#endif //FLOW_SOLVER_H
