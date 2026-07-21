//
// Created by jan on 19.11.2024.
//

#ifndef FLOW_SOLVER_H
#define FLOW_SOLVER_H

#include "solver_state.h"
#include "transformation.h"

/**
 * @brief Compute the velocity induced by a single vortex line at a control point.
 *
 * Biot-Savart calculation for a finite vortex line.
 *
 * @param positions Array of mesh point positions.
 * @param control_point Point at which to compute induced velocity.
 * @param i_line ID of the line (includes orientation).
 * @param mesh Mesh containing the line.
 * @param vortex_cutoff Distance threshold below which induction is zeroed.
 * @param vortex_smallest_size Minimum line length; shorter lines are treated as zero.
 * @return Induced velocity vector.
 */
real3_t compute_mesh_line_induction(const real3_t *restrict positions, real3_t control_point, geo_id_t i_line,
                                    const mesh_t *mesh, real_t vortex_cutoff, real_t vortex_smallest_size);

/**
 * @brief Compute the velocity induced by a vortex surface at a control point.
 *
 * The surface is treated as a vortex sheet with circulation distributed across its edges.
 *
 * @param positions Array of mesh point positions.
 * @param control_point Point at which to compute induced velocity.
 * @param i_surf ID of the surface.
 * @param mesh Mesh containing the surface.
 * @param vortex_cutoff Distance threshold for singularity handling.
 * @param vortex_smallest_size Minimum line length threshold.
 * @return Induced velocity vector.
 */
real3_t compute_mesh_surface_induction(const real3_t *restrict positions, real3_t control_point, geo_id_t i_surf,
                                       const mesh_t *mesh, real_t vortex_cutoff, real_t vortex_smallest_size);

/**
 * @brief Compute the self-induction matrix for all surface control points.
 *
 * This matrix represents how each surface induces velocity at every other surface.
 *
 * @param positions Array of mesh point positions.
 * @param mesh Mesh containing the surfaces.
 * @param vortex_cutoff Distance threshold for singularity handling.
 * @param vortex_smallest_size Minimum line length threshold.
 * @param mtx Output matrix (size: n_surfaces x n_surfaces x 3).
 */
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

/**
 * @brief Compute induced velocities from multiple lines to multiple control points.
 *
 * Supports parallel execution via OpenMP.
 *
 * @param n_lines Number of lines.
 * @param lines Array of line data.
 * @param n_positions Number of mesh point positions.
 * @param positions Array of mesh point positions.
 * @param n_cpts Number of control points.
 * @param cpts Array of control point positions.
 * @param out Output array (size: n_lines x n_cpts).
 * @param vortex_cutoff Distance threshold.
 * @param vortex_far_approximation Threshold for far-field arctan approximation.
 * @param vortex_smallest_size Minimum line length threshold.
 * @param n_threads Number of OpenMP threads (0 = use default).
 */
void compute_line_induction(unsigned n_lines, const line_t CVL_ARRAY_ARG(lines, static restrict n_lines),
                            unsigned n_positions, const real3_t CVL_ARRAY_ARG(positions, static restrict n_positions),
                            unsigned n_cpts, const real3_t CVL_ARRAY_ARG(cpts, static restrict n_cpts),
                            real3_t CVL_ARRAY_ARG(out, restrict n_lines *n_cpts), real_t vortex_cutoff,
                            real_t vortex_far_approximation, real_t vortex_smallest_size, unsigned n_threads);

/**
 * @brief Compute induced velocities with symmetry plane reflection.
 *
 * Same as compute_line_induction but includes image sources reflected across a symmetry plane.
 *
 * @param n_lines Number of lines.
 * @param lines Array of line data.
 * @param n_positions Number of mesh point positions.
 * @param positions Array of mesh point positions.
 * @param n_cpts Number of control points.
 * @param cpts Array of control point positions.
 * @param out Output array (size: n_lines x n_cpts).
 * @param vortex_cutoff Distance threshold.
 * @param vortex_far_approximation Threshold for far-field arctan approximation.
 * @param vortex_smallest_size Minimum line length threshold.
 * @param symmetry_plane Symmetry plane for image reflection.
 * @param n_threads Number of OpenMP threads (0 = use default).
 */
void compute_line_induction_symmetry(unsigned n_lines, const line_t CVL_ARRAY_ARG(lines, static restrict n_lines),
                                     unsigned n_positions,
                                     const real3_t CVL_ARRAY_ARG(positions, static restrict n_positions),
                                     unsigned n_cpts, const real3_t CVL_ARRAY_ARG(cpts, static restrict n_cpts),
                                     real3_t CVL_ARRAY_ARG(out, restrict n_lines *n_cpts), real_t vortex_cutoff,
                                     real_t vortex_far_approximation, real_t vortex_smallest_size,
                                     const transformation_plane_t *symmetry_plane, unsigned n_threads);

/**
 * @brief Aggregate line induction contributions to surface induction.
 *
 * Each surface's induction is computed by summing contributions from its bounding lines.
 *
 * @param n_surfaces Number of surfaces.
 * @param surface_offsets Offsets into surface_lines array.
 * @param surface_lines Surface line IDs (with orientation).
 * @param n_lines Number of lines.
 * @param n_cpts Number of control points.
 * @param line_inductions Pre-computed line inductions.
 * @param out Output surface inductions.
 * @param n_threads Number of threads for parallel execution.
 */
void line_induction_to_surface_induction(unsigned n_surfaces,
                                         const unsigned CVL_ARRAY_ARG(surface_offsets, static restrict n_surfaces + 1),
                                         const geo_id_t CVL_ARRAY_ARG(surface_lines, restrict), unsigned n_lines,
                                         unsigned n_cpts,
                                         const real3_t CVL_ARRAY_ARG(line_inductions, static restrict n_lines *n_cpts),
                                         real3_t CVL_ARRAY_ARG(out, restrict n_surfaces *n_cpts), unsigned n_threads);

/**
 * @brief Compute normal (perpendicular) component of line induction on surfaces.
 *
 * Useful for the no-penetration boundary condition.
 *
 * @param n_surfaces Number of surfaces.
 * @param surface_offsets Offsets into surface_lines array.
 * @param surface_lines Surface line IDs.
 * @param n_lines Number of lines.
 * @param n_cpts Number of control points.
 * @param normal_vectors Unit normals for each surface.
 * @param line_inductions Line induction vectors.
 * @param out Output scalar values (dot products).
 * @param n_threads Thread count.
 */
void line_induction_to_normal_surface_induction(
    unsigned n_surfaces, const unsigned CVL_ARRAY_ARG(surface_offsets, static restrict n_surfaces + 1),
    const geo_id_t CVL_ARRAY_ARG(surface_lines, restrict), unsigned n_lines, unsigned n_cpts,
    const real3_t CVL_ARRAY_ARG(normal_vectors, static restrict n_cpts),
    const real3_t CVL_ARRAY_ARG(line_inductions, static restrict n_lines *n_cpts),
    real_t CVL_ARRAY_ARG(out, restrict n_surfaces *n_cpts), unsigned n_threads);

/**
 * @brief Compute forces on mesh lines from surface circulations.
 *
 * Uses the Kutta-Joukowski theorem: force on each line is proportional to the circulation
 * and the relative velocity.
 *
 * @param positions Mesh point positions.
 * @param primal Primal mesh (surfaces).
 * @param dual Dual mesh (adjacency).
 * @param surface_circulations Circulation for each surface.
 * @param line_forces Output force vectors for each line.
 */
void line_forces_from_surface_circulation(const real3_t CVL_ARRAY_ARG(positions, restrict), const mesh_t *primal,
                                          const mesh_t *dual,
                                          const real_t CVL_ARRAY_ARG(surface_circulations, restrict),
                                          real3_t CVL_ARRAY_ARG(line_forces, restrict));

#endif // FLOW_SOLVER_H
