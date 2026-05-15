.. _pyvl.private_c.induction:

Induction Algorithms
====================

The flow solver module contains functions for computing the velocity induction
from vortex elements. These algorithms form the computational core of the panel
method solver.


Fundamental Induction Functions
-------------------------------

.. c:function:: real3_t compute_mesh_line_induction(const real3_t *restrict positions, real3_t control_point, geo_id_t i_line, const mesh_t *mesh, real_t tol)

   Compute the velocity induced by a single vortex line at a control point.
   This is the Biot-Savart calculation for a finite vortex line.

   :param positions: Array of mesh point positions
   :param control_point: Point at which to compute induced velocity
   :param i_line: ID of the line (includes orientation)
   :param mesh: Mesh containing the line
   :param tol: Distance threshold below which induction is zeroed
   :return: Induced velocity vector

.. c:function:: real3_t compute_mesh_surface_induction(const real3_t *restrict positions, real3_t control_point, geo_id_t i_surf, const mesh_t *mesh, real_t tol)

   Compute the velocity induced by a vortex surface at a control point.
   The surface is treated as a vortex sheet with circulation distributed
   across its edges.

   :param positions: Array of mesh point positions
   :param control_point: Point at which to compute induced velocity
   :param i_surf: ID of the surface
   :param mesh: Mesh containing the surface
   :param tol: Distance threshold for singularity handling
   :return: Induced velocity vector


Matrix Computation Functions
----------------------------

These functions compute full induction matrices for the system:

.. c:function:: void compute_mesh_self_matrix(const real3_t *restrict positions, const mesh_t *mesh, real_t tol, real_t *mtx)

   Compute the self-induction matrix for all surface control points.
   This matrix represents how each surface induces velocity at every other surface.

   :param positions: Array of mesh point positions
   :param mesh: Mesh containing the surfaces
   :param tol: Distance threshold for singularity handling
   :param mtx: Output matrix (size: n_surfaces × n_surfaces × 3)

.. c:function:: void compute_line_induction(unsigned n_lines, const line_t *lines, unsigned n_positions, const real3_t *positions, unsigned n_cpts, const real3_t *cpts, real3_t *out, real_t tol, unsigned n_threads)

   Compute induced velocities from multiple lines to multiple control points.
   Supports parallel execution via OpenMP.

   :param n_lines: Number of lines
   :param lines: Array of line data
   :param n_positions: Number of mesh point positions
   :param positions: Array of mesh point positions
   :param n_cpts: Number of control points
   :param cpts: Array of control point positions
   :param out: Output array (size: n_lines × n_cpts)
   :param tol: Distance threshold
   :param n_threads: Number of OpenMP threads (0 = use default)


Surface Induction Aggregation
-----------------------------

.. c:function:: void line_induction_to_surface_induction(unsigned n_surfaces, const unsigned *surface_offsets, const geo_id_t *surface_lines, unsigned n_lines, unsigned n_cpts, const real3_t *line_inductions, real3_t *out, unsigned n_threads)

   Aggregate line induction contributions to surface induction.
   Each surface's induction is computed by summing contributions from its bounding lines.

   :param n_surfaces: Number of surfaces
   :param surface_offsets: Offsets into surface_lines array
   :param surface_lines: Surface line IDs (with orientation)
   :param n_lines: Number of lines
   :param n_cpts: Number of control points
   :param line_inductions: Pre-computed line inductions
   :param out: Output surface inductions
   :param n_threads: Number of threads for parallel execution

.. c:function:: void line_induction_to_normal_surface_induction(unsigned n_surfaces, const unsigned *surface_offsets, const geo_id_t *surface_lines, unsigned n_lines, unsigned n_cpts, const real3_t *normal_vectors, const real3_t *line_inductions, real_t *out, unsigned n_threads)

   Compute normal (perpendicular) component of line induction on surfaces.
   This is useful for the no-penetration boundary condition.

   :param n_surfaces: Number of surfaces
   :param surface_offsets: Offsets into surface_lines array
   :param surface_lines: Surface line IDs
   :param n_lines: Number of lines
   :param n_cpts: Number of control points
   :param normal_vectors: Unit normals for each surface
   :param line_inductions: Line induction vectors
   :param out: Output scalar values (dot products)
   :param n_threads: Thread count


Force Computation
-----------------

.. c:function:: void line_forces_from_surface_circulation(const real3_t *positions, const mesh_t *primal, const mesh_t *dual, const real_t *surface_circulations, real3_t *line_forces)

   Compute forces on mesh lines from surface circulations using the Kutta-Joukowski theorem.
   The force on each line is proportional to the circulation and the relative velocity.

   :param positions: Mesh point positions
   :param primal: Primal mesh (surfaces)
   :param dual: Dual mesh (adjacency)
   :param surface_circulations: Circulation for each surface
   :param line_forces: Output force vectors for each line