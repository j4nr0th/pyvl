.. _pyvl.private_c.mesh:

Mesh Data Structures
====================

The mesh module defines the fundamental data structures for representing geometry
as points, lines, and surfaces. It also provides operations for computing geometric
properties and constructing dual meshes.


Line Type
---------

.. c:type:: line_t

Represents a connection between two points.

.. code-block:: c

    typedef struct {
        geo_id_t p1;  // First point ID
        geo_id_t p2;  // Second point ID
    } line_t;

The point IDs can include orientation information for proper direction handling.


Mesh Structure
--------------

.. c:type:: mesh_t

The main data structure containing all geometric information about a discretized surface.

.. code-block:: c

    typedef struct {
        unsigned n_points;           // Number of mesh points
        unsigned n_lines;            // Number of mesh lines
        line_t *lines;               // Array of line connectivity
        unsigned n_surfaces;         // Number of surfaces
        unsigned *surface_offsets;   // Offsets into surface_lines array
        geo_id_t *surface_lines;     // Surface line IDs (with orientation)
    } mesh_t;

**Dual Mesh Concept:**

The mesh module supports creating a "dual mesh" where:
- Primal mesh points → Dual mesh surfaces (surface centers)
- Primal mesh lines → Dual mesh lines (surface adjacency)
- Primal mesh surfaces → Dual mesh points (edge midpoints)

This duality enables efficient neighbor-finding operations.


Mesh Operations
---------------

.. c:function:: real3_t line_direction(const real3_t *restrict positions, const mesh_t *mesh, geo_id_t line_id)

   Compute the direction vector of a line. By setting ``line_id.orientation != 0``,
   the direction can be reversed.

   :param positions: Array of mesh point positions
   :param mesh: Mesh containing the geometry
   :param line_id: ID of the line to compute direction for
   :return: Vector from beginning to end of the line

.. c:function:: real3_t surface_center(const real3_t *restrict positions, const mesh_t *mesh, geo_id_t surface_id)

   Compute the centroid (center) of a surface element.

   :param positions: Array of mesh point positions
   :param mesh: Mesh containing the geometry
   :param surface_id: ID of the surface
   :return: Position vector of the surface center

.. c:function:: real3_t surface_normal(const real3_t *restrict positions, const mesh_t *mesh, geo_id_t surface_id)

   Compute the unit normal vector of a surface. By setting ``surface_id.orientation != 0``,
   the normal direction can be flipped.

   The normal is computed using the cross product of edge vectors, providing a robust
   result even for non-planar quadrilateral elements.

   :param positions: Array of mesh point positions
   :param mesh: Mesh containing the geometry
   :param surface_id: ID of the surface
   :return: Unit normal vector of the surface

.. c:function:: int mesh_dual_from_primal(mesh_t *p_out, const mesh_t *primal, const allocator_t *allocator)

   Create a dual mesh from a primal mesh. The dual mesh encodes surface adjacency
   information, enabling efficient neighbor-finding queries.

   :param p_out: Pointer to receive the resulting dual mesh
   :param primal: The primal mesh to create dual from
   :param allocator: Memory allocator for mesh allocation
   :return: 0 on success, -1 on failure

.. c:function:: int mesh_from_elements(mesh_t *p_out, unsigned n_elements, const unsigned *point_counts, const unsigned *flat_points, const allocator_t *allocator)

   Create a mesh from element connectivity data. This is the common format used
   by meshing tools like GMSH.

   :param p_out: Pointer to receive the resulting mesh
   :param n_elements: Number of elements
   :param point_counts: Array specifying point count per element
   :param flat_points: Flattened array of point indices for all elements
   :param allocator: Memory allocator
   :return: 0 on success, -1 on failure

.. c:function:: unsigned mesh_to_elements(const mesh_t *mesh, unsigned **p_point_counts, unsigned **p_flat_points, const allocator_t *allocator)

   Convert a mesh to element format. This is the inverse of :c:func:`mesh_from_elements`.

   :param mesh: Mesh to convert
   :param p_point_counts: Pointer to receive point counts per element array
   :param p_flat_points: Pointer to receive flattened point indices array
   :param allocator: Memory allocator
   :return: Number of elements, or 0 on failure

.. c:function:: void mesh_free(mesh_t *this, const allocator_t *allocator)

   Release all memory associated with a mesh.

   :param this: Mesh to deallocate
   :param allocator: Allocator used for the mesh (must match allocation allocator)
