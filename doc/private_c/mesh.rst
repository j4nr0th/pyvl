.. _pyvl.private_c.mesh:

Mesh Data Structures
====================

The mesh module defines the fundamental data structures for representing geometry
as points, lines, and surfaces. It also provides operations for computing geometric
properties and constructing dual meshes.


**Dual Mesh Concept:**

The mesh module supports creating a "dual mesh" where:
- Primal mesh points → Dual mesh surfaces (surface centers)
- Primal mesh lines → Dual mesh lines (surface adjacency)
- Primal mesh surfaces → Dual mesh points (edge midpoints)

This duality enables efficient neighbor-finding operations.


Mesh Operations
---------------

.. c:autodoc:: mesh.h
