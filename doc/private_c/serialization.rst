.. _pyvl.private_c.serialization:

Mesh and Reference Frame Serialization
======================================

The mesh I/O module provides functions for serializing and deserializing mesh
data structures. The format supports points, lines, and surface connectivity
in a compact, text-based representation.

Additionally, the :class:`ReferenceFrame` object can be serialized and
deserialized using the `save` and `load` methods, which use a `HirearchicalMap`
to store the position, velocity, orientation, and rotation properties.



File Format Specification
-------------------------

The mesh format (version 0) is a text-based format with the following structure:

.. code-block:: text

    [version number]
    [number of points] [number of lines] [number of elements]
    [x of point 1] [y of point 1] [z of point 1]
        ...
    [x of point n] [y of point n] [z of point n]
    [point 1 for line 1] [point 2 for line 2]
        ...
    [point 1 for line m] [point 2 for line m]
    [number of lines in surface 1] [line 1 for surface 1] ... [line l for surface 1]
        ....
    [number of lines in surface k] [line 1 for surface k] ... [line p for surface k]

All whitespace is ignored, with no difference between newlines and spaces.


Serialization Functions
-----------------------

.. c:function:: char *serialize_mesh(const mesh_t *this, const real3_t *positions, const allocator_t *allocator)

   Convert a mesh into a null-terminated UTF-8 string representation.

   :param this: Mesh to serialize
   :param positions: Array of mesh point positions
   :param allocator: Allocator for string memory (can be a stack allocator)
   :return: Pointer to the serialized string, or NULL on failure. The caller
            is responsible for freeing this memory using the provided allocator's
            deallocate function.

.. c:function:: int deserialize_mesh(mesh_t *p_out, real3_t **p_positions, const char *str, const allocator_t *allocator)

   Parse a serialized mesh string and reconstruct the mesh structure.

   :param p_out: Pointer to receive the deserialized mesh
   :param p_positions: Pointer to receive the position array
   :param str: String containing the serialized mesh data
   :param allocator: Allocator for mesh memory
   :return: 0 on success, -1 on failure. On failure, the error location can
            be determined by checking the string position where parsing failed
            (the function stops at the first error).
