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

.. c:autodoc:: mesh_io.h
