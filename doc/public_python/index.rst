.. _pyvl.public_python:

Public Python API
=================

This part of the documentation describes sections of the API which are
intended for a typical user.

Defining Geometry
-----------------

The most basic building block is the :class:`Geometry` object. Since
``PyVL`` is not a meshing library, the expectation is that the mesh will
be created with some other program, then loaded by either using
:class:`pyvista.PolyData` or :class:`meshio.Mesh`.

The class itself is defined as such:


.. autoclass:: pyvl.Geometry
    :members:
    :member-order: bysource
    :exclude-members: save, load
