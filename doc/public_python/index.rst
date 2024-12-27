.. _pyvl.public_python:

Public Python API
=================

This part of the documentation describes sections of the API which are
intended for a typical user.

Positioning Geometry
--------------------

Before there can be any talk of defining geometry, it must be first positioned in space.
If geometry consists of several components, it might be useful to define those in their
own coordinate system. Even if only a single component is analyzed, it might be defined
in a different coordinate system than expected. Take for example the absolute lunatics
known as flight dynamics engineers. These lot would have you believe that the z-axis
for a plane ought to be pointing towards the ground, while the x-axis is positive from
the tail towards the plane's nose.

To be able to co-exist with such savages, the :class:`ReferenceFrame` object can be used.

.. autoclass:: pyvl.ReferenceFrame
    :members:
    :exclude-members: save, load





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
