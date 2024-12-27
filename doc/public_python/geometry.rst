.. _pyvl.geometry:

.. currentmodule:: pyvl


The :class:`Geometry`
=====================

The :class:`Geometry` denotes the smallest building block of geometry. This is intended
for representing a geometrical entity with a separate coordinate system.

.. autoclass:: Geometry
    :members:
    :member-order: bysource
    :exclude-members: save, load

Combining :class:`Geometry` into :class:`SimulationGeometry`
============================================================

The :class:`SimulationGeometry` object represents one or more individual :class:`Geometry` objects,
which are to be simulated together.

.. autoclass:: SimulationGeometry
    :members:
    :member-order: bysource
    :exclude-members: save, load, __getitem__
