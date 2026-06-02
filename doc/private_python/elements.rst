.. _pyvl.private.elements:

.. currentmodule:: pyvl.elements

Element State Module
====================

The elements module provides the :class:`ImplicitElements` class, which represents the
state of the solver (positions, normals, control points, circulations, etc.) at a specific
point in time. This class is used internally for serialization and caching of intermediate
results.


The :class:`ImplicitElements` Class
-----------------------------------

.. autoclass:: ImplicitElements
    :members:
