.. _pyvl.reference_frame:

.. currentmodule:: pyvl

Reference Frames
================

This page details the implementations of the :class:`ReferenceFrame` type. This type
is used to describe how different coordinate systems are related to each other. Their
relative position, velocity, orientation, and rate of rotation can be either fixed or
time-dependant.

The :class:`ReferenceFrame` itself is immutable. The four properties it has (position,
velocity, orientation, and rate of rotation) are used to determine the relative position
and velocity with respect to any other :class:`ReferenceFrame`. The hierarchical relationship
of different :class:`ReferenceFrame` is based on the ``parent`` they have.


.. autoclass:: ReferenceFrame
    :members:
