.. _pyvl.reference_frame:

.. currentmodule:: pyvl

Reference Frames
================

This page details the implementations of the :class:`ReferenceFrame` type. This type
is used to describe how different coordinate systems are related to each other. Their
relative position, velocity, orientation, and rate of rotation can be either fixed or
time-dependant.

The :class:`ReferenceFrame` itself is immutable and cannot be subclassed.
The four properties it has (position, velocity, orientation, and rate of rotation)
are used to determine the relative position and velocity with respect to
any other :class:`ReferenceFrame`. The hierarchical relationship
of different :class:`ReferenceFrame` is based on the ``parent`` they have.

This type has many methods which allows for transforming quantities from one reference
frame to another. These quantities are:

- position vectors (influenced by orientation and offset),
- velocity vectors (influenced by orientation, offset, rotation rate, and translation),
- other vectors (influenced by orientation only).

These can be transformed between any two reference frames using the
:meth:`ReferenceFrame.transform_position`, :meth:`ReferenceFrame.transform_velocity`, and
:meth:`ReferenceFrame.transform_vector`. There are also more convenient versions of these
that transform to or from either the parent or the global reference frame, as those are the
most common use cases.

.. autoclass:: ReferenceFrame
    :members:
