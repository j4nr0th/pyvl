.. _pyvl.reference_frame:

.. currentmodule:: pyvl

Reference Frames
================

This page details the implementations of the :class:`ReferenceFrame` type, as well
as some types derived from it. These can be further sub-classed in order to define
different motion behaviour.

Note, that in order for a custom defined reference frame to used in restarting/reloading
the solver, it is necessary for it to override the :meth:`ReferenceFrame.save` and
:meth:`ReferenceFrame.load` methods. If not, it will be loaded as the basic
:class:`ReferenceFrame`, which is sufficient for post-processing, but not for
resuming a simulation.


The Basic :class:`ReferenceFrame`
---------------------------------

This is the base type of all other types. The entire implementation of the type
is written in C and the type itself is immutable. It describes a coordinate system
in terms of an offset and three rotations relative to its parent.

.. autoclass:: ReferenceFrame
    :members:
    :exclude-members: save, load

The Uniformly Moving :class:`TranslatingReferenceFrame`
-------------------------------------------------------

The :class:`TranslatingReferenceFrame` is a sub-type of the :class:`ReferenceFrame`
and is meant for describing a coordinate system which translates with an uniform velocity.

.. autoclass:: TranslatingReferenceFrame
    :members:
    :exclude-members: save, load

The Uniformly Rotating :class:`RotorReferenceFrame`
---------------------------------------------------

This sub-type of the :class:`ReferenceFrame` describes a coordinate system, which is
rotating with a constant angular velocity around its origin. This is intended to allow
for simulation of rotors or any other rotating objects.

.. autoclass:: RotorReferenceFrame
    :members:
    :exclude-members: save, load
