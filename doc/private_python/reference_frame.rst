.. _pyvl.private.reference_frame:

.. currentmodule:: pyvl

Reference Frame Module
======================

The reference frame module provides the :class:`ReferenceFrame` class, which
is used to define position and orientation of geometry.

Serialization
-------------

The :class:`ReferenceFrame` can be serialized and deserialized using the :meth:`ReferenceFrame.save`
and :meth:`ReferenceFrame.load` methods, which use a :class:`HierarchicalMap`
to store the position, velocity, orientation, and rotation properties.
