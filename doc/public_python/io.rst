.. _pyvl.io:

.. currentmodule:: pyvl

File I/O and Serialization
===========================

The :class:`HirearchicalMap` class is the fundamental data structure used throughout
pyvl for serialization and deserialization. It provides a hierarchical key-value store
that can contain nested maps, scalars, strings, and arrays.

The :class:`HirearchicalMap` is used internally to serialize solver state, geometry,
reference frames, and wake models. It can be written to disk in either JSON or HDF5
format for long-term storage or cross-platform compatibility.


The :class:`HirearchicalMap` Class
-----------------------------------

.. autoclass:: HirearchicalMap
    :members:
