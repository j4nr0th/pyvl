.. _pyvl.private_python:

Private Python API
==================

This part of the documentation covers internals of the code and
is intended for those who wish to change or enhance the internal
Python code.


File I/O Implementation
-----------------------

The file I/O module provides functions for serializing and deserializing
HirearchicalMap objects to various file formats.

.. toctree::
    :maxdepth: 1

    fio


Element State
-------------

The elements module provides the ImplicitElements class for representing
solver state at a specific moment.

.. toctree::
    :maxdepth: 1

    elements