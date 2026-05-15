.. _pyvl.private.fio:

.. currentmodule:: pyvl.fio.io_hdf5

File I/O Module
===============

The file I/O module provides functions to serialize and deserialize :class:`pyvl.HirearchicalMap`
objects to and from various file formats. This is used internally for saving solver state,
geometry, and simulation results.


HDF5 Format
-----------

The HDF5 format provides efficient storage for large numerical arrays and hierarchical
data structures. It is particularly suitable for large simulations with many time steps.

.. autofunction:: serialize_hdf5

.. autofunction:: deserialize_hdf5


JSON Format
-----------

The JSON format provides human-readable text-based storage. It is useful for debugging,
small simulations, or when interoperability with other tools is required.

.. currentmodule:: pyvl.fio.io_json

.. autofunction:: serialize_json

.. autofunction:: deserialize_json