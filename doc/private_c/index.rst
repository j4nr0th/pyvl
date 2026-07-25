.. _pyvl.private_c:

Private C API
=============

This part of the documentation covers internals of the code written
in C. This overlaps somewhat with the documentation in for the
:ref:`private Python code <pyvl.private_python>`, but covers it from the
C side.

The C implementation is organized into logical groups based on functionality.


Fundamental Data Types
----------------------

Core types and mathematical utilities used throughout the implementation.

.. toctree::
    :maxdepth: 1

    fundamental_types


Mesh Operations
---------------

Data structures and operations for geometric meshes.

.. toctree::
    :maxdepth: 1

    mesh


Velocity Induction
------------------

Algorithms for computing vortex-induced velocities.

.. toctree::
    :maxdepth: 1

    induction
    multipole
    fmm_operators
    fmm_tree
    barnes_hut_tree
    approx_atan
    cost_model


Coordinate Transforms
---------------------


Utilities for coordinate transformation composition and inversion.

.. toctree::
    :maxdepth: 1

    transforms


Mesh Serialization
------------------

Functions for serializing and deserializing mesh data.

.. toctree::
    :maxdepth: 1

    serialization
