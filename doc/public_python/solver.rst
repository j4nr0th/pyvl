.. _pyvl.solver:

.. currentmodule:: pyvl

The Solver
==========

The solver computes circulation distributions that satisfy the no-penetration boundary
condition on the mesh surfaces. This circulation can then be used to compute induced
velocities, pressures, and forces.


Running the Flow Solver
-----------------------

.. autofunction:: run_solver


Induced Velocity Computation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :func:`compute_induced_velocities` function allows you to compute the velocity field
induced by the mesh circulation at arbitrary positions. This is useful for analyzing the
flow field in regions away from the geometry surface.

.. autofunction:: compute_induced_velocities


Output Configuration
--------------------

.. autoclass:: OutputSettings
    :members:
    :no-index: