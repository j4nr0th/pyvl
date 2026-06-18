.. _pyvl.postprocessing:

.. currentmodule:: pyvl.postprocess

Post-Processing
===============

After running the solver, the circulation distribution is typically not the final
quantity of interest. The post-processing module provides functions to compute
velocity fields, pressure distributions, and forces resulting from the circulation.

All post-processing functions operate on :class:`pyvl.SolverResults` objects returned
by the solver.


Velocity Field Computation
--------------------------

The velocity at arbitrary positions can be computed by superimposing the contributions
from all vortex elements. The induced velocity is added to the freestream flow condition
and, if present, the wake model contribution.

.. autofunction:: compute_velocities

For cases where different positions are needed at each time step, use the variable variant:

.. autofunction:: compute_velocities_variable


Pressure Distribution
---------------------

The dynamic pressure on the mesh surface or at arbitrary points can be computed using
Bernoulli's equation:

.. math::

    q = -\frac{1}{2} \rho |v|^2

where :math:`\rho` is the fluid density and :math:`v` is the total velocity (freestream
plus induced).

.. autofunction:: compute_surface_dynamic_pressure

For pressure computation at arbitrary positions:

.. autofunction:: compute_dynamic_pressure_variable


Force Computation
-----------------

The forces on each mesh edge can be computed from the circulation distribution using
the Kutta-Joukowski theorem. This yields the circulatory force on each vortex line.

What is important to note is that according to potential flow, a closed horseshoe vortex
will not produce any force. As such, the only places where a non-zero force will be observed
according to this model of the flow is where the wake is being shed. This is because those
vortices are not closed on the surface, but extend into the flow.

.. autofunction:: circulatory_forces
