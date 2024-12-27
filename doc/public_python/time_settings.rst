.. _pyvl.time_settings:

.. currentmodule:: pyvl

Time Settings
=============

The simulation will be run for each time step specified in the :class:`TimeSettings`.
These go from :math:`t = 0` to :math:`t = n \cdot \Delta t`. Optionally, the output
interval can be specified, so that the solver results are saved only every
:math:`n_\text{out}` steps instead of each and every step.

.. autoclass:: TimeSettings
    :members:
    :exclude-members: save, load
