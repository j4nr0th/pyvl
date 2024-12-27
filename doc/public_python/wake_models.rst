.. _pyvl.wake_models:

.. currentmodule:: pyvl

Wake Models
===========

Modeling of the wake can be done in several ways. There are several options
available, such as whether or not to include the wake self-induction, if the
induction of the geometry should be included at all, should the model be
steady-state or unsteady, etc.

The Base Type
-------------

.. autoclass:: WakeModel
    :members:
    :exclude-members: save, load


Unsteady Wakes
--------------

.. autoclass:: WakeModelLineExplicitUnsteady
    :members:
    :exclude-members: save, load
