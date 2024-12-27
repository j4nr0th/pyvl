.. _pyvl.flow_conditions:

.. currentmodule:: pyvl

Flow Conditions
===============

The free-stream conditions of the fluid flow are described by an object of
any type, which inherits from the base type :class:`FlowConditions`.

The Base Type
-------------

.. autoclass:: FlowConditions
    :members:
    :exclude-members: save, load


Uniform Flow
------------

The most common case is the case of the uniform free-stream flow. To this
end, the :class:`FlowConditionsUniform` object can be used. It returns
a constant free-stream velocity for any position and any time.

.. autoclass:: FlowConditionsUniform
    :members:
    :exclude-members: save, load



Rotating Flow
-------------


.. autoclass:: FlowConditionsRotating
    :members:
    :exclude-members: save, load
