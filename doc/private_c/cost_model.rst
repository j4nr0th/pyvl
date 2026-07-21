.. _pyvl.private_c.cost_model:

Cost Model
==========

Static models for estimating the floating-point operation (FLOP) cost
of multipole evaluation versus direct summation. Used at tree-build
time to decide when a far-field multipole expansion is cheaper than
enumerating all sources.

The model counts every mul, add, sub, div, and sqrt occurring in the
respective kernels. See the individual function docs for breakdowns.

.. c:autodoc:: cost_model.h
