.. _pyvl.private_c.approx_atan:

Approximate Trigonometric Functions
====================================

Fast polynomial approximations for ``atan`` and ``atan2``, used in the
vortex filament induction kernel where the standard library's
``atan2`` would be a bottleneck.

The approximations use a lookup table of polynomial coefficients for
different intervals of the input. The maximum error is below
:math:`6 \times 10^{-9}` when compared to NumPy's implementation.

.. c:autodoc:: approx_atan.h
