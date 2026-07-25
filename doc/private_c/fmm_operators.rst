.. _pyvl.private_c.fmm_operators:

FMM Operators
=============

The FMM operators module provides local expansion types and conversion
operators for the Fast Multipole Method.  While the multipole expansion
is valid far from a source cluster, the *local expansion* is its dual:
it approximates the field *near* a target point from well-separated
sources.  The operators are defined in ``src/core/fmm_operators.h``.

The kernel is the same scalar :math:`1/r^2` induction as the multipole
expansion:

.. math::

    \vec{v}(\vec{r}) = \sum_i \frac{\vec{\Gamma}_i}{|\vec{r} - \vec{s}_i|^2}.

For a source cluster centred at :math:`\mathbf{S}` and a local expansion
centred at :math:`\mathbf{R}`, let

.. math::

    \mathbf{r}' &= \mathbf{r} - \mathbf{R} \\
    \mathbf{R}' &= \mathbf{R} - \mathbf{S}

Then the field near :math:`\mathbf{R}` from sources near :math:`\mathbf{S}`
can be expanded as:

.. math::

    \vec{v}(\vec{r}) \approx
    \frac{1}{|\mathbf{R}'|^2}\sum_{m=0}^{p}\frac{1}{|\mathbf{R}'|^{2m}}
    \sum_i \vec{\Gamma}_i\,
    \bigl(2\mathbf{R}'\cdot\mathbf{r}' + {\mathbf{r}'}^2\bigr)^m

The series converges when :math:`|\mathbf{r}'| < |\mathbf{R}'|`, which
holds for all FMM V-list pairs since :math:`|\mathbf{R}'| \ge 3h` and
:math:`|\mathbf{r}'| \le h`.

Operators
---------

Four operators connect the multipole and local representations:

- **M2L** (``multipole_to_local``): converts a multipole at
  :math:`\mathbf{S}` to a local expansion at :math:`\mathbf{R}`.
  The denominator factor :math:`2\mathbf{R}'\cdot\mathbf{r}' +
  {\mathbf{r}'}^2` is **quadratic** in :math:`\mathbf{r}'`, so this
  operator uses a quadratic polynomial multiplier (unlike the linear
  multiplier in ``multipole_add_shift``).

- **L2L** (``local_expansion_shift``): shifts a local expansion from one
  centre to another (parent to child during the downward sweep).
  The denominator factor :math:`2\mathbf{d}\cdot\mathbf{r}' + d^2` is
  **linear**, reusing the same polynomial multiplier as the multipole
  M2M shift.

- **L2P** (``local_expansion_eval``): evaluates a local expansion at
  a target point using the same Horner-style nested evaluation as
  ``multipole_eval``, but without the :math:`1/r^2` scaling (the
  :math:`1/|\mathbf{R}'|^{2(m+1)}` factors are baked into the
  coefficients during M2L).

- **P2L** (``particle_to_local``): builds a local expansion directly
  from a single source particle.  This is the single-particle limit
  of M2L and is useful for testing and for medium-sized near-field
  transfers.

All operators use the same tetrahedral coefficient layout and scratch
buffer conventions as the multipole library.

Data Structure
--------------

.. c:autodoc:: fmm_operators.h
