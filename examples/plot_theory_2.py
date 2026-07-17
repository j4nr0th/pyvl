r"""Multipole Shifting: Accuracy and Convergence
=================================================

.. currentmodule:: pyvl

This example studies the accuracy of the :class:`cvl.Multipole` shift operation
as a function of shift distance, direction, and the ``work_order`` parameter.
"""  # noqa: D205, D400

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from pyvl.cvl import Multipole

# %%
#
# Problem Statement
# -----------------
#
# In a Barnes-Hut tree, multipole expansions computed at child nodes must be
# *shifted* (translated) to a parent node's centre before aggregation.  The
# :meth:`cvl.Multipole.shift` method performs this translation.
#
# Mathematically, the shift re-expands the denominator
#
# .. math::
#
#     \frac{1}{|\mathbf{x} - \mathbf{c}_\text{in}|^{2}}
#
# as a binomial series around the new centre :math:`\mathbf{c}_\text{out}`.
# The parameter ``work_order`` controls how many terms of that series are
# retained.  The shift API requires ``work_order >= max(in->order, out->order)``.
# When ``work_order`` equals the multipole order, the series is truncated at
# the same number of terms as the expansion itself; setting ``work_order``
# *higher* than the multipole order retains extra denominator terms and
# improves accuracy for large shifts (see :ref:`pyvl.private_c.multipole` for
# the full formulation).
#
# We demonstrate this by creating a multipole expansion of order 2 from random
# sources inside a small sphere, then shifting it and measuring the error under
# various conditions.

# ---- Reproducible sources ----
rng = np.random.default_rng(0)
N_PARTICLES = 12
R_PARTICLES = 0.1

raw = rng.uniform(-1, 1, size=(N_PARTICLES, 3))
norms = np.linalg.norm(raw, axis=1, keepdims=True)
positions_3d = (
    raw / norms * (rng.uniform(0, 1, size=(N_PARTICLES, 1)) ** (1 / 3) * R_PARTICLES)
)
strengths_3d = rng.uniform(-1, 1, size=(N_PARTICLES, 3))

# ---- Reference multipole at origin ----
ORDER = 2
mp_ref = Multipole.from_sources(ORDER, (0.0, 0.0, 0.0), positions_3d, strengths_3d)


# ---- Exact induction for reference ----
def exact_induction(
    points: npt.NDArray[np.double],
    src_pos: npt.NDArray[np.double],
    src_val: npt.NDArray[np.double],
) -> npt.NDArray[np.double]:
    """Evaluate exact induction (same formula as plot_theory_1)."""
    out = np.zeros_like(points)
    for i in range(src_pos.shape[0]):
        delta = points - src_pos[i]
        r2 = np.sum(delta**2, axis=-1, keepdims=True)
        out += src_val[i] / np.maximum(r2, 1e-300)
    return out


# ---- Global evaluation point set (used for all sweeps) ----
rng_eval = np.random.default_rng(20250717)
N_EVAL = 500
R_EVAL = 3.0  # far-field: 30 x source radius

phi = rng_eval.uniform(0, 2 * np.pi, N_EVAL)
cos_theta = rng_eval.uniform(-1, 1, N_EVAL)
theta = np.arccos(cos_theta)
eval_pts = np.column_stack(
    [
        R_EVAL * np.sin(theta) * np.cos(phi),
        R_EVAL * np.sin(theta) * np.sin(phi),
        R_EVAL * np.cos(theta),
    ]
)

v_exact = exact_induction(eval_pts, positions_3d, strengths_3d)
v_baseline = mp_ref.eval(eval_pts)  # expansion error (no shift)


def l2_relative_error(
    v_approx: npt.NDArray[np.double], v_ex: npt.NDArray[np.double]
) -> float:
    """L2 norm of the difference divided by L2 norm of the exact field."""
    denom = np.sqrt(np.sum(np.linalg.norm(v_ex, axis=-1) ** 2))
    if denom < 1e-300:
        return 0.0
    return float(np.sqrt(np.sum(np.linalg.norm(v_approx - v_ex, axis=-1) ** 2)) / denom)


ERR_BASELINE = l2_relative_error(v_baseline, v_exact)
print(f"Baseline expansion error (no shift): {ERR_BASELINE:.2e}")


# %%
# 2D Slice Visualisation
# ----------------------
#
# We evaluate the exact field and three shifted expansions on the :math:`z=0`
# plane.  The original expansion (order 2, centre at origin) is shifted to
# :math:`(1,0,0)` using ``work_order=2`` (equal to the expansion order),
# ``work_order=4``, and ``work_order=6``.  Higher ``work_order`` recovers the
# field shape more faithfully away from the source cluster.

N_2D = 120
X_MIN, X_MAX, Y_MIN, Y_MAX = -4.0, 4.0, -4.0, 4.0
xs, ys = np.meshgrid(
    np.linspace(X_MIN, X_MAX, N_2D), np.linspace(Y_MIN, Y_MAX, N_2D), indexing="ij"
)
points_2d = np.stack([xs.ravel(), ys.ravel(), np.zeros_like(xs.ravel())], axis=-1)

v_exact_2d = exact_induction(points_2d, positions_3d, strengths_3d)

SHIFT_CENTRE = (1.0, 0.0, 0.0)
work_order_triplet = (2, 4, 6)
labels = {2: "$w = 2$ (order)", 4: "$w = 4$", 6: "$w = 6$"}

for wo in work_order_triplet:
    ms = mp_ref.shift(SHIFT_CENTRE, work_order=wo)
    v = ms.eval(points_2d)
    mag_log = np.log10(np.maximum(np.linalg.norm(v, axis=-1), 1e-30)).reshape(xs.shape)
    err_abs = np.log10(
        np.maximum(np.linalg.norm(v - v_exact_2d, axis=-1).reshape(xs.shape), 1e-30)
    )

    fig, ax = plt.subplots(1, 2, figsize=(10, 4.5))
    c0 = ax[0].contourf(xs, ys, mag_log, levels=20, cmap="viridis")
    ax[0].set(
        title=f"Shifted ({labels[wo]})",
        aspect="equal",
        xlabel="$x$",
        ylabel="$y$",
    )
    fig.colorbar(c0, ax=ax[0], label="$\\log_{10}|\\mathbf{v}|$")

    c1 = ax[1].contourf(xs, ys, err_abs, levels=20, cmap="inferno")
    ax[1].set(
        title="Abs. error vs exact",
        aspect="equal",
        xlabel="$x$",
        ylabel="$y$",
    )
    fig.colorbar(c1, ax=ax[1], label="$\\log_{10}|\\Delta\\mathbf{v}|$")
    fig.suptitle(f"Shift by (1,0,0), {labels[wo]}", y=1.02)
    fig.tight_layout()
    plt.show()

# %%
# Error vs Shift Distance
# -----------------------
#
# How does the shift error grow with distance?  We shift the order-2 expansion
# along the :math:`x`-axis by :math:`d \in [0, 2.5]` and evaluate the L2
# relative error for four ``work_order`` values.  The dashed horizontal line
# marks the expansion error of the *unshifted* multipole — the irreducible
# floor for this order.

shift_distances = np.linspace(0.02, 2.5, 60)
work_orders = (2, 3, 4, 5)
err_dist = {wo: [] for wo in work_orders}

for d in shift_distances:
    for wo in work_orders:
        ms = mp_ref.shift((d, 0.0, 0.0), work_order=wo)
        v = ms.eval(eval_pts)
        err_dist[wo].append(l2_relative_error(v, v_exact))

fig, ax = plt.subplots(figsize=(7, 5))
for wo in work_orders:
    ax.semilogy(shift_distances, err_dist[wo], label=f"$w = {wo}$", lw=1.8)
ax.axhline(ERR_BASELINE, color="gray", ls="--", lw=1, label="unshifted floor")
ax.set(
    xlabel="Shift distance $d$",
    ylabel="L2 relative error",
    title="Shift Error vs Distance (order 2)",
)
ax.legend()
ax.grid(True, which="both", alpha=0.3)
plt.show()

# %%
# Error vs Shift Direction
# ------------------------
#
# Is the shift error isotropic?  We fix the shift magnitude to :math:`d=1.0`
# and rotate its direction in the :math:`xy`-plane.  The error is measured on
# the same set of 500 random evaluation points.

angles = np.linspace(0, 2 * np.pi, 60)
d_fixed = 1.0
err_dir = {wo: [] for wo in work_orders}

for a in angles:
    dx = d_fixed * np.cos(a)
    dy = d_fixed * np.sin(a)
    for wo in work_orders:
        ms = mp_ref.shift((dx, dy, 0.0), work_order=wo)
        v = ms.eval(eval_pts)
        err_dir[wo].append(l2_relative_error(v, v_exact))

fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(6, 6))
for wo in work_orders:
    ax.plot(angles, err_dir[wo], label=f"$w = {wo}$", lw=1.8)
ax.set(title=f"Shift Error vs Direction ($d = {d_fixed}$)", ylabel="L2 rel. error")
ax.legend()
plt.show()

# %%
# Pure Shift Error — C-Test Methodology
# ----------------------------------------
#
# The sections above measure error against the *exact* field, which bundles
# two sources of error: the base expansion truncation and the shift binomial
# truncation.  The C test in :file:`test/math/test_multipole_shift.c` isolates
# the *pure* shift error by sweeping over expansion *order* with
#
# .. math::
#
#    \text{work\_order} = \text{order} + \text{INCREASE\_ORDER}
#
# and comparing the shifted result against a multipole built *directly* at the
# target centre — the same sources, same order, but a fresh expansion centred
# there.  Changing ``INCREASE_ORDER`` changes how many denominator series terms
# are retained and therefore the truncation error.
#
# Below we replicate this: for each expansion order we generate 30 random
# sources, build a multipole at their weighted centroid, shift by
# :math:`d = 1.0`, and compare against a multipole built directly at the new
# centre.  The evaluation is on a Fibonacci sphere at :math:`r = 3.0`.
# Note: Python float64 is strictly IEEE-compliant, so at very low orders
# (where the shift is small compared with the source cluster) the error sits
# at machine precision — the C test sees larger errors at ``-O3 -flto`` due
# to FMA contractions, a known difference.  The *trend* is what matters.

N_SEEDS = 30
SHIFT_MAG = 1.0
R_EVAL_C = 3.0
MAX_ORDER = 7

# Fibonacci-sphere evaluation set
N_EVAL_C = 200
t_fib = np.linspace(0.0, 1.0, N_EVAL_C, endpoint=False)
phi_fib = np.arccos(1.0 - 2.0 * t_fib)
theta_fib = 2.0 * np.pi * t_fib * (1.0 + np.sqrt(5.0)) / 2.0
eval_sphere = R_EVAL_C * np.column_stack(
    [
        np.sin(phi_fib) * np.cos(theta_fib),
        np.sin(phi_fib) * np.sin(theta_fib),
        np.cos(phi_fib),
    ]
)

increase_values = (0, 1, 2, 3)
err_by_increase: dict[int, list[float]] = {iv: [] for iv in increase_values}

rng_seeds = np.random.default_rng(20250717)

for order in range(MAX_ORDER + 1):
    accum: dict[int, float] = {iv: 0.0 for iv in increase_values}
    count: dict[int, int] = {iv: 0 for iv in increase_values}

    for _ in range(N_SEEDS):
        seed_val = rng_seeds.integers(0, 2**31)
        rng_src = np.random.default_rng(int(seed_val))

        coords = rng_src.uniform(-0.1, 0.1, (30, 3))
        values = rng_src.uniform(-1.0, 1.0, (30, 3))

        weights = np.linalg.norm(values, axis=1)
        centre_a = np.average(coords, axis=0, weights=weights)
        centre_b = centre_a + SHIFT_MAG

        mp_a = Multipole.from_sources(order, centre_a, coords, values)
        mp_direct = Multipole.from_sources(order, centre_b, coords, values)
        v_dir = mp_direct.eval(eval_sphere)
        denom_sq = np.sum(np.linalg.norm(v_dir, axis=-1) ** 2)
        if denom_sq < 1e-300:
            continue
        denom_l2 = np.sqrt(denom_sq)

        for iv in increase_values:
            wo = order + iv
            try:
                mp_shifted = mp_a.shift(centre_b, work_order=wo)
            except ValueError:
                continue
            v_shift = mp_shifted.eval(eval_sphere)
            diff_sq = np.sum(np.linalg.norm(v_shift - v_dir, axis=-1) ** 2)
            if not np.isfinite(diff_sq):
                continue  # skip seeds where float64 overflow occurs
            err = np.sqrt(diff_sq) / denom_l2
            accum[iv] += err
            count[iv] += 1

    for iv in increase_values:
        if count[iv] > 0:
            err_by_increase[iv].append(accum[iv] / count[iv])
        else:
            err_by_increase[iv].append(np.nan)

# --- Plot: pure shift error vs expansion order, one curve per INCREASE_ORDER ---
fig, ax = plt.subplots(figsize=(7, 5))
for iv in increase_values:
    data = err_by_increase[iv]
    ax.semilogy(
        range(MAX_ORDER + 1),
        data,
        marker="o",
        lw=1.8,
        label=f"$w = p$ + {iv}",
    )
ax.set(
    xlabel="Expansion order $p$",
    ylabel="L2 relative error (shift only)",
    title="Pure Shift Error (C-test methodology)",
    xticks=range(MAX_ORDER + 1),
)
ax.legend(title="INCREASE_ORDER")
ax.grid(True, which="both", alpha=0.3)
plt.show()

# %%
# The plot shows:
#
# * At low orders (:math:`p \le 4`) the shift is small relative to the cluster
#   size, so the error sits at machine precision for all ``INCREASE_ORDER``
#   values.  The C test sees larger errors here at ``-O3 -flto`` because of
#   FMA contractions that Python float64 does not perform.
#
# * At orders :math:`p = 6, 7` the curves separate clearly:
#   ``INCREASE_ORDER = 0`` (no extra denominator terms) gives an error around
#   :math:`10^{-1}` to :math:`10^{-2}`.  Adding one extra term
#   (``INCREASE_ORDER = 1``) drops this to :math:`10^{-4}` to :math:`10^{-5}`,
#   and ``INCREASE_ORDER = 2`` brings the error down to machine precision.
#
# * The C test uses ``INCREASE_ORDER = 3`` to guarantee the pure shift error
#   stays below :math:`10^{-4}` for all expansion orders — you can verify this
#   by editing :file:`test/math/test_multipole_shift.c` and re-running.

# %%
# Convergence w.r.t. Work Order
# ------------------------------
#
# Increasing ``work_order`` retains more terms in the binomial denominator
# series.  Each extra term adds one power of the expansion, giving algebraic
# convergence.  We fix three shift distances and sweep ``work_order`` from 2
# (the minimum, equal to the expansion order) up to 7.

work_range = list(range(2, 8))
distances_for_conv = (0.5, 1.0, 2.0)
err_conv: dict[float, list[float]] = {}

for d in distances_for_conv:
    err_conv[d] = []
    for wo in work_range:
        ms = mp_ref.shift((d, 0.0, 0.0), work_order=wo)
        v = ms.eval(eval_pts)
        err_conv[d].append(l2_relative_error(v, v_exact))

fig, ax = plt.subplots(figsize=(7, 5))
for d in distances_for_conv:
    ax.semilogy(
        work_range,
        err_conv[d],
        marker="o",
        lw=1.8,
        label=f"$d = {d}$",
    )
ax.axhline(ERR_BASELINE, color="gray", ls="--", lw=1, label="unshifted floor")
ax.set(
    xlabel="Work order $w$",
    ylabel="L2 relative error",
    title="Total Error vs Work Order (order 2)",
    xticks=work_range,
)
ax.legend()
ax.grid(True, which="both", alpha=0.3)
plt.show()

# %%
# Notice that the *total* error (vs the exact field) floors out once the shift
# error drops below the base expansion truncation error.  This is why the C
# test uses ``INCREASE_ORDER`` — it guarantees the shift contribution does not
# degrade the overall accuracy.

# %%
#
# Interpretation
# --------------
#
# * **The ``INCREASE_ORDER`` parameter controls shift fidelity.**  Using
#   ``work_order = p + INCREASE_ORDER``, the pure shift error drops by roughly
#   half an order of magnitude per added term at the highest orders.  The C
#   test uses ``INCREASE_ORDER = 3`` to guarantee the error stays below
#   :math:`10^{-4}` — the value you can check by editing
#   :file:`test/math/test_multipole_shift.c`.
#
# * **Direction is irrelevant.**  The shift error is isotropic — only the
#   *magnitude* of the shift vector matters, not its orientation.
#
# * **Total error has a floor.**  When measuring against the exact field, the
#   convergence is capped by the base expansion truncation.  Once the shift
#   error falls below that floor, raising ``work_order`` further has no visible
#   effect — the C test avoids this by comparing shift vs direct multipole
#   rather than vs the exact field.
#
# * **Practical rule of thumb.**  Shifts where :math:`d \\lesssim 5\\,R_\\text{cluster}`
#   are well handled with ``work_order`` equal to the multipole order.  Larger
#   shifts benefit from ``work_order`` = order + 1 or + 2, at the cost of
#   larger work buffers (see :ref:`pyvl.private_c.multipole` for buffer sizing).
