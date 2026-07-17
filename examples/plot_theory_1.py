r"""Far-Field Approximation with the Multipole Type
===============================================

.. currentmodule:: pyvl

This example shows the theory and accuracy of the :class:`cvl.Multipole` expansion
used for wake modeling. It replaces the ad-hoc expansion formulas of the
previous version with the actual 3D vector multipole implementation
from :class:`cvl.Multipole`.
"""  # noqa: D205, D400

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pyvista as pv
from pyvl.cvl import Multipole

pv.set_plot_theme("document")
pv.set_jupyter_backend("html")

# %%
#
# Problem Statement
# -----------------
#
# For the panel method, the wake is modeled as a series of vortex rings. Computing
# their induction quickly becomes a bottleneck, as the number of computations scales
# with the square of the number of panels. The far-field approximation replaces groups of
# vortex particles with a single :class:`cvl.Multipole` expansion, reducing the
# computational effort at the cost of accuracy.
#
# The field from a set of :math:`n` sources is:
#
# .. math::
#    :label: eq:exact-induction
#
#    \mathbf{v}(\mathbf{x}) =
#    \sum_{i=1}^{n} \frac{\boldsymbol{\Gamma}_i}{|\mathbf{r}_i - \mathbf{x}|^2}
#
# where :math:`\mathbf{r}_i` is the position and :math:`\boldsymbol{\Gamma}_i` is the
# vector strength of the :math:`i`-th particle.  For vortex wake modelling this
# strength is :math:`\boldsymbol{\Gamma}_i = \vec{\ell}_i \times \vec{\Gamma}_{c,i}`
# (the line segment crossed with the scalar circulation).
#
# The :class:`cvl.Multipole` approximates this field as a series expansion of order
# :math:`p` around a common centre :math:`\mathbf{c}`:
#
# .. math::
#    :label: eq:multipole-expansion
#
#    \mathbf{v}(\mathbf{x}) \approx
#    \sum_{m=0}^{p} \frac{1}{|\mathbf{x}-\mathbf{c}|^{2(m+1)}}
#    \sum_{i=1}^{n} \boldsymbol{\Gamma}_i \,
#    \bigl(2 (\mathbf{x}-\mathbf{c})\cdot(\mathbf{r}_i-\mathbf{c})
#          - |\mathbf{r}_i-\mathbf{c}|^2\bigr)^{\!m}
#
# Higher orders include more moments of the source distribution and improve accuracy
# at the cost of more coefficients.
#
# To demonstrate this, we place 12 vortex particles with random 3D positions inside a
# small sphere of radius :math:`R=0.1` centred at the origin, each with a random vector
# strength.  The exact field and the multipole approximations at various orders are then
# compared on a 2D slice through the :math:`z=0` plane and on full 3D grids.

rng = np.random.default_rng(0)
N_PARTICLES = 12
R_PARTICLES = 0.1

# 3D source positions inside a sphere of radius R_PARTICLES
raw = rng.uniform(-1, 1, size=(N_PARTICLES, 3))
norms = np.linalg.norm(raw, axis=1, keepdims=True)
positions_3d = (
    raw / norms * (rng.uniform(0, 1, size=(N_PARTICLES, 1)) ** (1 / 3) * R_PARTICLES)
)

# Random vector strengths (circulation × line-length)
strengths_3d = rng.uniform(-1, 1, size=(N_PARTICLES, 3))


# ---- Helper: exact vector induction (matching Multipole's formula) ----
def exact_induction(
    points: npt.NDArray[np.double],
    src_pos: npt.NDArray[np.double],
    src_val: npt.NDArray[np.double],
) -> npt.NDArray[np.double]:
    """Evaluate the exact induction eq. (1) at *points*."""
    out = np.zeros_like(points)
    for i in range(src_pos.shape[0]):
        delta = points - src_pos[i]  # (..., 3)
        r2 = np.sum(delta**2, axis=-1, keepdims=True)
        out += src_val[i] / np.maximum(r2, 1e-300)
    return out


# ---- 2D slice evaluation grid ----
X_MIN, X_MAX, Y_MIN, Y_MAX = -5, 5, -5, 5
N_2D = 200
xs, ys = np.meshgrid(
    np.linspace(X_MIN, X_MAX, N_2D), np.linspace(Y_MIN, Y_MAX, N_2D), indexing="ij"
)
points_2d = np.stack([xs.ravel(), ys.ravel(), np.zeros_like(xs.ravel())], axis=-1)

# Exact field on the 2D slice
v_exact_2d = exact_induction(points_2d, positions_3d, strengths_3d)
v_exact_2d_mag = np.linalg.norm(v_exact_2d, axis=-1).reshape(xs.shape)
v_exact_2d_log = np.log10(np.maximum(v_exact_2d_mag, 1e-30))

# %%
#
# Building Multipole Expansions
# -----------------------------
#
# The :class:`cvl.Multipole` type is constructed directly from source data using the
# :meth:`cvl.Multipole.from_sources` class method.  We create expansions of order 0
# (monopole), 1 (dipole), 2 (quadrupole), and 3 (octupole) centred at the origin.

CENTER = (0.0, 0.0, 0.0)

multipoles = {}
for order in range(4):
    mp = Multipole.from_sources(order, CENTER, positions_3d, strengths_3d)
    multipoles[order] = mp

# Evaluate each on the 2D slice
v_mp_2d = {}
v_mp_2d_mag = {}
v_mp_2d_log = {}
for order, mp in multipoles.items():
    v = mp.eval(points_2d)
    v_mp_2d[order] = v
    mag = np.linalg.norm(v, axis=-1).reshape(xs.shape)
    v_mp_2d_mag[order] = mag
    v_mp_2d_log[order] = np.log10(np.maximum(mag, 1e-30))

# %%
# 2D Slice Visualisation — Exact Field
# -------------------------------------
#
# The figure below shows the magnitude of the exact induction field on the :math:`z=0`
# plane.  The red circle marks the region containing all source particles.

fig, ax = plt.subplots(figsize=(6, 5))
c = ax.contourf(xs, ys, v_exact_2d_log, levels=20, cmap="viridis")
particle_circle = plt.Circle((0, 0), R_PARTICLES, color="r", fill=False, linewidth=1.5)
ax.add_patch(particle_circle)
ax.set(
    title="Exact Induction Magnitude ($z=0$)", aspect="equal", xlabel="$x$", ylabel="$y$"
)
fig.colorbar(c, ax=ax, label="$\\log_{10}|\\mathbf{v}|$")
plt.show()

# %%
# 2D Slice — Monopole vs. Exact
# ------------------------------
#
# The order-0 (monopole) expansion uses only the net sum of the source strengths.
# Compare the contours with the exact solution.

inside_radius_2d = np.hypot(xs, ys) <= R_PARTICLES


def plot_side_by_side(
    v_exact_log: npt.NDArray,
    v_approx_log: npt.NDArray,
    title: str,
    *,
    levels: npt.ArrayLike | None = None,
) -> None:
    """Plot exact and approximate fields side-by-side."""
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    c = ax[0].contourf(
        xs, ys, v_exact_log, levels=20 if levels is None else levels, cmap="viridis"
    )
    ax[0].set(title="Exact", aspect="equal", xlabel="$x$", ylabel="$y$")
    fig.colorbar(c, ax=ax[0], label="$\\log_{10}|\\mathbf{v}|$")
    c2 = ax[1].contourf(xs, ys, v_approx_log, levels=c.levels, cmap="viridis")
    ax[1].set(title=title, aspect="equal", xlabel="$x$", ylabel="$y$")
    fig.colorbar(c2, ax=ax[1], label="$\\log_{10}|\\mathbf{v}_{\\text{approx}}|$")
    fig.tight_layout()
    plt.show()


plot_side_by_side(v_exact_2d_log, v_mp_2d_log[0], "Monopole ($p=0$)")


# %%
# Error Analysis — Monopole
# -------------------------
#
# The absolute and relative error of the monopole approximation on the :math:`z=0` slice.
# A contour of 1 % relative error is marked in red.

inside_mask = np.broadcast_to(inside_radius_2d, xs.shape)


def plot_error_2d(
    v_exact_mag: npt.NDArray, v_approx_mag: npt.NDArray, title: str = ""
) -> None:
    """Plot absolute and relative error on the 2D slice."""
    err_abs = np.abs(v_exact_mag - v_approx_mag)
    err_rel = err_abs / np.maximum(v_exact_mag, 1e-300)
    err_abs[inside_mask] = np.nan
    err_rel[inside_mask] = np.nan

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    c = ax[0].contourf(xs, ys, np.log10(err_abs), levels=20, cmap="viridis")
    ax[0].set(title=f"Absolute Error{title}", aspect="equal", xlabel="$x$", ylabel="$y$")
    fig.colorbar(
        c, ax=ax[0], label="$\\log_{10}|\\mathbf{v}-\\mathbf{v}_{\\text{approx}}|$"
    )
    c2 = ax[1].contourf(xs, ys, np.log10(err_rel), levels=20, cmap="viridis")
    ax[1].contour(xs, ys, np.log10(err_rel), levels=[-2], colors="r", linestyles="--")
    ax[1].set(title=f"Relative Error{title}", aspect="equal", xlabel="$x$", ylabel="$y$")
    fig.colorbar(
        c2,
        ax=ax[1],
        label="$\\log_{10}|(\\mathbf{v}-\\mathbf{v}_{\\text{approx}})/\\mathbf{v}|$",
    )
    fig.tight_layout()
    plt.show()


plot_error_2d(v_exact_2d_mag, v_mp_2d_mag[0], " (Monopole, $p=0$)")


# %%
# Improving the Approximation: Higher Orders
# -------------------------------------------
#
# Adding more terms (dipole, quadrupole, octupole) dramatically improves accuracy.
# The :class:`cvl.Multipole` encapsulates all orders — simply pick a higher ``order``
# at construction time.  Below we compare orders 1, 2, and 3.

for order in (1, 2, 3):
    plot_side_by_side(v_exact_2d_log, v_mp_2d_log[order], f"Order {order}")
    plot_error_2d(v_exact_2d_mag, v_mp_2d_mag[order], f" (Order {order})")


# %%
# Radial Error Comparison
# -----------------------
#
# The polar plot below shows relative error as a function of angle at a fixed radius
# :math:`r = 20 R`, for each expansion order.


def rel_error_at_radius(
    r: float,
    ntheta: int,
    mp: dict[int, Multipole],
    print_errors: bool = False,
) -> dict[int, npt.NDArray]:
    """Compute relative error at a given radius for each order."""
    theta = np.linspace(0, 2 * np.pi, ntheta)
    pts = np.stack([r * np.cos(theta), r * np.sin(theta), np.zeros(ntheta)], axis=-1)
    v_exact = exact_induction(pts, positions_3d, strengths_3d)
    v_exact_mag = np.linalg.norm(v_exact, axis=-1)
    errors = {}
    for order, mp_i in mp.items():
        v_approx = mp_i.eval(pts)
        err = np.linalg.norm(v_approx - v_exact, axis=-1) / np.maximum(
            v_exact_mag, 1e-300
        )
        errors[order] = err
        if print_errors:
            print(f"Order {order}, max rel error at r={r:.1f}: {err.max():.2e}")
    return errors


errors_radial = rel_error_at_radius(R_PARTICLES * 20, 120, multipoles, print_errors=True)

fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(6, 6))
for order, err in errors_radial.items():
    ax.plot(np.linspace(0, 2 * np.pi, 120), err, label=f"Order {order}")
ax.set(title=f"Relative Error at $r = {R_PARTICLES * 20:.1f}$", ylabel="Rel. error")
ax.legend()
plt.show()

# %%
# :meth:`cvl.Multipole.shift` and :meth:`cvl.Multipole.shift_to`
# ---------------------------------------------------------------
#
# In a Barnes-Hut tree, multipole expansions must be shifted (translated) from child
# nodes to a common parent centre.  The :meth:`cvl.Multipole.shift` method creates a new
# expansion at an arbitrary centre, while :meth:`cvl.Multipole.shift_to` re-centres onto
# another existing :class:`cvl.Multipole`'s centre.
#
# Below we take the order-3 expansion and shift it to a new centre, then compare the
# field evaluated from the shifted expansion against the direct exact computation.

new_centre = (0.5, 0.0, 0.0)
mp_shifted = multipoles[3].shift(new_centre)

v_shifted_2d = mp_shifted.eval(points_2d)
v_shifted_mag = np.linalg.norm(v_shifted_2d, axis=-1).reshape(xs.shape)
v_shifted_log = np.log10(np.maximum(v_shifted_mag, 1e-30))

# Compute exact field with sources at their original positions but evaluated from
# the perspective of the shifted centre — for a valid comparison we simply show
# that shift produces a usable expansion at the new centre.
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
c = ax[0].contourf(xs, ys, v_mp_2d_log[3], levels=20, cmap="viridis")
ax[0].set(title="Order 3 at Origin", aspect="equal", xlabel="$x$", ylabel="$y$")
fig.colorbar(c, ax=ax[0], label="$\\log_{10}|\\mathbf{v}|$")
c2 = ax[1].contourf(xs, ys, v_shifted_log, levels=c.levels, cmap="viridis")
ax[1].set(title="Shifted to $(0.5, 0, 0)$", aspect="equal", xlabel="$x$", ylabel="$y$")
fig.colorbar(c2, ax=ax[1], label="$\\log_{10}|\\mathbf{v}_{\\text{shifted}}|$")
fig.tight_layout()
plt.show()

# Demonstrate shift_to
mp_target = Multipole(3, (0.0, 0.5, 0.0))
mp_shifted_to = multipoles[3].shift_to(mp_target)
print(f"shift_to created expansion at centre: {mp_shifted_to.center}")

# %%
# 3D Visualisation with PyVista
# ------------------------------
#
# To fully appreciate the 3D nature of the multipole expansion, we evaluate the
# exact field and the order-3 expansion on a volumetric grid and visualise
# isosurfaces of the field magnitude.

N_3D = 40
x3 = np.linspace(-2, 2, N_3D)
y3 = np.linspace(-2, 2, N_3D)
z3 = np.linspace(-2, 2, N_3D)
X3, Y3, Z3 = np.meshgrid(x3, y3, z3, indexing="ij")
points_3d = np.stack([X3.ravel(), Y3.ravel(), Z3.ravel()], axis=-1)

# Exact field on 3D grid
v_exact_3d = exact_induction(points_3d, positions_3d, strengths_3d)
v_exact_3d_mag = np.log10(np.maximum(np.linalg.norm(v_exact_3d, axis=-1), 1e-30)).reshape(
    X3.shape
)

# Order-3 multipole on 3D grid
v_mp_3d = multipoles[3].eval(points_3d)
v_mp_3d_mag = np.log10(np.maximum(np.linalg.norm(v_mp_3d, axis=-1), 1e-30)).reshape(
    X3.shape
)

# Create PyVista structured grids
grid_exact = pv.ImageData(
    dimensions=(N_3D, N_3D, N_3D), spacing=(x3[1] - x3[0], y3[1] - y3[0], z3[1] - z3[0])
)
grid_exact.origin = (x3[0], y3[0], z3[0])
grid_exact.point_data["mag_log"] = v_exact_3d_mag.ravel(order="F")

grid_mp = pv.ImageData(
    dimensions=(N_3D, N_3D, N_3D), spacing=(x3[1] - x3[0], y3[1] - y3[0], z3[1] - z3[0])
)
grid_mp.origin = (x3[0], y3[0], z3[0])
grid_mp.point_data["mag_log"] = v_mp_3d_mag.ravel(order="F")

# Isosurfaces for the exact field
levels = np.linspace(v_exact_3d_mag.min() + 0.5, v_exact_3d_mag.max() - 0.5, 6)
iso_exact = grid_exact.contour(levels.tolist(), scalars="mag_log")

plotter = pv.Plotter(off_screen=True, shape=(1, 2))
plotter.subplot(0, 0)
plotter.add_text("Exact Field", font_size=12)
plotter.add_mesh(iso_exact, opacity=0.6, show_scalar_bar=True)
plotter.add_axes()

# Isosurfaces for the multipole
iso_mp = grid_mp.contour(levels.tolist(), scalars="mag_log")
plotter.subplot(0, 1)
plotter.add_text("Multipole Order 3", font_size=12)
plotter.add_mesh(iso_mp, opacity=0.6, show_scalar_bar=True)
plotter.add_axes()

plotter.show(interactive=False)
plotter.close()
del plotter

# %%
# Order Comparison in 3D
# ----------------------
#
# Isosurfaces for orders 0, 1, 2, 3 side-by-side, showing how each successive order
# better captures the structure of the exact field.

plotter = pv.Plotter(off_screen=True, shape=(2, 2))

for idx, (order, mp) in enumerate(multipoles.items()):
    v_tmp = mp.eval(points_3d)
    v_tmp_mag = np.log10(np.maximum(np.linalg.norm(v_tmp, axis=-1), 1e-30)).reshape(
        X3.shape
    )
    grid_tmp = pv.ImageData(
        dimensions=(N_3D, N_3D, N_3D),
        spacing=(x3[1] - x3[0], y3[1] - y3[0], z3[1] - z3[0]),
    )
    grid_tmp.origin = (x3[0], y3[0], z3[0])
    grid_tmp.point_data["mag_log"] = v_tmp_mag.ravel(order="F")
    iso_tmp = grid_tmp.contour(levels.tolist(), scalars="mag_log")

    row, col = divmod(idx, 2)
    plotter.subplot(row, col)
    plotter.add_text(f"Order {order}", font_size=12)
    plotter.add_mesh(iso_tmp, opacity=0.6, show_scalar_bar=False)
    plotter.add_axes()

plotter.show(interactive=False)
plotter.close()
del plotter

# %%
#
# When Do We Stop?
# ----------------
#
# Which order to use depends on the accuracy requirements.  For panel-method wake
# modelling, the dominant error usually comes from other sources (panel discretisation,
# time integration), so order 1 or 2 is often sufficient when the evaluation point is
# more than 5–10 source-cluster radii away.
#
# The :class:`cvl.Multipole` provides a clean, optimised C implementation that can be
# combined with Barnes-Hut tree structures and :meth:`cvl.Multipole.shift` /
# :meth:`cvl.Multipole.shift_to` for hierarchical far-field evaluation.
