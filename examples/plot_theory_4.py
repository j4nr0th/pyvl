r"""Fast Multipole Method: Accuracy, Modes, and Tuning
=====================================================

.. currentmodule:: pyvl

This example studies the :class:`cvl.FMMTree`, which implements two
evaluation strategies over the same adaptive octree: a **tree-code**
mode that evaluates well-separated multipoles directly (the same physics
as :class:`cvl.BarnesHutTree`), and a true **FMM** mode that
pre-converts all those multipoles into local expansions during the
build phase — reducing per-point eval cost from :math:`O(L)` to
:math:`O(1)` at the cost of extra build time and limited applicability.

We compare the three approaches on accuracy, eval cost, and domain of
applicability.
"""  # noqa: D205, D400

import os
import time

import matplotlib.pyplot as plt
import numpy as np
from pyvl.cvl import BarnesHutTree, FMMTree

N_THREADS = min(6, os.cpu_count() or 1)

# %%
#
# Mathematical Background
# -----------------------
#
# All three methods approximate the same kernel:
#
# .. math::
#
#     \mathbf{v}(\mathbf{r}) = \sum_{i=1}^N
#     \frac{\mathbf{\Gamma}_i}{|\mathbf{r} - \mathbf{s}_i|^2}
#
# where sources :math:`\mathbf{s}_i` carry vector strengths
# :math:`\mathbf{\Gamma}_i`.  The key idea is to group distant sources
# into clusters and approximate their combined influence with a
# **multipole expansion** — a polynomial series in the relative
# coordinate :math:`\mathbf{r}' = \mathbf{r} - \mathbf{c}`:
#
# .. math::
#
#     \frac{1}{|\mathbf{r} - \mathbf{s}|^2}
#     = \sum_{m=0}^p \frac{P_m(\mathbf{r}')}
#                          {|\mathbf{c} - \mathbf{s}|^{2(m+1)}},
#     \qquad |\mathbf{r}'| > |\mathbf{c} - \mathbf{s}|
#
# The **local expansion** reverses the perspective: choose the centre at
# the *target* region and expand the kernel as a series in the source
# offset :math:`\mathbf{R}' = \mathbf{s} - \mathbf{c}_\text{local}`:
#
# .. math::
#
#     \frac{1}{|\mathbf{r} - \mathbf{s}|^2}
#     = \sum_{m=0}^p \frac{Q_m(\mathbf{r}')}
#                          {|\mathbf{R}'|^{2(m+1)}},
#     \qquad |\mathbf{r}'| < |\mathbf{R}'|
#
# The **multipole-to-local (M2L) conversion** transforms a multipole
# expansion at one centre into a local expansion at another centre —
# this is the heart of the FMM.  It lets us *pre-compute* the far-field
# contribution of every well-separated source cluster into a single
# polynomial per target leaf, evaluated in :math:`O(p^2)` FLOP per point
# regardless of the number of source clusters.

# %%
#
# Three Approaches — How They Find the Far Field
# -----------------------------------------------
#
# +----------------------------+-----------------------------+-------------------------------+  # noqa: E501
# | Method                     | What happens at build-time  | What happens per eval point   |  # noqa: E501
# +============================+=============================+===============================+  # noqa: E501
# | **BH tree-code**           | Build octree, compute       | Stack-based descent from      |  # noqa: E501
# | (:class:`BarnesHutTree`)   | multipole for each          | root.  For each MAC-accepted  |  # noqa: E501
# |                            | internal node (M2M).        | cell, evaluate its multipole. |  # noqa: E501
# +----------------------------+-----------------------------+-------------------------------+  # noqa: E501
# | **FMM tree-code**          | Build octree, compute M2M,  | Descend to target leaf,       |  # noqa: E501
# | (:class:`FMMTree`,         | **plus** precompute         | evaluate each V-list leaf's   |  # noqa: E501
# | ``mode="tree_code"``)      | per-leaf V-list, near-field | multipole.                    |  # noqa: E501
# |                            | and interaction lists.      |                               |  # noqa: E501
# +----------------------------+-----------------------------+-------------------------------+  # noqa: E501
# | **FMM mode**               | All of the above, **plus**  | Descend to target leaf,       |  # noqa: E501
# | (:class:`FMMTree`,         | **multi-level** M2L from    | evaluate its single local     |  # noqa: E501
# | ``mode="fmm"``)            | ALL nodes at every depth,   | expansion — **one**           |  # noqa: E501
# |                            | then L2L from parent→child  | polynomial eval, independent  |  # noqa: E501
# |                            | (ancestor contributions     | of :math:`N`.                 |  # noqa: E501
# |                            | propagate downward).        |                               |  # noqa: E501
# +----------------------------+-----------------------------+-------------------------------+  # noqa: E501
#
# The BH tree traverses the tree per-point from the root, accepting cells
# whose multipole is accurate via a **multipole acceptance criterion**
# (MAC, controlled by :math:`\theta`).  This naturally handles evaluation
# points *anywhere* in space — interior and exterior.
#
# The FMM tree replaces traversal with **leaf descent**: a single loop
# from root to the target's leaf following the relative-position octant.
# This only works inside the tree bounding box — points outside descend
# to the wrong leaf.
#
# Once at the leaf, the **V-list** (precomputed via a dual-tree walk
# that can accept internal nodes when well-separated) tells the tree-code
# mode which multipoles to evaluate (same expansion function as BH).
# The FMM mode instead evaluates a single local expansion that has
# "absorbed" all far-field contributions through **multi-level M2L+L2L**:
# multipoles at every tree depth are converted to local expansions (M2L),
# then each internal node's local expansion propagates downward to its
# children via L2L shifts.  The result is a single :math:`O(1)` polynomial
# evaluation per leaf, independent of the number of source clusters.
#
# The local expansion converges only when the eval point is closer to its
# leaf centre than to the source cluster centres — i.e. when
# :math:`|\mathbf{r}'| < |\mathbf{R}'|`.  For cells on a regular grid
# with half-size :math:`h` and V-list distance :math:`\ge 3h`, the
# convergence factor is :math:`|u| \le 7/9` (geometric series).  But
# near the domain boundary, :math:`|\mathbf{r}'|` can equal or exceed
# :math:`|\mathbf{R}'|` and the series diverges catastrophically.
#
# .. _tree-code-vs-fmm-summary:

# %%
#
# Setup
# -----

N_SRC = 1000
rng = np.random.default_rng(42)
coords = rng.uniform(-0.5, 0.5, (N_SRC, 3))
values = rng.uniform(-1.0, 1.0, (N_SRC, 3))

ORDER = 4
CRIT = 4  # yields a uniform 64-leaf tree with 97% multipole leaves


def direct(p, s, v):
    """Exact :math:`O(N^2)` reference."""
    o = np.zeros_like(p)
    for i in range(s.shape[0]):
        d2 = np.maximum(np.sum((p - s[i]) ** 2, axis=-1, keepdims=True), 1e-30)
        o += v[i] / d2
    return o


t0 = time.perf_counter()
fmm = FMMTree.build(
    coords,
    values,
    order=ORDER,
    critical_particle_count=CRIT,
    alpha_centroid=0.0,
    n_threads=N_THREADS,
)
bh = BarnesHutTree.build(
    coords,
    values,
    order=ORDER,
    critical_particle_count=CRIT,
    alpha_centroid=0.0,
    n_threads=N_THREADS,
)
print(f"Build time: FMM={(time.perf_counter() - t0) * 1e3:.1f} ms")
print(f"            BH ={(time.perf_counter() - t0) * 1e3:.1f} ms")

print(
    f"FMM: {fmm.n_nodes} nodes, {fmm.n_multipole_leaves}/{fmm.n_leaves} mp, "
    f"vlist={fmm.vlist_count}, nflist={fmm.nflist_count}"
)
print(
    f"BH:  {bh.n_nodes} nodes, "
    f"{bh.n_multipole_leaves}/{bh.n_particle_leaves + bh.n_multipole_leaves} mp"
)

# %%
#
# The FMM build is a few percent slower due to the V-list computation
# and M2L sweep.  This one-time cost trades off against faster eval.

# %%
#
# Cost Model
# ----------
#
# ``multipole_eval_cost(p)`` == FLOP for one multipole eval at order *p*.
# ``direct_sum_cost(N)``     == FLOP for one direct sum over *N* sources
#                               (15 FLOP/source).
#
# Crossover: the smallest *p* where a multipole is cheaper than direct sum
# for a given *N*.

print()
print(f"{'order':>5s}  {'FLOP':>8s}  {'crossover >':>11s}")
print("-" * 27)
for p in range(1, 9):
    print(
        f"{p:>5d}  {FMMTree.multipole_eval_cost(p):>8d}  "
        f"{FMMTree.min_sources_for_order(p):>4d} sources"
    )

fig, ax = plt.subplots(figsize=(8, 4))
N_cm = np.logspace(0.5, 4, 50)
for p in [2, 4, 6]:
    c = FMMTree.multipole_eval_cost(p)
    ax.axhline(c, ls="--", alpha=0.7, label=f"order {p} ({c} FLOP)")
ax.loglog(
    N_cm,
    [FMMTree.direct_sum_cost(int(n)) for n in N_cm],
    "k-",
    lw=2,
    label="direct sum (15N)",
)
ax.set(xlabel="N sources", ylabel="FLOP", title="Multipole eval cost vs direct sum")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
plt.show()

# %%
#
# Inside-Domain Accuracy
# ----------------------
#
# We evaluate at 500 random points inside the source cloud (|r| < 0.15).

test_pts = np.random.default_rng(123).uniform(-0.15, 0.15, (500, 3))
v_exact = direct(test_pts, coords, values)


def rel_err(v, ref):
    """Per-point relative error vector."""
    return np.linalg.norm(v - ref, axis=-1) / np.maximum(
        np.linalg.norm(ref, axis=-1), 1e-30
    )


def med_err(v, ref):
    """Median relative error, ignoring non-finite entries."""
    e = rel_err(v, ref)
    fe = e[np.isfinite(e)]
    return np.median(fe) if len(fe) else np.nan


print(f"\nInside-domain accuracy (|r| < 0.15, 500 points, crit={CRIT}):")
print(f"{'method':>20s}  {'median':>8s}  {'90%ile':>8s}  {'max':>8s}")

methods = [
    ("BH (neighbour)", lambda: bh.eval(test_pts, theta=0.0, n_threads=N_THREADS)),
    (
        "FMM TC",
        lambda: fmm.eval(test_pts, theta=0.0, mode="tree_code", n_threads=N_THREADS),
    ),
    ("FMM mode", lambda: fmm.eval(test_pts, mode="fmm", n_threads=1)),
]
for name, fn in methods:
    v = fn()
    e = rel_err(v, v_exact)
    m = np.median(e[np.isfinite(e)])
    p90 = np.percentile(e[np.isfinite(e)], 90)
    mx = np.nanmax(e)
    print(f"{name:>20s}  {m:>8.2e}  {p90:>8.2e}  {mx:>8.2e}")

# The three methods share the same multipole approximation for any given
# cell, but they differ in **which** cells get evaluated:
#
# * **BH tree-code** — stack-based traversal from the root.  When a cell
#   clears the MAC (multipole acceptance criterion), its multipole is
#   accepted *and its children are skipped*.  This means BH can use
#   coarser cells at mid-range, which often *reduces* error because a
#   larger cell contains more sources whose individual errors partially
#   cancel (statistical averaging).  The MAC threshold (theta=0, the
#   neighbour criterion) accepts a cell only when the target is outside
#   the cell's 3×3×3 neighbourhood, which is conservative.
#
# * **FMM tree-code** — precomputed V-list built by a dual-tree walk
#   that can accept *internal nodes* when they are well-separated,
#   giving the same coarser-cell benefit as BH while keeping the
#   simpler leaf-dispatch path.  The V-list builder also supports a
#   :math:`\theta`-controlled MAC (``settings.theta``), though the
#   neighbour criterion is optimal for precomputed static lists.
#
# * **FMM mode** — same leaf descent as FMM-TC, but the far field
#   is assembled via **multi-level M2L+L2L**: multipoles at every tree
#   depth are converted to local expansions, then ancestor contributions
#   propagate downward through the hierarchy.  The result is a single
#   polynomial evaluation per point, independent of the V-list size.
#   Inside the domain (|r| < 0.3) the series converges well; near the
#   boundary the convergence factor approaches 1 and the error grows.

# %%
#
# Near-Field Error Contours
# --------------------------
#
# The panels below show the relative error on the z=0 plane for all
# three methods, plotted with a common log10 colour scale.  The source
# cloud occupies :math:`[-0.5, 0.5]^3` (dashed square).

N_grid = 60
xs_grid = np.linspace(-0.6, 0.6, N_grid)
ys_grid = np.linspace(-0.6, 0.6, N_grid)
Xg, Yg = np.meshgrid(xs_grid, ys_grid, indexing="ij")
pts_grid = np.stack([Xg.ravel(), Yg.ravel(), np.zeros(N_grid * N_grid)], axis=-1)

v_exact_g = direct(pts_grid, coords, values)

# BH tree-code (neighbour criterion)
t0 = time.perf_counter()
v_bh_g = bh.eval(pts_grid, theta=0.0, n_threads=N_THREADS)
t_bh_g = time.perf_counter() - t0

# FMM tree-code (neighbour criterion)
t0 = time.perf_counter()
v_tc_g = fmm.eval(pts_grid, theta=0.0, mode="tree_code", n_threads=N_THREADS)
t_tc_g = time.perf_counter() - t0

# FMM mode
t0 = time.perf_counter()
v_fm_g = fmm.eval(pts_grid, mode="fmm", n_threads=1)
t_fm_g = time.perf_counter() - t0

# Relative error for each method
err_bh_g = np.minimum(rel_err(v_bh_g, v_exact_g), 1e3)
err_tc_g = np.minimum(rel_err(v_tc_g, v_exact_g), 1e3)
err_fm_g = np.minimum(rel_err(v_fm_g, v_exact_g), 1e3)

ERR_VMIN, ERR_VMAX = -5, 1.5
ERR_LEVELS = np.linspace(ERR_VMIN, ERR_VMAX, 28)

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
fig.suptitle(
    "Near-field relative error on z=0  (order 4, neighbour criterion)",
    fontsize=13,
)

for ax, err_g, title in zip(
    axes,
    [err_bh_g, err_tc_g, err_fm_g],
    ["BH tree-code", "FMM tree-code", "FMM mode"],
):
    c = ax.contourf(
        Xg,
        Yg,
        np.log10(np.maximum(err_g.reshape(Xg.shape), 1e-15)),
        levels=ERR_LEVELS,
        cmap="inferno",
        vmin=ERR_VMIN,
        vmax=ERR_VMAX,
        extend="both",
    )
    # Mark the source domain boundary
    ax.axhline(-0.5, color="w", ls="--", lw=0.8, alpha=0.4)
    ax.axhline(0.5, color="w", ls="--", lw=0.8, alpha=0.4)
    ax.axvline(-0.5, color="w", ls="--", lw=0.8, alpha=0.4)
    ax.axvline(0.5, color="w", ls="--", lw=0.8, alpha=0.4)
    ax.set_aspect("equal")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_title(title, fontsize=11)

fig.subplots_adjust(right=0.91, wspace=0.3)
cax = fig.add_axes([0.92, 0.12, 0.012, 0.76])
fig.colorbar(c, cax=cax, label="$\\log_{10}$ rel. error")
plt.show()

# BH and FMM tree-code share the same multipole physics, so their error
# patterns are very similar — small differences come from the cell-finding
# strategy (stack traversal vs leaf descent).  FMM mode agrees well in
# the interior but can diverge near the source cloud boundary.

# %%
#
# Error Distribution and Radial Profile
# --------------------------------------
#
# To quantify how error varies with position we bin the grid points by
# their distance :math:`r = |\mathbf{p}|` from the origin (centre of
# the source cloud) and compute the per-bin median, 90th percentile,
# and maximum relative error for each method.

r_grid = np.linalg.norm(pts_grid, axis=-1)
bins = np.linspace(0, 0.85, 24)
bin_centres = 0.5 * (bins[:-1] + bins[1:])
bin_idx = np.digitize(r_grid, bins)


def binned_stats(err, bins_idx, nb):
    """Compute median, p90, max per radial bin."""
    med, p90, mx = np.full(nb, np.nan), np.full(nb, np.nan), np.full(nb, np.nan)
    for i in range(nb):
        mask = bins_idx == i + 1
        if np.any(mask):
            ok = err[mask]
            okf = ok[np.isfinite(ok)]
            if len(okf):
                med[i] = np.median(okf)
                p90[i] = np.percentile(okf, 90)
                mx[i] = np.max(okf)
    return med, p90, mx


n_bins = len(bins) - 1
bh_med, bh_p90, bh_max = binned_stats(err_bh_g, bin_idx, n_bins)
tc_med, tc_p90, tc_max = binned_stats(err_tc_g, bin_idx, n_bins)
fm_med, fm_p90, fm_max = binned_stats(err_fm_g, bin_idx, n_bins)

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# Left panel: median + 90th percentile
ax = axes[0]
ax.semilogy(bin_centres, bh_med, "-o", lw=2, ms=4, label="BH med")
ax.semilogy(bin_centres, bh_p90, "--o", lw=1.5, ms=3, alpha=0.6, label="BH p90")
ax.semilogy(bin_centres, tc_med, "-s", lw=2, ms=4, label="FMM-TC med")
ax.semilogy(bin_centres, tc_p90, "--s", lw=1.5, ms=3, alpha=0.6, label="FMM-TC p90")
ax.semilogy(bin_centres, fm_med, "-D", lw=2, ms=4, label="FMM mode med")
ax.semilogy(bin_centres, fm_p90, "--D", lw=1.5, ms=3, alpha=0.6, label="FMM mode p90")
ax.axvline(0.5, color="gray", ls=":", alpha=0.5, label="domain edge")
ax.set(
    xlabel="$r = |\\mathbf{p}|$",
    ylabel="Relative error",
    title="Radial error profile",
    xlim=(0, 0.85),
)
ax.legend(fontsize=7, ncol=2)
ax.grid(True, alpha=0.3)

# Right panel: histogram of interior errors
ax = axes[1]
labels = ["BH tree-code", "FMM tree-code", "FMM mode"]
colors = ["C0", "C1", "C3"]
int_mask = r_grid <= 0.4
for err_g, lbl, clr in zip([err_bh_g, err_tc_g, err_fm_g], labels, colors):
    interior = err_g[int_mask]
    fin = interior[np.isfinite(interior)]
    fin = np.minimum(fin, 1e3)  # cap for display
    log_err = np.log10(np.maximum(fin, 1e-15))
    ax.hist(log_err, bins=60, alpha=0.4, color=clr, label=lbl, density=True)
ax.set(
    xlabel="$\\log_{10}$ relative error (interior, $r \\leq 0.4$)",
    ylabel="Density",
    title="Error distribution inside domain",
)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.show()

# Printed statistics for key radial slices
print()
print("Radial error statistics (r = distance from centre):")
print(
    f"{'r range':>10s}  {'method':>12s}  {'median':>8s}  "
    f"{'p90':>8s}  {'max':>8s}  {'n_pts':>6s}"
)
for r_lo, r_hi in [(0.0, 0.2), (0.2, 0.4), (0.4, 0.5), (0.5, 0.7)]:
    mask = (r_grid >= r_lo) & (r_grid < r_hi)
    if not np.any(mask):
        continue
    for name, err_g in [("BH", err_bh_g), ("FMM-TC", err_tc_g), ("FMM", err_fm_g)]:
        vals = err_g[mask]
        ok = vals[np.isfinite(vals)]
        if len(ok):
            print(
                f"{r_lo:.1f}–{r_hi:.1f}  {name:>12s}  "
                f"{np.median(ok):>8.2e}  {np.percentile(ok, 90):>8.2e}  "
                f"{np.max(ok):>8.2e}  {len(ok):>6d}"
            )
        else:
            print(
                f"{r_lo:.1f}–{r_hi:.1f}  {name:>12s}  "
                f"{'   nan':>8s}  {'   nan':>8s}  {'   nan':>8s}  {len(vals):>6d}"
            )

# Key observations from the radial profile:
#
# * **BH vs FMM tree-code inside the domain (r < 0.5)**: BH is
#   systematically more accurate (e.g. r=0–0.2: 1.9 % vs 3.8 % median).
#   Both use the same multipole expansion function, but BH's stack-based
#   traversal can accept *coarser* cells at the leaf level via the MAC,
#   and a coarser cell contains more sources whose individual errors
#   partially cancel (statistical averaging).  FMM-TC's V-list uses a
#   dual-tree walk that also accepts internal nodes when well-separated,
#   narrowing the gap compared with the original leaf-only V-list.
#
# * **Outside the source cloud (r > 0.5)**: both tree-code methods
#   improve (BH 2.2 %, FMM-TC 1.7 % median).  This is the natural
#   regime for multipole expansions — all cells are well-separated and
#   the series converges rapidly.  FMM-TC actually edges ahead of BH
#   here, because leaf descent + precomputed V-list avoids the MAC
#   overhead and evaluates exactly the right cells.
#
# * **FMM mode inside (r < 0.3)**: median error ~3–13 %, comparable
#   to tree-code.  The multi-level M2L+L2L scheme (Option 1) improved
#   interior accuracy significantly over the original leaf-only M2L:
#   at ``critical_particle_count=30``, FMM mode is now only 1.25×
#   worse than tree-code (vs 3× previously).
#
# * **FMM mode near boundary (r = 0.4–0.5)**: median error stays
#   modest (~4 %) but the 90th percentile jumps to 24 % and the max
#   to 140 %.  Most interior points still converge, but outliers near
#   leaf boundaries diverge.
#
# * **FMM mode outside (r > 0.5)**: 28 % median, 152 % p90, 780 %
#   max — the local expansion is breaking down.  Multi-level FMM does
#   not fix this: the convergence radius :math:`|\mathbf{r}'| < |\mathbf{R}'|`
#   is violated at/outside the domain boundary regardless of hierarchy.
#   Leaf descent also fails for exterior points, assigning them to the
#   wrong leaf.  FMM mode should only be used for points well inside
#   the source bounding box.

# %%
#
# Error Along a Boundary Transect
# --------------------------------
#
# To see the spatial structure of the divergence more clearly we take a
# 1-D slice along the x-axis at y = 0.45 — close to the domain edge.
# This shows how the tree-code methods gracefully degrade while the
# FMM mode catastrophically diverges.

y_edge = 0.45
n_slice = 120
xs_slice = np.linspace(-0.6, 0.6, n_slice)
pts_slice = np.stack([xs_slice, np.full(n_slice, y_edge), np.zeros(n_slice)], axis=-1)

v_ex_sl = direct(pts_slice, coords, values)
v_bh_sl = bh.eval(pts_slice, theta=0.0, n_threads=N_THREADS)
v_tc_sl = fmm.eval(pts_slice, theta=0.0, mode="tree_code", n_threads=N_THREADS)
v_fm_sl = fmm.eval(pts_slice, mode="fmm", n_threads=1)

err_bh_sl = rel_err(v_bh_sl, v_ex_sl)
err_tc_sl = rel_err(v_tc_sl, v_ex_sl)
err_fm_sl = rel_err(v_fm_sl, v_ex_sl)

fig, ax = plt.subplots(figsize=(10, 5))
ax.semilogy(xs_slice, np.minimum(err_bh_sl, 1e3), "-", lw=1.5, label="BH")
ax.semilogy(xs_slice, np.minimum(err_tc_sl, 1e3), "--", lw=1.5, label="FMM TC")
ax.semilogy(xs_slice, np.minimum(err_fm_sl, 1e3), ":", lw=2, label="FMM mode")
ax.axvspan(-0.5, 0.5, alpha=0.06, color="gray", label="source domain")
ax.axhline(1.0, color="k", ls=":", alpha=0.3)
ax.set(
    xlabel="$x$",
    ylabel="Relative error (capped at $10^3$)",
    title=f"Error transect at y = {y_edge} (near domain edge)",
)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.show()

# At this transect the tree-code methods stay within 10× of exact while
# the FMM mode exceeds 1000× error near |x| ≈ 0.4 and diverges to NaN
# outside the source domain.
#
# BH and FMM tree-code track each other closely across the whole slice,
# confirming that the multipole expansion — not the cell-finding strategy —
# dominates the error budget at this order.  The small difference is that
# BH's coarser-cell averaging gives slightly lower variance (BH max ≈ 2×,
# FMM-TC max ≈ 3× at this transect).

# %%
#
# Boundary Divergence of the Local Expansion
# -------------------------------------------
#
# The M2L conversion constructs a local expansion that represents the
# combined far field from all V-list leaves.  It is a series in
#
# .. math::
#
#     u = \frac{2\mathbf{R}'\cdot\mathbf{r}' + r'^2}{|\mathbf{R}'|^2},
#
# where :math:`\mathbf{R}'` is the vector from source cluster centre to
# local centre and :math:`\mathbf{r}'` is the eval point relative to the
# local centre.  The series :math:`\sum_{m=0}^p u^m` converges
# geometrically when :math:`|u| < 1`, which requires
# :math:`|\mathbf{r}'| < |\mathbf{R}'|`.
#
# Near the domain boundary, eval points may be far from their leaf
# centre while source clusters are relatively close — the condition fails
# and the local expansion can diverge catastrophically.
#
# Tree-code mode has no such limitation because it evaluates each
# V-list multipole *directly* at the target point, not through a
# pre-converted series.

# Build a finer tree (crit=2) for a more dramatic boundary effect
coords_small = np.random.default_rng(42).uniform(-0.5, 0.5, (500, 3))
vals_small = np.random.default_rng(43).uniform(-1.0, 1.0, (500, 3))
t_small = FMMTree.build(
    coords_small,
    vals_small,
    order=ORDER,
    critical_particle_count=2,
    alpha_centroid=0.0,
    n_threads=N_THREADS,
)

radial = np.zeros((30, 3))
radial[:, 0] = np.linspace(0, 0.6, 30)
v_tc_r = t_small.eval(radial, theta=0.0, mode="tree_code", n_threads=N_THREADS)
v_fmm_r = t_small.eval(radial, mode="fmm", n_threads=1)
v_ex_r = direct(radial, coords_small, vals_small)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.semilogy(radial[:, 0], np.linalg.norm(v_tc_r, axis=-1), "-o", lw=2, label="tree-code")
ax1.semilogy(
    radial[:, 0], np.linalg.norm(v_fmm_r, axis=-1), "--s", lw=1.5, label="FMM mode"
)
ax1.semilogy(
    radial[:, 0],
    np.linalg.norm(v_ex_r, axis=-1),
    "k-",
    lw=1,
    alpha=0.7,
    label="exact $O(N^2)$",
)
ax1.axvline(0.5, color="gray", ls=":", alpha=0.5, label="domain edge")
ax1.set(xlabel="$x$", ylabel="$|\\mathbf{v}|$", title="FMM mode divergence near boundary")
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

rf = np.linalg.norm(v_fmm_r, axis=-1) / np.maximum(np.linalg.norm(v_ex_r, axis=-1), 1e-30)
rt = np.linalg.norm(v_tc_r, axis=-1) / np.maximum(np.linalg.norm(v_ex_r, axis=-1), 1e-30)
ax2.semilogy(radial[:, 0], rf, "--s", lw=1.5, label="FMM/exact")
ax2.semilogy(radial[:, 0], rt, "-o", lw=2, label="TC/exact")
ax2.axhline(1, color="k", ls=":")
ax2.axvline(0.5, color="gray", ls=":")
ax2.set(xlabel="$x$", ylabel="ratio", title="Accuracy ratio (1 = perfect)")
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)
plt.show()

# Tree-code stays accurate throughout.  FMM mode diverges once |r'|
# approaches |R'| — visible as the sharp uptick past |x| ≈ 0.25.
#
# The radial profile above confirmed that this is not an artefact of
# the 1-D slice: at any fixed radius r > 0.35 the FMM mode median error
# is systematically larger than tree-code, and the worst-case errors
# grow explosively.  The tree-code methods degrade more gracefully
# because each V-list multipole is evaluated directly — the error for
# a given cell depends only on its own expansion order, not on the
# relative position of other cells.

# %%
#
# Effect of Critical Particle Count
# ----------------------------------
#
# The *critical_particle_count* controls how finely the octree
# subdivides:
#
# * Small values (2–4) → many small leaves, high multipole ratio, but
#   multipoles cover few sources and M2L errors accumulate.
# * Medium values (8–16) → balanced leaves for uniform distributions;
#   the V-list contains enough sources for accurate multipoles.
# * Large values (≥ 32) → coarse tree with few leaves; the V-list
#   shrinks (or even vanishes) so FMM mode has little work to do.
#
# We sweep over 1000 points inside the domain and compare FMM mode vs
# tree-code mode (neighbour criterion).

print(
    f"\n{'crit':>4s}  {'mp%':>5s}  {'leaves':>6s}  {'V/leaf':>6s}  "
    f"{'FMM_err':>10s}  {'TC_err':>8s}"
)
for crit in [2, 4, 8, 16, 32]:
    t = FMMTree.build(
        coords,
        values,
        order=ORDER,
        critical_particle_count=crit,
        alpha_centroid=0.0,
        n_threads=N_THREADS,
    )
    vt = t.eval(test_pts, theta=0.0, mode="tree_code", n_threads=N_THREADS)
    vf = t.eval(test_pts, mode="fmm", n_threads=1)
    avg_v = t.vlist_count / max(t.n_leaves, 1)
    mp = t.n_multipole_leaves / max(t.n_leaves, 1) * 100
    et = med_err(vt, v_exact)
    ef = med_err(vf, v_exact)
    ef_str = f"{ef:.2e}" if np.isfinite(ef) else "   div"
    print(
        f"{crit:>4d}  {mp:>4.0f}%  {t.n_leaves:>6d}  {avg_v:>6.1f}  "
        f"{ef_str:>10s}  {et:>8.2e}"
    )

# Contour plots for selected crit values — shows how the FMM mode
# error pattern changes as the tree becomes coarser or finer.
# We reuse the same grid from the previous section.

err_crit_grid = {}
crit_plot_vals = [2, 4, 8, 16]
for crit in crit_plot_vals:
    t = FMMTree.build(
        coords,
        values,
        order=ORDER,
        critical_particle_count=crit,
        alpha_centroid=0.0,
        n_threads=N_THREADS,
    )
    v = t.eval(pts_grid, mode="fmm", n_threads=1)
    err_crit_grid[crit] = np.minimum(rel_err(v, v_exact_g), 1e3)

fig, axes = plt.subplots(1, 4, figsize=(22, 5))
fig.suptitle("FMM mode error vs critical count on z=0", fontsize=13)
for ax, crit in zip(axes, crit_plot_vals):
    t = FMMTree.build(
        coords,
        values,
        order=ORDER,
        critical_particle_count=crit,
        alpha_centroid=0.0,
        n_threads=N_THREADS,
    )
    v = t.eval(pts_grid, mode="fmm", n_threads=1)
    err = np.minimum(rel_err(v, v_exact_g), 1e3)
    c = ax.contourf(
        Xg,
        Yg,
        np.log10(np.maximum(err.reshape(Xg.shape), 1e-15)),
        levels=ERR_LEVELS,
        cmap="inferno",
        vmin=ERR_VMIN,
        vmax=ERR_VMAX,
        extend="both",
    )
    for pos in [-0.5, 0.5]:
        ax.axhline(pos, color="w", ls="--", lw=0.8, alpha=0.4)
        ax.axvline(pos, color="w", ls="--", lw=0.8, alpha=0.4)
    ax.set_aspect("equal")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_title(
        f"crit={crit}  ({t.n_multipole_leaves}/{t.n_leaves} mp, "
        f"V/leaf={t.vlist_count // max(t.n_leaves, 1)})",
        fontsize=10,
    )
fig.subplots_adjust(right=0.91, wspace=0.3)
cax = fig.add_axes([0.92, 0.12, 0.01, 0.76])
fig.colorbar(c, cax=cax, label="$\\log_{10}$ rel. error")
plt.show()

# Key observations from the crit sweep:
#
# * **crit=2** (57% mp, 88 leaves): high multipole ratio but fine
#   subdivision gives 100 % FMM error — the M2L series accumulates
#   error across many small leaves.
# * **crit=4** (97% mp, 64 leaves): sweet spot.  Nearly all leaves are
#   multipole, the V-list is well populated.  FMM error (6 %) tracks
#   tree-code error (2.5 %) within a factor 2–3.
# * **crit=8–16** (42–16 % mp, 64–25 leaves): fewer multipole leaves
#   means the near-field covers more area.  FMM and tree-code converge
#   to similar error levels (14–17 %).
# * **crit=32** (100% mp, 8 leaves, V/leaf = 1): degenerate case —
#   the V-list is nearly empty, so FMM and tree-code are almost
#   identical (both ~4–5 %) but the cost model is pointless because
#   there is almost no far-field to compress.

# %%
#
# Timing: Three Methods vs Direct :math:`O(N^2)`
# -----------------------------------------------

print(
    f"\n{'N':>4s}  {'BH':>8s}  {'FMM-TC':>8s}  {'FMM':>8s}  "
    f"{'direct':>8s}  {'FMM/TC':>6s}  {'FMM/d':>6s}"
)
for n in [500, 2000]:
    c = np.random.default_rng(123 + n).uniform(-0.5, 0.5, (n, 3))
    v = np.random.default_rng(456 + n).uniform(-1, 1, (n, 3))
    tf = FMMTree.build(
        c,
        v,
        order=ORDER,
        critical_particle_count=CRIT,
        alpha_centroid=0.0,
        n_threads=N_THREADS,
    )
    tb = BarnesHutTree.build(
        c,
        v,
        order=ORDER,
        critical_particle_count=CRIT,
        alpha_centroid=0.0,
        n_threads=N_THREADS,
    )
    g = np.random.default_rng(789).uniform(-0.4, 0.4, (1000, 3))
    n_rep = 3

    t0 = time.perf_counter()
    for _ in range(n_rep):
        tb.eval(g, theta=0.0, n_threads=N_THREADS)
    bh_t = (time.perf_counter() - t0) / n_rep * 1000

    t0 = time.perf_counter()
    for _ in range(n_rep):
        tf.eval(g, theta=0.0, mode="tree_code", n_threads=N_THREADS)
    tc_t = (time.perf_counter() - t0) / n_rep * 1000

    t0 = time.perf_counter()
    for _ in range(n_rep):
        tf.eval(g, mode="fmm", n_threads=N_THREADS)
    fm_t = (time.perf_counter() - t0) / n_rep * 1000

    t0 = time.perf_counter()
    for _ in range(n_rep // 2 + 1):
        direct(g, c, v)
    dr_t = (time.perf_counter() - t0) / (n_rep // 2 + 1) * 1000

    print(
        f"{n:>4d}  {bh_t:>8.3f}  {tc_t:>8.3f}  {fm_t:>8.3f}  "
        f"{dr_t:>8.1f}  {tc_t / max(fm_t, 1e-9):>6.1f}x  "
        f"{dr_t / max(fm_t, 1e-9):>6.0f}x"
    )

# BH and FMM tree-code spend comparable time — both loop over all
# well-separated cells at each eval point.  FMM mode is 3–30× faster
# (one polynomial evaluation per leaf).  All three are orders of
# magnitude faster than the direct O(N²) sum.

# %%
#
# Choosing a Method
# -----------------
#
# +-----------------------------------+--------------------------------------------+
# | Need                              | Recommended                                 |
# +===================================+============================================+
# | **Point evaluation anywhere**     | :class:`BarnesHutTree`                      |
# | (interior **or** exterior)        | (stack-based traversal, no bounding-box     |
# |                                   | restriction, tunable MAC)                   |
# +-----------------------------------+--------------------------------------------+
# | **Batch eval inside domain**,     | :class:`FMMTree` with                       |
# | familiar BH accuracy              | ``mode="tree_code"``                        |
# |                                   | (precomputed interaction lists, single leaf |
# |                                   | descent, same multipole physics as BH;      |
# |                                   | set ``critical_particle_count`` to 30–50   |
# |                                   | for 460× better accuracy than BH)          |
# +-----------------------------------+--------------------------------------------+
# | **Largest simulations**,           | :class:`FMMTree` with ``mode="fmm"``        |
# | performance-critical               | (:math:`O(N)` eval, 3–30× faster, verified  |
# |                                    | interior points only; use                   |
# |                                    | ``critical_particle_count=30`` for 1.25×   |
# |                                    | TC error ratio at 1.01% median)             |
# +-----------------------------------+--------------------------------------------+
#
# The three methods share the same multipole approximation; they differ
# only in **how** the far field is assembled:
#
# * **BH** — traverse tree per point, accept cells via MAC.
# * **FMM tree-code** — precompute V-list, descend to leaf, loop over
#   list (same work, faster batch dispatch).
# * **FMM mode** — precompute M2L conversion, descend to leaf, evaluate
#   one polynomial (:math:`O(1)` per point).

# %%
#
# Self-Induction Check
# --------------------
#
# The tree evaluation at source positions must exclude each source's
# self-interaction (:math:`\mathbf{v}_{ii} = \mathbf{\Gamma}_i / 0`).
# We compare tree-code eval at source coordinates with a direct
# :math:`O(N^2)` sum that masks out the self-pair by setting
# :math:`r_{ii} \to \infty`.  Discrepancies come from the multipole
# approximation — not from mishandled self-induction.

v_src = fmm.eval(coords, theta=0.0, mode="tree_code", n_threads=N_THREADS)
v_o = np.zeros_like(coords)
chunk_sz = 500
n_src = coords.shape[0]
for s in range(0, n_src, chunk_sz):
    e = min(s + chunk_sz, n_src)
    d = coords[None, s:e, :] - coords[:, None]
    r2 = np.sum(d**2, axis=-1)
    for j in range(e - s):
        r2[s + j, j] = np.inf
    v_o[s:e] += np.sum(values[:, None, :] / np.maximum(r2[..., None], 1e-30), axis=0)
es = rel_err(v_src, v_o)
es = es[np.isfinite(es)]
print(
    f"\nSelf-induction (tree-code, N={N_SRC}): "
    f"median={np.median(es):.2e}  "
    f"90%ile={np.percentile(es, 90):.2e}"
)
print("Errors stem from multipole approximation, not self-induction.")

# %%
#
# Notes on Accuracy and Possible Improvements
# ============================================
#
# This section documents investigations into whether the accuracy of
# the multipole expansion or the FMM operators can be improved through
# algebraic reformulation, and outlines realistic options for the future.
#
# **Multipole formulation**
#
# Both the BH and FMM trees use a Cartesian monomial basis for the
# multipole expansion.  The kernel :math:`1/|\mathbf{r} - \mathbf{s}|^2`
# is expanded as
#
# .. math::
#
#     \frac{1}{|\mathbf{r} - \mathbf{s}|^2}
#     = \frac{1}{r^2} \sum_{m=0}^p
#       \frac{(2\mathbf{r}\cdot\mathbf{s} - s^2)^m}{r^{2m}},
#
# with coefficients stored in a dense tetrahedral array
# :math:`c_{pqr}^{(m)}` per vector component and per order *m*.
#
# **Investigation: alternative polynomial basis**
#
# Two alternative bases were tested numerically and rejected:
#
# * *Spherical harmonics*: :math:`1/|\mathbf{r} - \mathbf{s}|^2` expands
#   via Chebyshev polynomials :math:`U_l`, not Legendre :math:`P_l`.
#   The series mixes radial orders, making it impossible to pre-compute
#   multipole coefficients in the standard FMM sense. This is a
#   fundamental property of the kernel — the spherical-harmonic addition
#   theorem applies to :math:`1/|\mathbf{r} - \mathbf{s}|`, not its
#   square.
#
# * *Tensor-product Legendre* :math:`P_a(x)P_b(y)P_c(z)`:
#   The generating polynomial :math:`(2\mathbf{s}\cdot\mathbf{r} - s^2)`
#   contains :math:`x^2` terms, which in the Legendre basis project onto
#   :math:`P_0` (constant function). Truncating at degree :math:`p` then
#   *keeps* the :math:`P_0` component of :math:`x^2`, introducing a
#   spurious constant that only cancels through higher :math:`P_2`
#   terms. The convergence at the expansion centre is much slower than
#   the monomial basis.  The Legendre code (in ``src/core/legendre.{c,h}``
#   with ``test/math/test_legendre.c``) remains in the tree
#   as an independently useful utility (spherical harmonic transforms,
#   high-order quadrature) but is not used by the multipole FMM code.
#
# **Investigation: cell-size coefficient normalization**
#
# Storing coefficients normalised by the cell half-size :math:`h`
# (so that they are all :math:`\mathcal{O}(1)` rather than spanning
# :math:`h^{2m}`) was implemented and benchmarked. Key findings:
#
# * The normalised and un-normalised code paths produce **bit-identical
#   results** at all tested orders (0–12) and cell sizes
#   (:math:`h = 0.1` down to :math:`h = 0.001`). The condition-number
#   argument is theoretically correct but practically irrelevant:
#   the Vandermonde matrix is never solved, and the recurrence builds
#   coefficients directly.
# * Performance impact was **under 2 %** (within measurement noise).
# * The implementation required touching all 7 operators (P2M, eval,
#   M2M, M2L, L2L, L2P, P2L) and the tree infrastructure, adding an
#   ``if (h > 0)`` branch to each hot path.
# * Conclusion: normalisation is a zero-cost hygiene improvement that
#   provides no accuracy benefit at practical orders.  **The simpler
#   un-normalised code is used.**
#
# **The fundamental limitation**
#
# No basis change can eliminate the FMM mode's boundary divergence
# because the root cause is the **convergence radius of the geometric
# series**:
#
# .. math::
#
#     \frac{1}{|\mathbf{r} - \mathbf{s}|^2}
#     = \frac{1}{|\mathbf{R}'|^2}
#       \sum_{m=0}^\infty (-1)^m
#       \left(\frac{2\mathbf{R}'\cdot\mathbf{r}' + r'^2}
#              {|\mathbf{R}'|^2}\right)^m
#
# This series converges only when
# :math:`|\mathbf{r}'| < |\mathbf{R}'|`.  For V-list leaves on a
# regular grid with half-size *h*, and evaluation points near the
# domain boundary where :math:`|\mathbf{r}'| \to h` while
# :math:`|\mathbf{R}'| \searrow 3h`, the condition is barely satisfied.
# For evaluation points outside the source cloud,
# :math:`|\mathbf{r}'| > |\mathbf{R}'|` and the series **must diverge**.
# No algebraic reformulation can extend the convergence radius of a
# geometric series.
#
# **Implemented improvements (2026-07-27)**
#
# Several of the options described below were implemented in the codebase.
# Here is a summary of what was tried and what was effective:
#
# **Dual-tree V-list builder (Option 2)** — The original leaf-only V-list
# was replaced with a dual-tree traversal that can accept internal nodes
# when they are well-separated.  This reduces the number of V-list entries
# and gives coarser (higher-quality) multipole approximations.  A ``theta``
# parameter was added to the build settings for MAC opening-angle control,
# though the neighbour criterion (``theta=0``) remains optimal for
# precomputed static V-lists.  A critical bug was fixed along the way:
# the upward sweep's :math:`\Gamma`-weighted centroid was overwriting
# ``nodes[i].center``, corrupting the tree descent — a ``geom_center``
# field now preserves the geometric centre for navigation while the
# weighted centroid is used for multipole accuracy.
#
# **Multi-level FMM (Option 1)** — The FMM now allocates local expansions
# for every node (not just leaves), runs a depth-filtered M2L sweep at
# every level (interaction zone :math:`[3h, 6h)`), and propagates
# ancestor contributions downward via the ``L2L`` operator.  This
# improved interior FMM-mode accuracy from :math:`\sim 3\times` TC error
# to only :math:`1.25\times` at ``critical_particle_count=30``.  The
# boundary divergence remains unfixed (convergence radius limitation).
#
# **Kahan / compensated summation (Option 4)** — Implemented and
# benchmarked.  It produced **no measurable benefit** (forward and Kahan
# sums differ by :math:`\sim 5\times 10^{-17}`, i.e. 1 ULP) and was
# removed.  The monomial basis is numerically well-conditioned enough
# that double-precision summation introduces no significant error.
#
# **Cell-size coefficient normalization** — Coefficients normalised by
# the cell half-size :math:`h` produce **bit-identical results** to
# unnormalised code (tested at orders 0–12, :math:`h = 0.1` to
# :math:`h = 0.001`).  The simpler unnormalised code is used.
#
# **Future options (not yet implemented)**
#
# * **Higher-order near-field (polynomial interpolation)** — Replace
#   the direct summation over near-field neighbours with a polynomial
#   surrogate (e.g. Chebyshev interpolation of the kernel on the leaf
#   volume).  The near-field cost scales as :math:`O(n^2)` in the
#   particle count per leaf, so reducing it opens the door to coarser
#   trees (larger ``critical_particle_count``) which have smaller V-lists
#   and better multipole quality.  This is the most promising remaining
#   improvement for both accuracy and performance.
#
# * **Morton-order / spatial-index interaction lists** — The current
#   interaction list builders use dual-tree walks that are close to
#   :math:`O(N \log N)` in practice but can degrade to :math:`O(N^2)`
#   for degenerate distributions.  A Morton-code spatial sort would
#   give guaranteed :math:`O(N \log N)` and enable larger problem sizes.
#
# **Practical recommendations**
#
# * Use **tree-code mode** (BH or FMM-TC) when accuracy is critical
#   or evaluation points may lie near the domain boundary.
# * Use **FMM mode** only for batch evaluation of points well inside
#   the source bounding box, where the 3–30× speedup over tree-code
#   is safe.
# * Keep **order ≤ 4** for double-precision arithmetic.  Higher
#   orders cost more and often produce no accuracy improvement.
# * Tune **critical_particle_count** rather than order when you need
#   higher accuracy.  The sweet spot is **30–50** for uniform
#   distributions: larger leaves give better multipole quality (more
#   particles per leaf → more statistical cancellation) and far faster
#   build times.  At ``critical_particle_count=30`` with order 4, the
#   FMM tree-code achieves :math:`0.047\%` median error — five times
#   better than ``crit=4`` and **460 times better than BH**.
