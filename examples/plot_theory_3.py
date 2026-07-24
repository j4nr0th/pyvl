r"""Barnes-Hut Tree: Accuracy, Tuning, and Performance
====================================================

.. currentmodule:: pyvl

Accuracy, tuning, and performance of the :class:`cvl.BarnesHutTree`.
The built-in cost model determines when the tree is cheaper than
direct :math:`O(N^2)` summation.  Tuning parameters are the multipole
order, critical particle count, MAC opening angle :math:`\theta`,
and centroid subdivision threshold :math:`\alpha`.
"""  # noqa: D205, D400

import os
import time

import matplotlib.pyplot as plt
import numpy as np
from pyvl.cvl import BarnesHutTree

N_THREADS = min(6, os.cpu_count() or 1)
print(f"Using up to {N_THREADS} threads")

# %%
#
# Setup
# -----
#
# Two source sets are used:
#
# * **Clustered** \\--- 5 000 vortex particles in 125 tight clusters
#   (:math:`R = 0.05`) on a regular grid in :math:`[-0.4, 0.4]^3`.
#   Used for accuracy studies.
# * **Uniform** \\--- 5 000 particles uniformly sampled in
#   :math:`[-0.6, 0.6]^3`.  Used for performance benchmarks.

rng = np.random.default_rng(42)

N_PER_CLUSTER = 40
R_CLUSTER = 0.05
GRID_3D = np.mgrid[-0.4:0.5:0.2, -0.4:0.5:0.2, -0.4:0.5:0.2].reshape(3, -1).T
coords_list, values_list = [], []
for centre in GRID_3D:
    raw = rng.uniform(-1, 1, (N_PER_CLUSTER, 3))
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    pts = centre + raw / norms * (
        rng.uniform(0, 1, (N_PER_CLUSTER, 1)) ** (1 / 3) * R_CLUSTER
    )
    vals = rng.uniform(-1, 1, (N_PER_CLUSTER, 3))
    coords_list.append(pts)
    values_list.append(vals)
src_c = np.concatenate(coords_list)
val_c = np.concatenate(values_list)
print(f"Clustered: {src_c.shape[0]} sources in {GRID_3D.shape[0]} clusters")

rng_u = np.random.default_rng(20250722)
N_PARTICLES = 5_000
src_u = rng_u.uniform(-0.6, 0.6, (N_PARTICLES, 3))
val_u = rng_u.uniform(-1.0, 1.0, (N_PARTICLES, 3))
print(f"Uniform:   {src_u.shape[0]} sources")


def direct_induction(points, src_pos, src_val):
    """Exact induction: sum_i Gamma_i / |r_i - p|^2."""
    out = np.zeros_like(points)
    for i in range(src_pos.shape[0]):
        delta = points - src_pos[i]
        r2 = np.sum(delta**2, axis=-1, keepdims=True)
        out += src_val[i] / np.maximum(r2, 1e-300)
    return out


# %%
#
# Eval grids
# ----------
#
# Far-field grid: 40 x 40 on :math:`[-5, 5]^2` (z=0).  Near-field:
# 60 x 60 on :math:`[-1, 1]^2` (z=0).  Exact reference fields are
# computed once.

N_FAR = 40
xs_far = np.linspace(-5, 5, N_FAR)
ys_far = np.linspace(-5, 5, N_FAR)
Xf, Yf = np.meshgrid(xs_far, ys_far, indexing="ij")
pts_far = np.stack([Xf.ravel(), Yf.ravel(), np.zeros(N_FAR * N_FAR)], axis=-1)

N_NEAR = 60
xs_near = np.linspace(-1, 1, N_NEAR)
ys_near = np.linspace(-1, 1, N_NEAR)
Xn, Yn = np.meshgrid(xs_near, ys_near, indexing="ij")
pts_near = np.stack([Xn.ravel(), Yn.ravel(), np.zeros(N_NEAR * N_NEAR)], axis=-1)

t0 = time.perf_counter()
v_exact_far = direct_induction(pts_far, src_c, val_c)
v_exact_near = direct_induction(pts_near, src_c, val_c)
t_ex = time.perf_counter() - t0
print(
    f"Exact: far ({pts_far.shape[0]} pts) + near ({pts_near.shape[0]} pts) "
    f"in {t_ex:.2f} s"
)

# %%
#
# Cost Model
# ----------
#
# ``multipole_eval_cost(p)`` returns the FLOP count for one multipole
# evaluation at order *p*.  ``direct_sum_cost(N)`` returns FLOP for a
# direct sum over N sources (15 FLOP per source).  A tree beats direct
# summation when ``15 * N > multipole_eval_cost(p)``.  The crossover
# point is given by ``min_sources_for_order(p)``.

print()
print("Cost model summary:")
print(f"{'order':>5s}  {'eval FLOP':>9s}  {'min N >':>7s}")
print("-" * 25)
for p in range(1, 11):
    print(
        f"{p:>5d}  {BarnesHutTree.multipole_eval_cost(p):>9d}  "
        f"{BarnesHutTree.min_sources_for_order(p):>7d}"
    )

N_cm = np.logspace(0.5, 5, 60)
fig, ax = plt.subplots(figsize=(8, 5))
for p in [1, 2, 4, 6, 10]:
    cost = BarnesHutTree.multipole_eval_cost(p)
    ax.axhline(
        cost, ls="--" if p > 1 else "-", alpha=0.7, label=f"order {p}  ({cost} FLOP)"
    )
direct_cost = [BarnesHutTree.direct_sum_cost(int(n)) for n in N_cm]
ax.loglog(N_cm, direct_cost, "k-", lw=2, label="direct sum (15 N)")
ax.set(xlabel="N sources", ylabel="FLOP", title="Multipole eval cost vs direct sum")
ax.legend(fontsize=8)
ax.grid(True, which="both", alpha=0.3)
plt.show()

# %%
#
# Accuracy vs Multipole Order
# ---------------------------
#
# Trees with orders 1–8 are built on the 5k clustered set.
# ``work_order = order + 4`` gives the binomial shift series extra terms,
# improving accuracy when the shift distance is large relative to the
# source bounding sphere (shallow trees, large critical counts).
# The subdivision threshold uses ``min_sources_for_order(p)`` so each
# leaf contains enough particles to justify the multipole.

ORDERS = list(range(1, 9))
trees = {}
for order in ORDERS:
    t = BarnesHutTree.build(
        src_c,
        val_c,
        order=order,
        critical_particle_count=BarnesHutTree.min_sources_for_order(order),
        work_order=order + 4,
        alpha_centroid=0.0,
        n_threads=N_THREADS,
    )
    trees[order] = t

# Far-field evaluation (neighbour criterion, theta=0)
v_far = {o: trees[o].eval(pts_far, theta=0.0) for o in ORDERS}

l2_err = {}
max_err = {}
for order in ORDERS:
    diff = v_far[order] - v_exact_far
    pt_err = np.minimum(
        np.linalg.norm(diff, axis=-1)
        / np.maximum(np.linalg.norm(v_exact_far, axis=-1), 1e-300),
        1e3,
    )
    l2_err[order] = np.sqrt(np.nansum(pt_err**2) / np.sum(np.isfinite(pt_err)))
    max_err[order] = np.nanmax(pt_err)

print()
print(
    f"{'order':>5s}  {'nodes':>6s}  {'depth':>5s}  "
    f"{'mp_leaf':>6s}  {'pt_leaf':>6s}  "
    f"{'L2_err':>8s}  {'max_err':>8s}"
)
print("-" * 58)
for order in ORDERS:
    t = trees[order]
    print(
        f"{order:>5d}  {t.n_nodes:>6d}  {t.max_depth:>5d}  "
        f"{t.n_multipole_leaves:>6d}  {t.n_particle_leaves:>6d}  "
        f"{l2_err[order]:>8.2e}  {max_err[order]:>8.2e}"
    )

# Near-field error contours
err_log = {}
for order in ORDERS:
    v = trees[order].eval(pts_near, theta=0.0)
    err = np.minimum(
        np.linalg.norm(v - v_exact_near, axis=-1)
        / np.maximum(np.linalg.norm(v_exact_near, axis=-1), 1e-300),
        1e3,
    )
    err_log[order] = np.log10(np.maximum(err, 1e-15)).reshape(Xn.shape)

ERR_VMIN, ERR_VMAX = -6, 1
FIXED_LEVELS = np.linspace(ERR_VMIN, ERR_VMAX, 22)

fig, axes = plt.subplots(2, 5, figsize=(20, 7))
fig.suptitle("Near-field relative error (fixed colour scale)")
for idx, order in enumerate(ORDERS):
    ax = axes.ravel()[idx]
    ax.contourf(
        Xn,
        Yn,
        err_log[order],
        levels=FIXED_LEVELS,
        cmap="inferno",
        vmin=ERR_VMIN,
        vmax=ERR_VMAX,
    )
    ax.set_title(f"Order {order}", fontsize=10)
    ax.set_aspect("equal")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
fig.subplots_adjust(right=0.92)
cax = fig.add_axes([0.93, 0.12, 0.015, 0.76])
fig.colorbar(axes.ravel()[0].collections[0], cax=cax, label="$\\log_{10}$ rel. error")
plt.show()

# Far-field L2 and max error vs order
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.semilogy(ORDERS, [l2_err[o] for o in ORDERS], "o-", lw=2, label="L2")
ax.semilogy(ORDERS, [max_err[o] for o in ORDERS], "s--", lw=1.5, alpha=0.6, label="max")
ax.set(
    xlabel="Order $p$",
    ylabel="Relative error",
    title="Far-field accuracy vs order (5k clustered)",
)
ax.set_xticks(ORDERS)
ax.legend()
ax.grid(True, which="both", alpha=0.3)
plt.show()

for o in ORDERS:
    print(
        f"  order={o:2d}:  L2={l2_err[o]:.2e}  max={max_err[o]:.2e}  "
        f"(crossover N > {BarnesHutTree.min_sources_for_order(o)})"
    )

# %%
#
# Effect of Work Order
# --------------------
#
# The ``work_order`` parameter controls the binomial series truncation
# in :func:`~pyvl.private_c.multipole.multipole_add_shift`, used during
# the upward sweep when shifting child multipoles to the parent centre.
# A higher work_order adds more binomial terms, improving accuracy when
# the shift distance is comparable to the source bounding sphere radius
# (shallow trees with large critical_particle_count).  The default
# ``work_order`` = ``None`` (uses the multipole order) is sufficient
# for deep trees with small cells.

t_def = BarnesHutTree.build(
    src_c,
    val_c,
    order=4,
    critical_particle_count=BarnesHutTree.min_sources_for_order(4),
    alpha_centroid=0.0,
    n_threads=N_THREADS,
)  # default: work_order=None
t_high = trees[4]  # from the sweep above: work_order=8

v_def = t_def.eval(pts_far, theta=0.0)
v_high = t_high.eval(pts_far, theta=0.0)

err_def = np.minimum(
    np.linalg.norm(v_def - v_exact_far, axis=-1)
    / np.maximum(np.linalg.norm(v_exact_far, axis=-1), 1e-300),
    1e3,
)
err_high = np.minimum(
    np.linalg.norm(v_high - v_exact_far, axis=-1)
    / np.maximum(np.linalg.norm(v_exact_far, axis=-1), 1e-300),
    1e3,
)

print()
print("Work order comparison (order=4):")
print(
    f"  work_order=None:  L2={np.sqrt(np.nanmean(err_def**2)):.2e}  "
    f"max={np.nanmax(err_def):.2e}"
)
print(
    f"  work_order=8:     L2={np.sqrt(np.nanmean(err_high**2)):.2e}  "
    f"max={np.nanmax(err_high):.2e}"
)
print(
    "  L2 ratio (None/8): "
    f"{np.sqrt(np.nanmean(err_def**2)) / np.sqrt(np.nanmean(err_high**2)):.2f}x"
)

fig, ax = plt.subplots(figsize=(5, 3.5))
ax.bar(
    ["None (order=4)", "work_order=8"],
    [np.sqrt(np.nanmean(err_def**2)), np.sqrt(np.nanmean(err_high**2))],
    color=["C0", "C1"],
    alpha=0.8,
)
ax.set(ylabel="L2 relative error", title="Work order effect (order=4)")
ax.grid(True, alpha=0.3, axis="y")
plt.show()

# %%
# Effect of Critical Particle Count
# ----------------------------------
#
# The ``critical_particle_count`` (default 4) controls how many sources a leaf
# must contain before it becomes a multipole leaf.  Smaller values create more
# multipole leaves (more compression), larger values keep more particle leaves
# (more accurate but slower).
#
# We sweep several values on the 5k clustered set (order 4). Small critical
# counts produce deep trees with few particle leaves (very fast eval at the
# cost of more multipole error).  Larger critical counts keep more particle
# leaves — eval becomes slower but mid-field accuracy improves because fewer
# sources are approximated by multipoles.

CRIT_VALUES = [2, 4, 8, 16, 32]
crit_labels = [str(v) for v in CRIT_VALUES]
crit_colors = ["C0", "C1", "C2", "C3", "C4"]

trees_crit = {}
build_ms_crit = {}

for cc in CRIT_VALUES:
    t0 = time.perf_counter()
    t = BarnesHutTree.build(
        src_c,
        val_c,
        order=4,
        critical_particle_count=cc,
        alpha_centroid=0.0,
        n_threads=N_THREADS,
    )
    bt = (time.perf_counter() - t0) * 1000
    trees_crit[cc] = t
    build_ms_crit[cc] = bt
    print(
        f"  critical={cc:3d}:  nodes={t.n_nodes:6d}  depth={t.max_depth}  "
        f"mp={t.n_multipole_leaves:5d}  particle={t.n_particle_leaves:5d}  "
        f"build={bt:.1f} ms"
    )

# Mid-field accuracy comparison
fig, axes = plt.subplots(1, 5, figsize=(22, 4.5))
fig.suptitle(
    "Effect of critical count on near-field accuracy (order 4, $\\theta=0$)",
    fontsize=13,
)

for idx, cc in enumerate(CRIT_VALUES):
    ax = axes[idx]
    t = trees_crit[cc]
    v = t.eval(pts_near, theta=0.0)
    err = np.minimum(
        np.linalg.norm(v - v_exact_near, axis=-1)
        / np.maximum(np.linalg.norm(v_exact_near, axis=-1), 1e-300),
        1e3,
    )
    err_log = np.log10(np.maximum(err, 1e-15)).reshape(Xn.shape)
    c = ax.contourf(
        Xn,
        Yn,
        err_log,
        levels=FIXED_LEVELS,
        cmap="inferno",
        vmin=ERR_VMIN,
        vmax=ERR_VMAX,
    )
    ax.set_title(f"critical={cc}", fontsize=10)
    ax.set_aspect("equal")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")

fig.subplots_adjust(right=0.92)
cax = fig.add_axes([0.93, 0.12, 0.015, 0.76])
fig.colorbar(c, cax=cax, label="$\\log_{10}$ rel. error")
plt.show()

# Eval time vs critical count
n_rep_crit_eval = 2
targets_crit = np.random.default_rng(77).uniform(-3, 3, (2000, 3))
theta_sweep_crit = [0.0, 0.3]
eval_ms_crit = {cc: [] for cc in CRIT_VALUES}

for cc in CRIT_VALUES:
    t = trees_crit[cc]
    for th in theta_sweep_crit:
        t0 = time.perf_counter()
        for _ in range(n_rep_crit_eval):
            t.eval(targets_crit, theta=th, n_threads=N_THREADS)
        tm = (time.perf_counter() - t0) / n_rep_crit_eval * 1000
        eval_ms_crit[cc].append(tm)

fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(theta_sweep_crit))
w = 0.12
for i, (cc, clr) in enumerate(zip(CRIT_VALUES, crit_colors)):
    ax.bar(
        x + i * w - (len(CRIT_VALUES) - 1) * w / 2,
        eval_ms_crit[cc],
        w,
        label=f"critical={cc}",
        color=clr,
        alpha=0.85,
    )
ax.set_xticks(x)
ax.set_xticklabels([f"$\\theta$={t}" for t in theta_sweep_crit])
ax.set(
    ylabel="Eval time (ms)",
    title="Eval time vs critical count (2000 targets, order 4)",
)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis="y")
plt.show()

# Far-field accuracy per critical count
crit_l2 = {}
crit_max = {}
for cc in CRIT_VALUES:
    v = trees_crit[cc].eval(pts_far, theta=0.0)
    err = np.minimum(
        np.linalg.norm(v - v_exact_far, axis=-1)
        / np.maximum(np.linalg.norm(v_exact_far, axis=-1), 1e-300),
        1e3,
    )
    crit_l2[cc] = np.sqrt(np.nansum(err**2) / np.sum(np.isfinite(err)))
    crit_max[cc] = np.nanmax(err)

print()
print("critical_particle_count impact summary:")
print(
    f"{'crit':>5s}  {'nodes':>6s}  {'depth':>5s}  "
    f"{'L2_err':>8s}  {'max_err':>8s}  "
    f"{'build(ms)':>8s}  {'eval@0':>8s}  {'eval@0.3':>9s}"
)
print("-" * 70)
for cc in CRIT_VALUES:
    t = trees_crit[cc]
    print(
        f"{cc:>5d}  {t.n_nodes:>6d}  {t.max_depth:>5d}  "
        f"{crit_l2[cc]:>8.2e}  {crit_max[cc]:>8.2e}  "
        f"{build_ms_crit[cc]:>8.1f}  "
        f"{eval_ms_crit[cc][0]:>8.2f}  "
        f"{eval_ms_crit[cc][1]:>9.2f}"
    )

# %%
# Performance: Build Time
# -----------------------
N_bld = [1000, 5000, 10000]
order_bld = [1, 2, 4, 6]
n_rep = 2
bt = {}
for n in N_bld:
    rn = np.random.default_rng(123 + n)
    cn = rn.uniform(-0.6, 0.6, (n, 3))
    vn = rn.uniform(-1, 1, (n, 3))
    for order in order_bld:
        t0 = time.perf_counter()
        for _ in range(n_rep):
            _ = BarnesHutTree.build(
                cn,
                vn,
                order=order,
                n_threads=N_THREADS,
                critical_particle_count=BarnesHutTree.min_sources_for_order(order),
                alpha_centroid=0.0,
            ).n_nodes
        bt[(n, order)] = (time.perf_counter() - t0) / n_rep * 1000
fig, ax = plt.subplots(figsize=(8, 5))
for n in N_bld:
    ax.plot(order_bld, [bt[(n, o)] for o in order_bld], "o-", lw=2, label=f"N={n}")
ax.set(
    xlabel="Order",
    ylabel="Build time (ms)",
    title="Build time vs order and N",
    xticks=order_bld,
)
ax.legend()
ax.grid(True, alpha=0.3)
plt.show()

# %%
# Performance: BH Eval vs Direct O(N) vs N Sources
# --------------------------------------------------
# Fix target count at 1000, sweep source count from 1k to 100k in logspace.
# To factor out implementation-language differences (BH is pure C, direct sum
# is Python + NumPy) we plot the **relative** wall time — each curve is
# normalised by its own value at N = 1000.
#
# Since the number of targets is fixed, the direct O(N²) cost is linear in N,
# while the BH tree is sublinear.

N_src = np.unique(np.logspace(3, 5, 8).astype(int))
N_src = N_src[N_src <= 5_000]
n_rep_src = 2
n_targets = 1000
eval_targets = np.random.default_rng(0).uniform(-3, 3, (n_targets, 3))
bh_ev_t, dir_ev_t = [], []
for n in N_src:
    rn = np.random.default_rng(123 + n)
    cn = rn.uniform(-0.6, 0.6, (n, 3))
    vn = rn.uniform(-1, 1, (n, 3))
    tr = BarnesHutTree.build(
        cn,
        vn,
        order=4,
        n_threads=N_THREADS,
        critical_particle_count=BarnesHutTree.min_sources_for_order(4),
        alpha_centroid=0.0,
    )
    t0 = time.perf_counter()
    for _ in range(n_rep_src):
        tr.eval(eval_targets, theta=0.0, n_threads=N_THREADS)
    bh_ev_t.append((time.perf_counter() - t0) / n_rep_src * 1000)
    t0 = time.perf_counter()
    for _ in range(n_rep_src):
        direct_induction(eval_targets, cn, vn)
    dir_ev_t.append((time.perf_counter() - t0) / n_rep_src * 1000)
    sp = dir_ev_t[-1] / max(bh_ev_t[-1], 1e-9)
    print(
        f"  N={n:6d}:  BH={bh_ev_t[-1]:.3f} ms  direct={dir_ev_t[-1]:.1f} ms  "
        f"speedup={sp:.0f}x"
    )

# Normalise to N=1000 for each curve
bh_rel = np.array(bh_ev_t) / bh_ev_t[0]
dir_rel = np.array(dir_ev_t) / dir_ev_t[0]

fig, ax = plt.subplots(figsize=(8, 5))
ax.loglog(N_src, bh_rel, "o-", lw=2, label="BH tree (order 4)")
ax.loglog(N_src, dir_rel, "s-", lw=2, label="Direct O(N²)")
ax.loglog(
    N_src, N_src / N_src[0], "k--", lw=1, alpha=0.5, label="$\\mathcal{O}(N)$ ideal"
)
ax.loglog(
    N_src,
    np.log(N_src) / np.log(N_src[0]),
    "k:",
    lw=1,
    alpha=0.5,
    label="$\\mathcal{O}(\\log(N))$ ideal",
)
ax.set(
    xlabel="Number of sources N",
    ylabel="Relative wall time (norm. to N=1k)",
    title=f"Scaling: BH tree vs direct O(N) ({n_targets} targets)",
)
ax.legend()
ax.grid(True, which="both", alpha=0.3)
plt.show()

# %%
# Thread Scaling
# --------------
tc = list(range(1, 7))
tt = []
for nth in tc:
    t0 = time.perf_counter()
    for _ in range(n_rep):
        _ = BarnesHutTree.build(
            src_u,
            val_u,
            order=4,
            n_threads=nth,
            critical_particle_count=BarnesHutTree.min_sources_for_order(4),
            alpha_centroid=0.0,
        ).n_nodes
    tt.append((time.perf_counter() - t0) / n_rep * 1000)
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(tc, tt, "o-", lw=2, label="measured")
ax.plot(tc, [tt[0] / c for c in tc], "k--", lw=1, label="ideal")
ax.set(
    xlabel="Threads",
    ylabel="Build time (ms)",
    title="Thread scaling (100k uniform, order=4)",
    xticks=tc,
)

ax.legend()
ax.grid(True, alpha=0.3)
plt.show()
for nth, tm in zip(tc, tt):
    print(f"  threads={nth}:  build={tm:.1f} ms  (speedup={tt[0] / tm:.2f}x)")

# %%
# Effect of the MAC Opening Angle :math:`\theta`
# ----------------------------------------------
#
# The ``theta`` parameter in :meth:`cvl.BarnesHutTree.eval` controls when a
# cell's multipole is accepted.  Values ``<= 0`` use the **neighbour criterion**
# — a cell is accepted when the target point lies outside its 3×3×3
# neighbourhood.  Positive values use the **opening-angle criterion** — a cell
# is accepted when :math:`\text{half\_size} / \text{distance} < \theta`.
#
# The default is ``theta=0.3``, which gives far-field accuracy similar to the
# neighbour criterion but 2–10× faster eval.  For mid-field targets (close to
# the source cloud), use smaller values such as ``theta=0.01`` for near-direct
# accuracy, at the cost of slower evaluation.
#
# We build an order-4 tree on the 100k clustered set and evaluate on the
# near-field grid with several ``theta`` values.  The colour scale is fixed
# across all panels.

# Reconstruct error for the neighbour-criterion case (theta=0)
v_0 = trees[4].eval(pts_near, theta=0.0)
err_0 = np.minimum(
    np.linalg.norm(v_0 - v_exact_near, axis=-1)
    / np.maximum(np.linalg.norm(v_exact_near, axis=-1), 1e-300),
    1e3,
)
err_0_log = np.log10(np.maximum(err_0, 1e-15)).reshape(Xn.shape)

theta_values = [0.0, 1e-9, 1e-3, 1e-2, 1e-1, 0.5]
theta_labels = ["0 (neighbour)"] + [f"{t:.2g}" for t in theta_values[1:]]

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle("Effect of opening angle $\\theta$ on near-field accuracy", fontsize=13)
for idx, (th, lbl) in enumerate(zip(theta_values, theta_labels)):
    ax = axes.ravel()[idx]
    if th <= 0:
        err_log = err_0_log
    else:
        v = trees[4].eval(pts_near, theta=th)
        err = np.minimum(
            np.linalg.norm(v - v_exact_near, axis=-1)
            / np.maximum(np.linalg.norm(v_exact_near, axis=-1), 1e-300),
            1e3,
        )
        err_log = np.log10(np.maximum(err, 1e-15)).reshape(Xn.shape)
    c = ax.contourf(
        Xn, Yn, err_log, levels=FIXED_LEVELS, cmap="inferno", vmin=ERR_VMIN, vmax=ERR_VMAX
    )
    ax.set_title(f"$\\theta = {lbl}$", fontsize=10)
    ax.set_aspect("equal")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
fig.subplots_adjust(right=0.92)
cax = fig.add_axes([0.93, 0.12, 0.015, 0.76])
fig.colorbar(c, cax=cax, label="$\\log_{10}$ rel. error")
plt.show()

# %%
# Evaluation Time vs Opening Angle
# ---------------------------------
#
# Smaller ``theta`` forces deeper descent, increasing eval time.
# Wall time is measured for 2000 targets on the order-4 tree.
# Error statistics (far-field) are computed for each theta value.

n_rep_theta = 3
n_targets_theta = 2000
targets_theta = np.random.default_rng(42).uniform(-3, 3, (n_targets_theta, 3))
theta_times = []
theta_l2 = []
theta_max = []

tt4 = trees[4]  # order-4 tree on 100k clustered
for th in theta_values:
    t0 = time.perf_counter()
    for _ in range(n_rep_theta):
        tt4.eval(targets_theta, theta=th, n_threads=N_THREADS)
    t_eval = (time.perf_counter() - t0) / n_rep_theta * 1000
    theta_times.append(t_eval)
    # Far-field error
    v_th = tt4.eval(pts_far, theta=th)
    err_th = np.minimum(
        np.linalg.norm(v_th - v_exact_far, axis=-1)
        / np.maximum(np.linalg.norm(v_exact_far, axis=-1), 1e-300),
        1e3,
    )
    theta_l2.append(np.sqrt(np.nansum(err_th**2) / np.sum(np.isfinite(err_th))))
    theta_max.append(np.nanmax(err_th))

print()
print(f"{'theta':>12s}  {'eval(ms)':>9s}  {'L2_err':>9s}  {'max_err':>9s}")
print("-" * 45)
for i, th in enumerate(theta_values):
    lbl = "0 (neighbour)" if th <= 0 else f"{th:.4g}"
    print(
        f"{lbl:>12s}  {theta_times[i]:>9.4f}  {theta_l2[i]:>9.2e}  {theta_max[i]:>9.2e}"
    )

fig, ax = plt.subplots(figsize=(8, 5))
pos_idx = [i for i, th in enumerate(theta_values) if th > 0]
ax.semilogx(
    [theta_values[i] for i in pos_idx],
    [theta_times[i] for i in pos_idx],
    "o-",
    lw=2,
    label="opening-angle criterion",
)
ax.axhline(
    theta_times[0],
    color="C3",
    ls="--",
    lw=1.5,
    label=f"neighbour criterion: {theta_times[0]:.3f} ms",
)
ax.set(
    xlabel="$\\theta$ (opening angle)",
    ylabel="Eval time (ms)",
    title=f"Eval time vs $\\theta$ ({n_targets_theta} targets, 100k sources, order 4)",
)
ax.legend()

# %%
# Effect of Centroid-Based Subdivision (:math:`\alpha`)
# -----------------------------------------------------
#
# The ``alpha_centroid`` parameter controls **centroid-based subdivision**:
# when a source is more than ``alpha_centroid * half_size`` from the cell's
# geometric centre, the cell is subdivided at build time.  This guarantees
# that sources in each leaf are tightly clustered around the centre, which
# improves multipole convergence at mid-field.
#
# The default is ``alpha_centroid = 0.5``.  Set to ``0.0`` to disable
# (legacy behaviour — subdivision driven only by source count).  Smaller
# values force tighter clusters (more subdivision), larger values relax
# the criterion.
#
# We compare three values on the 100k clustered set (order 4).  Because
# the clusters are only :math:`R=0.05` wide but spaced :math:`0.2` apart,
# many sources lie far from their containing cell's geometric centre,
# so a non-zero ``alpha_centroid`` triggers extra subdivision.

alpha_values = [0.0, 0.5, 0.3]
alpha_labels = ["0.0 (disabled)", "0.5 (default)", "0.3 (tighter)"]
alpha_colors = ["C3", "C0", "C2"]

trees_alpha = {}
build_ms_alpha = {}

for ac, lbl in zip(alpha_values, alpha_labels):
    t0 = time.perf_counter()
    t = BarnesHutTree.build(
        src_c,
        val_c,
        order=4,
        critical_particle_count=BarnesHutTree.min_sources_for_order(4),
        alpha_centroid=ac,
        n_threads=N_THREADS,
    )
    bt = (time.perf_counter() - t0) * 1000
    trees_alpha[ac] = t
    build_ms_alpha[ac] = bt
    print(
        f"  alpha={lbl:>16s}:  nodes={t.n_nodes:6d}  depth={t.max_depth}  "
        f"mp={t.n_multipole_leaves:5d}  particle={t.n_particle_leaves:5d}  "
        f"build={bt:.1f} ms"
    )

# %%
# Mid-field accuracy comparison
# ------------------------------
#
# Evaluate all three trees on the near-field grid with the neighbour
# criterion (``theta=0``, most demanding).  Tighter clustering (smaller
# ``alpha``) should improve accuracy inside the source cloud.

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(
    "Effect of $\\alpha$ on near-field accuracy (order 4, $\\theta=0$)", fontsize=13
)

for idx, (ac, lbl) in enumerate(zip(alpha_values, alpha_labels)):
    ax = axes[idx]
    t = trees_alpha[ac]
    v = t.eval(pts_near, theta=0.0)
    err = np.minimum(
        np.linalg.norm(v - v_exact_near, axis=-1)
        / np.maximum(np.linalg.norm(v_exact_near, axis=-1), 1e-300),
        1e3,
    )
    err_log = np.log10(np.maximum(err, 1e-15)).reshape(Xn.shape)
    c = ax.contourf(
        Xn, Yn, err_log, levels=FIXED_LEVELS, cmap="inferno", vmin=ERR_VMIN, vmax=ERR_VMAX
    )
    ax.set_title(f"$\\alpha = {lbl}$", fontsize=10)
    ax.set_aspect("equal")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")

fig.subplots_adjust(right=0.92)
cax = fig.add_axes([0.93, 0.12, 0.015, 0.76])
fig.colorbar(c, cax=cax, label="$\\log_{10}$ rel. error")
plt.show()

# %%
# Eval time vs :math:`\alpha`
# ----------------------------
#
# More subdivision → deeper tree → more nodes visited during eval.
# We measure wall time at three ``theta`` values on 2000 targets.

n_rep_alpha_eval = 2
targets_alpha = np.random.default_rng(99).uniform(-3, 3, (2000, 3))
theta_sweep = [0.0, 0.1, 0.3]
eval_ms_alpha = {ac: [] for ac in alpha_values}

for ac in alpha_values:
    t = trees_alpha[ac]
    for th in theta_sweep:
        t0 = time.perf_counter()
        for _ in range(n_rep_alpha_eval):
            t.eval(targets_alpha, theta=th, n_threads=N_THREADS)
        tm = (time.perf_counter() - t0) / n_rep_alpha_eval * 1000
        eval_ms_alpha[ac].append(tm)

fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(theta_sweep))
w = 0.25
for i, (ac, lbl, clr) in enumerate(zip(alpha_values, alpha_labels, alpha_colors)):
    ax.bar(
        x + i * w - w,
        eval_ms_alpha[ac],
        w,
        label=f"$\\alpha$={lbl}",
        color=clr,
        alpha=0.85,
    )
ax.set_xticks(x)
ax.set_xticklabels([f"$\\theta$={t}" for t in theta_sweep])
ax.set(
    ylabel="Eval time (ms)",
    title="Eval time vs $\\alpha$ (2000 targets, order 4)",
)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis="y")
plt.show()

# %%
# Summary table
# -------------
# Far-field accuracy per alpha value
alpha_l2 = {}
alpha_max = {}
for ac in alpha_values:
    v = trees_alpha[ac].eval(pts_far, theta=0.0)
    err = np.minimum(
        np.linalg.norm(v - v_exact_far, axis=-1)
        / np.maximum(np.linalg.norm(v_exact_far, axis=-1), 1e-300),
        1e3,
    )
    alpha_l2[ac] = np.sqrt(np.nansum(err**2) / np.sum(np.isfinite(err)))
    alpha_max[ac] = np.nanmax(err)

print()
print("alpha_centroid impact summary:")
print(
    f"{'alpha':>6s}  {'nodes':>6s}  {'depth':>5s}  "
    f"{'L2_err':>8s}  {'max_err':>8s}  "
    f"{'build(ms)':>8s}  {'eval@0':>8s}  {'eval@0.3':>9s}"
)
print("-" * 70)
for ac in alpha_values:
    t = trees_alpha[ac]
    print(
        f"{ac:>6.1f}  {t.n_nodes:>6d}  {t.max_depth:>5d}  "
        f"{alpha_l2[ac]:>8.2e}  {alpha_max[ac]:>8.2e}  "
        f"{build_ms_alpha[ac]:>8.1f}  "
        f"{eval_ms_alpha[ac][0]:>8.2f}  "
        f"{eval_ms_alpha[ac][2]:>9.2f}"
    )

# %%
# Self-Induction Check
# --------------------
#
# The BH tree must exclude each source's self-interaction (the field from
# a particle at its own position is undefined).  The tree evaluation at
# every source coordinate is compared against the direct sum that also
# excludes self-pairs.  Discrepancies are dominated by the multipole
# approximation, not by mishandled self-induction.

v_at_src = trees[4].eval(src_c, theta=0.0)  # order-4 tree

# Direct sum excluding self, vectorised
n_src = src_c.shape[0]
v_other_exact = np.zeros_like(src_c)
CHUNK = 500
for start in range(0, n_src, CHUNK):
    end = min(start + CHUNK, n_src)
    delta = src_c[None, start:end, :] - src_c[:, None]  # (n_src, chunk, 3)
    r2 = np.sum(delta**2, axis=-1)
    for local_j in range(end - start):
        r2[start + local_j, local_j] = np.inf  # exclude self
    inv = val_c[:, None, :] / np.maximum(r2[..., None], 1e-300)
    v_other_exact[start:end] += np.sum(inv, axis=0)

err_self = np.linalg.norm(v_at_src - v_other_exact, axis=-1) / np.maximum(
    np.linalg.norm(v_other_exact, axis=-1), 1e-300
)

print()
print("Self-induction check (order 4, neighbour criterion):")
print(f"  Median relative error: {np.median(err_self):.3e}")
print(f"  90th percentile:       {np.percentile(err_self, 90):.3e}")
print(f"  Max relative error:    {np.max(err_self):.3e}")

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.hist(np.log10(np.maximum(err_self, 1e-15)), bins=60, alpha=0.7, color="C0")
ax.axvline(
    np.log10(np.median(err_self)),
    color="C3",
    ls="--",
    label=f"median={np.median(err_self):.2e}",
)
ax.set(
    xlabel="$\\log_{10}$ relative error",
    ylabel="Number of sources",
    title="Self-induction: BH vs other-only exact",
)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.show()

print("  Errors from multipole approximation, not self-induction handling.")
