r"""Barnes-Hut Tree: Accuracy, Tuning, and Performance
====================================================

.. currentmodule:: pyvl

This example explores the :class:`cvl.BarnesHutTree`: how its accuracy depends
on the multipole order, the work order, the critical particle count, and how
to choose these parameters using the built-in cost model.
"""  # noqa: D205, D400

import time

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from pyvl.cvl import BarnesHutTree

# %%
#
# Setup
# -----
#
# We create **clustered sources** — several tight clusters spread across a
# cube.  This mimics the particle distribution you get from a panel-method
# wake: groups of vortex particles near trailing edges, separated by large
# empty regions.
#
# Each cluster contains 12 sources inside a small sphere of radius
# :math:`R_c = 0.1`.  The cluster centres are on a :math:`3 \times 3 \times 3`
# grid spaced by 0.6, for a total of **324** sources.

rng = np.random.default_rng(42)

N_PER_CLUSTER = 40
R_CLUSTER = 0.06
# 125 cluster centres packed into a tight volume so the tree has to subdivide.
# With 125 × 60 = 7500 sources in a ~0.8³ box, the root octants each get
# ~940 particles → well above the subdivide threshold (critical_particle_count=4
# with order=4 gives threshold = 4×5³ = 500).
GRID_3D = np.mgrid[-0.4:0.5:0.2, -0.4:0.5:0.2, -0.4:0.5:0.2].reshape(3, -1).T
N_CLUSTERS = GRID_3D.shape[0]

coords_list = []
values_list = []
for centre in GRID_3D:
    raw = rng.uniform(-1, 1, (N_PER_CLUSTER, 3))
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    pts = centre + raw / norms * (
        rng.uniform(0, 1, (N_PER_CLUSTER, 1)) ** (1 / 3) * R_CLUSTER
    )
    vals = rng.uniform(-1, 1, (N_PER_CLUSTER, 3))
    coords_list.append(pts)
    values_list.append(vals)

sources_coords = np.concatenate(coords_list, axis=0)
sources_values = np.concatenate(values_list, axis=0)
N_SOURCES = sources_coords.shape[0]
print(f"Sources: {N_SOURCES} particles in {N_CLUSTERS} clusters")


# %%
# Exact Direct Sum
# ----------------
#
# The reference solution is the direct :math:`O(N^2)` :math:`1/r^2` induction
# used throughout pyvl.


def direct_induction(
    points: npt.NDArray[np.double],
    src_pos: npt.NDArray[np.double],
    src_val: npt.NDArray[np.double],
) -> npt.NDArray[np.double]:
    """Exact induction: sum_i Γ_i / |r_i - p|²."""
    out = np.zeros_like(points)
    for i in range(src_pos.shape[0]):
        delta = points - src_pos[i]
        r2 = np.sum(delta**2, axis=-1, keepdims=True)
        out += src_val[i] / np.maximum(r2, 1e-300)
    return out


# %%
# Build Reference Tree
# --------------------
#
# We build a default tree (order 4, critical_particle_count=4, work_order=4)
# and a low-work-order tree (work_order=0 → uses order) for comparison.

tree_default = BarnesHutTree.build(
    sources_coords, sources_values, order=4, work_order=4, n_threads=4
)
tree_low_wo = BarnesHutTree.build(
    sources_coords, sources_values, order=4, work_order=0, n_threads=4
)

print(
    f"Default tree: {tree_default.n_nodes} nodes "
    f"(internal={tree_default.n_internal}, mp_leaves={tree_default.n_multipole_leaves}, "
    f"particle_leaves={tree_default.n_particle_leaves}), depth={tree_default.max_depth}"
)
print(f"Low work-order tree: {tree_low_wo.n_nodes} nodes")

# %%
# Accuracy: Far-Field Slice
# --------------------------
#
# We evaluate the exact field and the BH tree on a coarse **far-field**
# :math:`z=0` slice covering :math:`x, y \in [-5, 5]`.  Far from the sources
# the multipole compression is very accurate even at moderate order.

# Far-field grid — coarse
N_FAR = 60
xs_far = np.linspace(-5, 5, N_FAR)
ys_far = np.linspace(-5, 5, N_FAR)
Xf, Yf = np.meshgrid(xs_far, ys_far, indexing="ij")
pts_far = np.stack([Xf.ravel(), Yf.ravel(), np.zeros(N_FAR * N_FAR)], axis=-1)

v_exact_far = direct_induction(pts_far, sources_coords, sources_values)
v_bh_far = tree_default.eval(pts_far, theta=0.0)

# L2 relative error per point
err_far = np.linalg.norm(v_bh_far - v_exact_far, axis=-1) / np.maximum(
    np.linalg.norm(v_exact_far, axis=-1), 1e-300
)
err_far_2d = np.log10(np.maximum(err_far, 1e-15)).reshape(Xf.shape)
mag_far = np.log10(np.maximum(np.linalg.norm(v_exact_far, axis=-1), 1e-30)).reshape(
    Xf.shape
)

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
c0 = axes[0].contourf(Xf, Yf, mag_far, levels=20, cmap="viridis")
axes[0].set(
    title="Exact field (log magnitude)", aspect="equal", xlabel="$x$", ylabel="$y$"
)
fig.colorbar(c0, ax=axes[0])

c1 = axes[1].contourf(Xf, Yf, err_far_2d, levels=20, cmap="inferno")
axes[1].set(
    title="BH rel. error $\\log_{10}$", aspect="equal", xlabel="$x$", ylabel="$y$"
)
fig.colorbar(c1, ax=axes[1])

# Mark source cluster centres
src_centres_xy = GRID_3D[:, :2]
axes[1].scatter(src_centres_xy[:, 0], src_centres_xy[:, 1], c="cyan", s=8, alpha=0.6)

# Only show meaningful errors outside immediate source region
# Use 3D distance from origin — sources sit inside [-0.4, 0.4]³
mask_far_pts = np.linalg.norm(pts_far, axis=-1) > 1.0
err_far_masked = np.where(mask_far_pts, err_far, np.nan).reshape(Xf.shape)
c2 = axes[2].contourf(
    Xf, Yf, np.log10(np.maximum(err_far_masked, 1e-15)), levels=20, cmap="inferno"
)
axes[2].set(title="Error outside $r=0.5$", aspect="equal", xlabel="$x$", ylabel="$y$")
fig.colorbar(c2, ax=axes[2])
fig.tight_layout()
plt.show()

# %%
# Accuracy: Near-Field Zoom
# --------------------------
#
# Now zoom into the source region :math:`x, y \in [-1, 1]` with a **finer**
# grid.  Near the sources the BH tree must fall back to direct particle
# summation; the error there reflects the multipole acceptance criterion and
# the residual truncation of compressed leaves.

N_NEAR = 120
xs_near = np.linspace(-1, 1, N_NEAR)
ys_near = np.linspace(-1, 1, N_NEAR)
Xn, Yn = np.meshgrid(xs_near, ys_near, indexing="ij")
pts_near = np.stack([Xn.ravel(), Yn.ravel(), np.zeros(N_NEAR * N_NEAR)], axis=-1)

v_exact_near = direct_induction(pts_near, sources_coords, sources_values)
v_bh_near = tree_default.eval(pts_near, theta=0.0)

err_near = np.linalg.norm(v_bh_near - v_exact_near, axis=-1) / np.maximum(
    np.linalg.norm(v_exact_near, axis=-1), 1e-300
)
mag_near = np.log10(np.maximum(np.linalg.norm(v_exact_near, axis=-1), 1e-30)).reshape(
    Xn.shape
)

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
c0 = axes[0].contourf(Xn, Yn, mag_near, levels=20, cmap="viridis")
axes[0].set(
    title="Exact field (log mag, zoom)", aspect="equal", xlabel="$x$", ylabel="$y$"
)
fig.colorbar(c0, ax=axes[0])

c1 = axes[1].contourf(
    Xn,
    Yn,
    np.log10(np.maximum(err_near, 1e-15)).reshape(Xn.shape),
    levels=20,
    cmap="inferno",
)
axes[1].set(title="BH rel. error", aspect="equal", xlabel="$x$", ylabel="$y$")
fig.colorbar(c1, ax=axes[1])

axes[2].contourf(
    Xn,
    Yn,
    np.log10(np.maximum(err_near, 1e-15)).reshape(Xn.shape),
    levels=np.linspace(-8, 0, 17),
    cmap="inferno",
)
axes[2].set(title="Error (fine levels)", aspect="equal", xlabel="$x$", ylabel="$y$")
fig.colorbar(c2 := axes[2].collections[0], ax=axes[2])
fig.tight_layout()
plt.show()

# %%
# Effect of Work Order
# --------------------
#
# The ``work_order`` parameter controls the binomial series truncation when
# multipoles are shifted from child centres to parent centres.  A low work
# order (= expansion order) introduces extra truncation error; setting it
# higher retains more denominator terms and improves fidelity.
#
# We compare the default tree (``work_order=4``) against the low-work-order
# tree (``work_order=0`` → internally uses order=4) on the far-field grid.

v_bh_low = tree_low_wo.eval(pts_far, theta=0.0)
err_low = np.linalg.norm(v_bh_low - v_exact_far, axis=-1) / np.maximum(
    np.linalg.norm(v_exact_far, axis=-1), 1e-300
)
err_def = np.linalg.norm(v_bh_far - v_exact_far, axis=-1) / np.maximum(
    np.linalg.norm(v_exact_far, axis=-1), 1e-300
)

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

c0 = axes[0].contourf(
    Xf,
    Yf,
    np.log10(np.maximum(err_low, 1e-15)).reshape(Xf.shape),
    levels=20,
    cmap="inferno",
)
axes[0].set(title="Rel. error $w=0$ (uses order)", aspect="equal")
fig.colorbar(c0, ax=axes[0])

c1 = axes[1].contourf(
    Xf,
    Yf,
    np.log10(np.maximum(err_def, 1e-15)).reshape(Xf.shape),
    levels=20,
    cmap="inferno",
)
axes[1].set(title="Rel. error $w=4$", aspect="equal")
fig.colorbar(c1, ax=axes[1])

# Error ratio
ratio = np.where(err_def > 1e-300, err_low / err_def, np.nan)
c2 = axes[2].contourf(
    Xf,
    Yf,
    np.log10(np.maximum(ratio, 1e-15)).reshape(Xf.shape),
    levels=np.linspace(-1, 3, 17),
    cmap="RdBu_r",
)
axes[2].set(title="$\\log_{10}$ (err$_{w=0}$ / err$_{w=4}$)", aspect="equal")
fig.colorbar(c2, ax=axes[2])
fig.tight_layout()
plt.show()

wo_max = err_low.max() / max(err_def.max(), 1e-300)
wo_med = np.nanmedian(err_low / np.maximum(err_def, 1e-300))
print(
    f"Work-order effect: median error ratio w=0 / w=4 = {wo_med:.2f}x, max ="
    f" {wo_max:.2f}x"
)

# %%
# Accuracy vs Multipole Order
# ---------------------------
#
# Higher multipole orders capture more moments of the source distribution.
# We sweep orders 1..6 and measure the far-field L2 relative error.  As a
# reference we compute the cost-model crossover.

orders = [1, 2, 3, 4, 5, 6]
trees_by_order = {}
err_by_order = []

for order in orders:
    t = BarnesHutTree.build(
        sources_coords, sources_values, order=order, work_order=order, n_threads=4
    )
    trees_by_order[order] = t
    v = t.eval(pts_far, theta=0.0)
    err = np.linalg.norm(v - v_exact_far, axis=-1) / np.maximum(
        np.linalg.norm(v_exact_far, axis=-1), 1e-300
    )
    # Mask out points inside the source region where error is dominated by
    # the MAC (particle-leaf direct sum) rather than truncation
    mask = np.linalg.norm(pts_far, axis=-1) > 1.5
    err_by_order.append(np.sqrt(np.nanmean(err[mask] ** 2)))

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.semilogy(orders, err_by_order, "o-", lw=2, label="far-field rel. error")
ax.set(
    xlabel="Multipole order $p$",
    ylabel="L2 relative error",
    title="Accuracy vs Order (far-field, $r > 0.5$)",
    xticks=orders,
)
ax.grid(True, which="both", alpha=0.3)
ax.legend()
plt.show()

for order, err in zip(orders, err_by_order):
    src_min = BarnesHutTree.min_sources_for_order(order)
    print(f"  order={order}: L2 error={err:.2e}  (crossover N > {src_min})")

# %%
# Effect of Critical Particle Count
# ----------------------------------
#
# The ``critical_particle_count`` controls when a leaf is subdivided.  A low
# value produces a deeper, more refined tree that is more accurate near the
# sources but takes longer to build and consumes more memory.  A high value
# produces a shallower tree with fewer multipole leaves.
#
# We fix order=4 and vary the critical count from 4 to 64.

critical_values = [2, 3, 4, 6, 8, 16]
trees_by_crit = {}
n_nodes_by_crit = []
n_mp_leaves_by_crit = []

for cc in critical_values:
    t = BarnesHutTree.build(
        sources_coords, sources_values, order=4, critical_particle_count=cc, n_threads=4
    )
    trees_by_crit[cc] = t
    n_nodes_by_crit.append(t.n_nodes)
    n_mp_leaves_by_crit.append(t.n_multipole_leaves)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(critical_values, n_nodes_by_crit, "o-", lw=2)
axes[0].set(
    xlabel="Critical particle count",
    ylabel="Tree nodes",
    title="Tree size vs critical count",
)
axes[0].grid(True, alpha=0.3)

axes[1].plot(critical_values, n_mp_leaves_by_crit, "s-", lw=2)
axes[1].set(
    xlabel="Critical particle count",
    ylabel="Multipole leaves",
    title="Multipole leaves vs critical count",
)
axes[1].grid(True, alpha=0.3)
fig.tight_layout()
plt.show()

for cc, nn in zip(critical_values, n_nodes_by_crit):
    print(f"  critical={cc:3d}:  nodes={nn:5d}")

# %%
# Critical particle count also affects accuracy: a coarser tree means larger
# leaves with more particles, so the multipole approximation inside each leaf
# covers a wider spatial extent.

err_by_crit = []
for cc in critical_values:
    t = trees_by_crit[cc]
    v = t.eval(pts_far, theta=0.0)
    err = np.linalg.norm(v - v_exact_far, axis=-1) / np.maximum(
        np.linalg.norm(v_exact_far, axis=-1), 1e-300
    )
    mask = np.linalg.norm(pts_far, axis=-1) > 1.5
    err_by_crit.append(np.sqrt(np.nanmean(err[mask] ** 2)))

fig, ax = plt.subplots(figsize=(7, 4))
ax.semilogy(critical_values, err_by_crit, "o-", lw=2)
ax.set(
    xlabel="Critical particle count",
    ylabel="L2 relative error",
    title="Far-field accuracy vs critical particle count (order=4)",
)
ax.grid(True, alpha=0.3)
plt.show()

# %%
# Cost Model
# ----------
#
# The :class:`cvl.BarnesHutTree` exposes four static methods that help you
# choose parameters without running any simulation:

print("=" * 60)
print("Cost-model reference (static methods)")
print("=" * 60)
for p in range(1, 9):
    mp_cost = BarnesHutTree.multipole_eval_cost(p)
    min_n = BarnesHutTree.min_sources_for_order(p)
    print(
        f"  order={p}:  eval_cost={mp_cost:5d} FLOP  ⇒  "
        f"beats direct sum for N > {min_n:2d}"
    )

print()
n_test = [10, 50, 100, 500, 1000, 5000]
print(f"{'N':>6}  {'direct FLOP':>12}  {'crossover order':>16}")
for n in n_test:
    direct = BarnesHutTree.direct_sum_cost(n)
    co = BarnesHutTree.crossover_order(n)
    print(f"{n:>6}  {direct:>12}  {co:>16}")

# %%
# Visual comparison: cost of a single multipole eval vs direct sum

N_range = np.logspace(0.5, 4, 50)
orders_to_plot = [1, 2, 4, 6]

fig, ax = plt.subplots(figsize=(8, 5))
for p in orders_to_plot:
    ax.axhline(
        BarnesHutTree.multipole_eval_cost(p),
        ls="--" if p > 1 else "-",
        alpha=0.7,
        label=f"order {p} eval",
    )

direct_costs = [BarnesHutTree.direct_sum_cost(int(n)) for n in N_range]
ax.loglog(N_range, direct_costs, "k-", lw=2, label="direct sum (15 N)")
ax.set(
    xlabel="Number of sources N",
    ylabel="FLOP count",
    title="Multipole eval vs direct sum — crossover points",
)
ax.legend()
ax.grid(True, which="both", alpha=0.3)
plt.show()

# %%
# Performance: Build Time
# -----------------------
#
# How does the build time scale with the number of sources and the multipole
# order?  We measure wall time for several (N, order) combinations.

N_values = [1000, 2500, 5000]
order_values = [1, 2, 3, 4]
n_repeat = 3

build_times: dict[tuple[int, int], float] = {}

for n in N_values:
    # Generate sources for this size
    rng_n = np.random.default_rng(123 + n)
    coord_n = rng_n.uniform(-1, 1, (n, 3))
    val_n = rng_n.uniform(-1, 1, (n, 3))
    for order in order_values:
        t0 = time.perf_counter()
        for _ in range(n_repeat):
            t = BarnesHutTree.build(coord_n, val_n, order=order, n_threads=4)
            _ = t.n_nodes  # force tree to stay alive
        t1 = time.perf_counter()
        build_times[(n, order)] = (t1 - t0) / n_repeat * 1000  # ms

fig, ax = plt.subplots(figsize=(8, 5))
for n in N_values:
    times = [build_times[(n, order)] for order in order_values]
    ax.plot(order_values, times, "o-", lw=2, label=f"N={n}")
ax.set(
    xlabel="Order",
    ylabel="Build time (ms)",
    title="Build time vs order and N",
    xticks=order_values,
)
ax.legend()
ax.grid(True, alpha=0.3)
plt.show()

# %%
# Performance: Eval Time
# ----------------------
#
# Eval time scales with the number of target points and the tree depth.
# We fix N=2000 and measure eval time for different target counts.

N_BIG = 5000
# Use same clustered sources for a deeper tree
coord_big, val_big = sources_coords, sources_values
tree_big = BarnesHutTree.build(coord_big, val_big, order=4, n_threads=4)
print(f"Eval tree: {tree_big.n_nodes} nodes, depth={tree_big.max_depth}")

target_counts = [100, 500, 1000, 2000, 5000]
n_eval_repeat = 5
eval_times = []

for nt in target_counts:
    rng_t = np.random.default_rng(789)
    targets = rng_t.uniform(-5, 5, (nt, 3))
    t0 = time.perf_counter()
    for _ in range(n_eval_repeat):
        _ = tree_big.eval(targets, theta=0.0, n_threads=4)
    t1 = time.perf_counter()
    eval_times.append((t1 - t0) / n_eval_repeat * 1000)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(target_counts, eval_times, "o-", lw=2)
axes[0].set(
    xlabel="Number of targets", ylabel="Eval time (ms)", title="Eval time vs target count"
)
axes[0].grid(True, alpha=0.3)

axes[1].plot(
    target_counts, [t / n * 1e6 for t, n in zip(eval_times, target_counts)], "s-", lw=2
)
axes[1].set(xlabel="Number of targets", ylabel="ns per target", title="Per-target cost")
axes[1].grid(True, alpha=0.3)
fig.tight_layout()
plt.show()

# %%
# Performance: Effect of Thread Count
# ------------------------------------
#
# The BH tree build uses OpenMP for the downward pass and multipole leaf
# construction.  We measure build time at N=2000 for 1–8 threads.

thread_counts = [1, 2, 4, 8]
n_thread_repeat = 3
thread_times = []

rng_th = np.random.default_rng(101)
# Use 2000 sources from a tighter cluster set for the thread benchmark
N_THR_SRC = 2000
coord_th = rng_th.uniform(-0.3, 0.3, (N_THR_SRC, 3))
val_th = rng_th.uniform(-1, 1, (N_THR_SRC, 3))

for nth in thread_counts:
    t0 = time.perf_counter()
    for _ in range(n_thread_repeat):
        _ = BarnesHutTree.build(coord_th, val_th, order=4, n_threads=nth)
    t1 = time.perf_counter()
    thread_times.append((t1 - t0) / n_thread_repeat * 1000)

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(thread_counts, thread_times, "o-", lw=2)
ax.plot(
    thread_counts,
    [thread_times[0] / tc for tc in thread_counts],
    "k--",
    lw=1,
    label="ideal scaling",
)
ax.set(
    xlabel="Threads",
    ylabel="Build time (ms)",
    title=f"Build-time scaling (N={N_BIG}, order=4)",
    xticks=thread_counts,
)
ax.legend()
ax.grid(True, alpha=0.3)
plt.show()

for nth, tm in zip(thread_counts, thread_times):
    speedup = thread_times[0] / tm
    print(f"  threads={nth}:  build={tm:.1f} ms  (speedup={speedup:.2f}x)")

# %%
# Summary of Guidelines
# ---------------------
#
# 1. **Order selection.**  Use at least order 2; order 4 gives excellent
#    far-field accuracy for most wake applications.  The cost-model static
#    methods can help decide: where
#    :meth:`~cvl.BarnesHutTree.crossover_order` crosses the direct-sum cost
#    curve.
#
# 2. **Work order.**  Always set ``work_order >= order``.  Setting it higher
#    (e.g. ``work_order = order + 2``) costs no extra memory for the tree
#    itself (only the build scratch) and substantially improves shift
#    accuracy.  The default of ``work_order = order`` is a safe minimum.
#
# 3. **Critical particle count.**  The default (4) produces a well-balanced
#    tree.  Increase it to 8–16 if you need a shallower tree (less memory,
#    faster build); decrease it only if you observe accuracy issues near the
#    source region.
#
# 4. **Thread count.**  Build times scale well up to 4–8 threads.  Eval
#    times also benefit from OpenMP parallelism for large target batches.
#
# 5. **Cost model.**  Use :meth:`cvl.BarnesHutTree.multipole_eval_cost` and
#    :meth:`cvl.BarnesHutTree.min_sources_for_order` at runtime to make
#    informed parameter choices.
