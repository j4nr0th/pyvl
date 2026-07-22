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
from pyvl.cvl import BarnesHutTree

# %%
#
# Setup — Clustered Sources
# -------------------------
#
# **100 000** vortex particles in 125 tight clusters inside
# :math:`[-0.4, 0.4]^3`.  A second set of **100 000** uniform sources
# is used for the performance benchmarks.

rng = np.random.default_rng(42)

N_PER_CLUSTER = 800
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
src_u = rng_u.uniform(-0.6, 0.6, (100_000, 3))
val_u = rng_u.uniform(-1.0, 1.0, (100_000, 3))
print(f"Uniform:   {src_u.shape[0]} sources")


def direct_induction(points, src_pos, src_val):
    """Exact induction: sum_i Γ_i / |r_i - p|²."""
    out = np.zeros_like(points)
    for i in range(src_pos.shape[0]):
        delta = points - src_pos[i]
        r2 = np.sum(delta**2, axis=-1, keepdims=True)
        out += src_val[i] / np.maximum(r2, 1e-300)
    return out


# %%
# Eval grids
# ----------
# Far-field: 40×40, near-field: 60×60, zoom: 60×60.

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

# Exact field on all 100k sources — done once
t0 = time.perf_counter()
v_exact_far = direct_induction(pts_far, src_c, val_c)
v_exact_near = direct_induction(pts_near, src_c, val_c)
t_ex = time.perf_counter() - t0
print(
    f"Exact far ({pts_far.shape[0]} pts) + near ({pts_near.shape[0]} pts): {t_ex:.1f} s"
)

# %%
# Build Trees for Orders 1–10
# ----------------------------
ORDERS = list(range(1, 11))
trees = {}
for order in ORDERS:
    t = BarnesHutTree.build(
        src_c,
        val_c,
        order=order,
        critical_particle_count=BarnesHutTree.min_sources_for_order(order),
        work_order=order + 4,
        n_threads=6,
    )
    trees[order] = t
    v = t.eval(pts_far, theta=0.0)
    diff = np.linalg.norm(v - v_exact_far)
    norm = np.linalg.norm(v_exact_far)
    l2_err = diff / max(norm, 1e-300)
    print(
        f"  order={order:2d}:  nodes={t.n_nodes:6d}  depth={t.max_depth}  "
        f"L2 err={l2_err:.2e}"
    )

# %%
# Near-Field Error Panel
# -----------------------
err_log_by_order = {}
for order in ORDERS:
    v = trees[order].eval(pts_near, theta=0.0)
    err = np.minimum(
        np.linalg.norm(v - v_exact_near, axis=-1)
        / np.maximum(np.linalg.norm(v_exact_near, axis=-1), 1e-300),
        1e3,
    )
    err_log_by_order[order] = np.log10(np.maximum(err, 1e-15)).reshape(Xn.shape)

ERR_VMIN, ERR_VMAX = -6, 1
FIXED_LEVELS = np.linspace(ERR_VMIN, ERR_VMAX, 22)
# -------------
fig, axes = plt.subplots(2, 5, figsize=(20, 7))
fig.suptitle("Near-field relative error — fixed colour scale")
for idx, order in enumerate(ORDERS):
    ax = axes.ravel()[idx]
    ax.contourf(
        Xn,
        Yn,
        err_log_by_order[order],
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

# %%
# Zoom Around Origin Cluster
# ---------------------------
R_ZOOM = 0.15
N_ZOOM = 60
xs_z = np.linspace(-R_ZOOM, R_ZOOM, N_ZOOM)
ys_z = np.linspace(-R_ZOOM, R_ZOOM, N_ZOOM)
Xz, Yz = np.meshgrid(xs_z, ys_z, indexing="ij")
pts_z = np.stack([Xz.ravel(), Yz.ravel(), np.zeros(N_ZOOM * N_ZOOM)], axis=-1)

# Only origin-cluster sources
om = (
    (np.abs(src_c[:, 0]) < R_ZOOM)
    & (np.abs(src_c[:, 1]) < R_ZOOM)
    & (np.abs(src_c[:, 2]) < R_ZOOM)
)
src_zo = src_c[om]
val_zo = val_c[om]
print(f"Origin-cluster sources: {src_zo.shape[0]}")
v_ex_z = direct_induction(pts_z, src_zo, val_zo)
v_ex_z_mag = np.log10(
    np.maximum(np.linalg.norm(v_ex_z, axis=-1).reshape(Xz.shape), 1e-30)
)

zoom_mag = {}
for order in [2, 4]:
    v = trees[order].eval(pts_z, theta=0.0)
    zoom_mag[order] = np.log10(
        np.maximum(np.linalg.norm(v, axis=-1).reshape(Xz.shape), 1e-30)
    )

ZMV = np.linspace(v_ex_z_mag.min(), v_ex_z_mag.max(), 22)
ZEV = np.linspace(-3, 1, 22)
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle("Zoom around one cluster (origin)", fontsize=13)
# Row 0 — field magnitude
axes[0, 0].contourf(Xz, Yz, v_ex_z_mag, levels=ZMV, cmap="viridis")
axes[0, 0].set_title("Exact (cluster only)", fontsize=10)
axes[0, 0].set_aspect("equal")
axes[0, 0].scatter(src_zo[:, 0], src_zo[:, 1], c="white", s=2, alpha=0.3, linewidths=0)
for col, order in enumerate([2, 4]):
    c = axes[0, col + 1].contourf(Xz, Yz, zoom_mag[order], levels=ZMV, cmap="viridis")
    axes[0, col + 1].set_title(f"BH order={order}", fontsize=10)
    axes[0, col + 1].set_aspect("equal")
fig.colorbar(c, ax=axes[0, :], location="bottom", pad=0.02, aspect=40, shrink=0.8)
# Row 1 — relative error at various settings
for col, (order, th, lbl) in enumerate(
    [
        (2, 0.0, "order 2, $\\theta$=0"),
        (4, 0.0, "order 4, $\\theta$=0"),
        (4, 0.5, "order 4, $\\theta$=0.5"),
    ]
):
    v = trees[order].eval(pts_z, theta=th)
    e_log = np.log10(
        np.maximum(
            np.minimum(
                np.linalg.norm(v - v_ex_z, axis=-1)
                / np.maximum(np.linalg.norm(v_ex_z, axis=-1), 1e-300),
                1e3,
            ),
            1e-15,
        )
    ).reshape(Xz.shape)
    c = axes[1, col].contourf(Xz, Yz, e_log, levels=ZEV, cmap="inferno")
    axes[1, col].set_title(f"Rel. error\n{lbl}", fontsize=10)
    axes[1, col].set_aspect("equal")
fig.colorbar(c, ax=axes[1, :], location="bottom", pad=0.02, aspect=40, shrink=0.8)
for a in axes.ravel():
    a.set_xlabel("$x$")
    a.set_ylabel("$y$")
fig.subplots_adjust(hspace=0.35, wspace=0.35)
plt.show()

# %%
# Accuracy vs Multipole Order
# ---------------------------
l2_err_by_order = []
for order in ORDERS:
    v = trees[order].eval(pts_far, theta=0.0)
    diff = np.sum(np.linalg.norm(v - v_exact_far, axis=-1) ** 2)
    norm = np.sum(np.linalg.norm(v_exact_far, axis=-1) ** 2)
    l2_err_by_order.append(np.sqrt(diff / max(norm, 1e-300)))
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.semilogy(ORDERS, l2_err_by_order, "o-", lw=2)
ax.set(
    xlabel="Order $p$",
    ylabel="Far-field L2 relative error",
    title="Accuracy vs Order (100k clustered)",
)
ax.set_xticks(ORDERS)
ax.grid(True, which="both", alpha=0.3)
plt.show()
for o, r in zip(ORDERS, l2_err_by_order):
    print(
        f"  order={o:2d}:  L2 err={r:.2e}  (crossover N > "
        f"{BarnesHutTree.min_sources_for_order(o)})"
    )

# %%
# Effect of Work Order
# --------------------
t_ref = BarnesHutTree.build(
    src_c,
    val_c,
    order=4,
    critical_particle_count=BarnesHutTree.min_sources_for_order(4),
    work_order=8,
    n_threads=6,
)
t_low = BarnesHutTree.build(
    src_c,
    val_c,
    order=4,
    critical_particle_count=BarnesHutTree.min_sources_for_order(4),
    work_order=4,
    n_threads=6,
)
v_ref = t_ref.eval(pts_far, theta=0.0)
v_low = t_low.eval(pts_far, theta=0.0)
e_ref = np.minimum(
    np.linalg.norm(v_ref - v_exact_far, axis=-1)
    / np.maximum(np.linalg.norm(v_exact_far, axis=-1), 1e-300),
    1e3,
)
e_low = np.minimum(
    np.linalg.norm(v_low - v_exact_far, axis=-1)
    / np.maximum(np.linalg.norm(v_exact_far, axis=-1), 1e-300),
    1e3,
)

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
levs = np.linspace(-6, 1, 22)
for ax_idx, (err, title) in enumerate(
    zip([e_low, e_ref], ["Rel. error $w=4$ (uses order=4)", "Rel. error $w=8$"])
):
    c = axes[ax_idx].contourf(
        Xf,
        Yf,
        np.log10(np.maximum(err, 1e-15)).reshape(Xf.shape),
        levels=levs,
        cmap="inferno",
    )
    axes[ax_idx].set_title(title, fontsize=10)
    axes[ax_idx].set_aspect("equal")
fig.colorbar(c, ax=axes[:2], location="bottom", pad=0.08, aspect=40, shrink=0.8)

mask = (e_ref > 1e-300) & np.isfinite(e_low) & np.isfinite(e_ref) & (e_ref < 1e3)
r = np.full_like(e_ref, np.nan)
if np.any(mask):
    r[mask] = np.minimum(e_low[mask] / e_ref[mask], 1e6)
    c2 = axes[2].contourf(
        Xf,
        Yf,
        np.log10(np.maximum(r, 1e-15)).reshape(Xf.shape),
        levels=np.linspace(-1, 3, 17),
        cmap="RdBu_r",
    )
    axes[2].set_aspect("equal")
    fig.colorbar(c2, ax=axes[2], location="bottom", pad=0.08, aspect=20, shrink=0.6)
    rm = r[mask]
    print(
        f"Work-order: median ratio = {np.nanmedian(rm):.2f}x, max = {np.nanmax(rm):.2f}x "
        f" ({len(rm)} pts)"
    )
else:
    axes[2].text(0.5, 0.5, "identical errors", transform=axes[2].transAxes, ha="center")
    print("Work-order: both trees have same error (identical topologies)")

# %%
# Critical Particle Count
# ------------------------
N_cc = [2, 3, 4, 6, 8, 16, 32]
n_nodes_cc = []
for cc in N_cc:
    t = BarnesHutTree.build(
        src_c, val_c, order=4, critical_particle_count=cc, n_threads=6
    )
    n_nodes_cc.append(t.n_nodes)
    print(f"  critical={cc:3d}:  nodes={t.n_nodes:6d}  depth={t.max_depth}")
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(N_cc, n_nodes_cc, "o-", lw=2)
ax.set(
    xlabel="Critical count",
    ylabel="Tree nodes",
    title="Tree size vs critical count (order=4, 100k)",
)
ax.grid(True, alpha=0.3)
plt.show()

# %%
# Cost Model
# ----------
print("=" * 60)
for p in range(1, 11):
    print(
        f"  order={p:2d}:  eval={BarnesHutTree.multipole_eval_cost(p):5d} FLOP  "
        f"→ beats direct for N > {BarnesHutTree.min_sources_for_order(p):3d}"
    )
N_cm = np.logspace(0.5, 5, 60)
fig, ax = plt.subplots(figsize=(8, 5))
for p in [1, 2, 4, 6, 10]:
    ax.axhline(
        BarnesHutTree.multipole_eval_cost(p),
        ls="--" if p > 1 else "-",
        alpha=0.7,
        label=f"order {p}",
    )
ax.loglog(
    N_cm,
    [BarnesHutTree.direct_sum_cost(int(n)) for n in N_cm],
    "k-",
    lw=2,
    label="direct (15N)",
)
ax.set(xlabel="N", ylabel="FLOP", title="Multipole eval vs direct sum")
ax.legend()
ax.grid(True, which="both", alpha=0.3)
plt.show()

# %%
# Performance: Build Time
# -----------------------
N_bld = [1000, 5000, 10000, 50000, 100000]
order_bld = [1, 2, 4, 6]
n_rep = 3
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
                n_threads=6,
                critical_particle_count=BarnesHutTree.min_sources_for_order(order),
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
N_src = N_src[N_src <= 100_000]
n_rep_src = 3
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
        n_threads=6,
        critical_particle_count=BarnesHutTree.min_sources_for_order(4),
    )
    t0 = time.perf_counter()
    for _ in range(n_rep_src):
        tr.eval(eval_targets, theta=0.0, n_threads=6)
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
# Smaller ``theta`` means the BH tree must descend deeper before accepting
# a cell's multipole, increasing eval time.  We measure wall time for 2000
# targets at several ``theta`` values on the 100k clustered tree (order 4).

n_rep_theta = 10
n_targets_theta = 2000
targets_theta = np.random.default_rng(42).uniform(-3, 3, (n_targets_theta, 3))
theta_times = []

tt4 = trees[4]  # order-4 tree on 100k clustered
for th in theta_values:
    t0 = time.perf_counter()
    for _ in range(n_rep_theta):
        tt4.eval(targets_theta, theta=th, n_threads=6)
    t_eval = (time.perf_counter() - t0) / n_rep_theta * 1000
    theta_times.append(t_eval)
    print(f"  theta={th:.4f}:  eval={t_eval:.3f} ms")

fig, ax = plt.subplots(figsize=(8, 5))
# Opening-angle values (theta > 0) on a log scale
pos_idx = [i for i, th in enumerate(theta_values) if th > 0]
ax.semilogx(
    [theta_values[i] for i in pos_idx],
    [theta_times[i] for i in pos_idx],
    "o-",
    lw=2,
    label="opening-angle criterion",
)
# Neighbour-criterion point (theta=0, not on log scale)
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
