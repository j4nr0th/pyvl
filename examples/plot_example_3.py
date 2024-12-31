r"""Finding a Pressure Distribution
===============================

.. currentmodule:: pyvl

Often, one of the main results of interest, besides the forces and the velocity
field itself, is the pressure distribution over the geometry. This too can be
obtained by post-processing.
"""  # noqa: D205, D400

import meshio as mio
import numpy as np
import pyvista as pv
import pyvl
from pyvl import examples

pv.set_plot_theme("document")
pv.set_jupyter_backend("html")
pv.global_theme.show_edges = False

# %%
#
# Simulation Setup
# ----------------
#
# This time, the geometry used won't be a flat plate, but instead a straight
# uniform wing with the NACA2412 airfoil profile.

geo = pyvl.Geometry.from_meshio(
    label="wing",
    reference_frame=pyvl.ReferenceFrame(),
    mesh=mio.read(examples.example_file_name("wing1.obj")),
)

sim_geo = pyvl.SimulationGeometry(geo)

alpha = np.radians(10)  # 10 degrees
v_inf = 10  # 10 m/s
flow_conditions = pyvl.FlowConditionsUniform(
    v_inf * np.cos(alpha), 0, v_inf * np.sin(alpha)
)
model_settings = pyvl.ModelSettings(vortex_limit=1e-6)

# Plot it just to show what it looks like
sim_geo.polydata_at_time(0.0).plot(interactive=False)

# %%
#
# The case will be run with and without the wake model, to show the difference between
# the two cases.

time_settings = pyvl.TimeSettings(1, 1)  # does not really matter for now

settings = pyvl.SolverSettings(flow_conditions, model_settings, time_settings)

# %%
#
# Running the Solver
# ------------------
#
# With the settings (an no wake model), the solver can now be run


results = pyvl.run_solver(sim_geo, settings, None, None)

# %%
#
# Post-Processing for Pressure
# ----------------------------
#
# While post-processing, only dynamic pressure will be computed. If you
# wish to deal with absolute pressure, the all you need to do is to add
# the free-stream pressure. However, this is unnecessary if you are
# interested in pressure force or pressure coefficient.
#

pressures = pyvl.postprocess.compute_surface_dynamic_pressure(results)

plotter = pv.Plotter()

sg = sim_geo.polydata_at_time(0.0)
sg.cell_data["Pressure"] = pressures[0]
sg.set_active_scalars("Pressure")

plotter.add_mesh(sg, label="Geometry")

plotter.add_axes()
plotter.show(interactive=False)

# %%
#
# Dynamic pressure can also be displayed in different positions. For example, let us
# plot the pressure around the airfoil at the middle of the wing and at the tip using
# :mod:`matplotlib`.

from matplotlib import pyplot as plt  # noqa: E402

nx = 51
nz = 51
y0 = 0.0
y1 = 4.75
plane1 = pv.Plane(
    center=(0.5, y0, 0),
    direction=(0, 1, 0),
    i_size=3,
    j_size=3,
    i_resolution=nx - 1,
    j_resolution=nz - 1,
)
assert isinstance(plane1, pv.PolyData)
plane2 = pv.Plane(
    center=(0.5, y1, 0),
    direction=(0, 1, 0),
    i_size=3,
    j_size=3,
    i_resolution=nx - 1,
    j_resolution=nz - 1,
)
assert isinstance(plane2, pv.PolyData)

pressures1 = pyvl.postprocess.compute_dynamic_pressure_variable(
    results, (plane1.points,)
)[0]
pressures2 = pyvl.postprocess.compute_dynamic_pressure_variable(
    results, (plane2.points,)
)[0]

max_p = np.max((np.abs(pressures1), np.abs(pressures2)))

plt.figure()

cplt1 = plt.tricontourf(
    plane1.points[:, 0],
    plane1.points[:, 2],
    pressures1,
    vmin=0,
    vmax=+max_p,
    cmap="magma",
)
plt.colorbar(cplt1)

plt.gca().set(
    title=f"Pressure Distribution at $y = {y0:g}$",
    aspect="equal",
    xlabel="$x$",
    ylabel="$z$",
)
plt.show(block=False)
plt.figure()

cplt2 = plt.tricontourf(
    plane2.points[:, 0],
    plane2.points[:, 2],
    pressures2,
    vmin=0,
    vmax=+max_p,
    cmap="magma",
)
plt.colorbar(cplt2)

plt.gca().set(
    title=f"Pressure Distribution at $y = {y1:g}$",
    aspect="equal",
    xlabel="$x$",
    ylabel="$z$",
)
plt.show(block=False)


# %%
#
# Adding a Wake
# -------------
#
# To now add the effect of the wake into the simulation, all that is needed is to add a
# wake model and re-run the solver. For geometry such as this, namely with a sharp, closed
# trailing edge, the :class:`SimulationGeometry` has the
# :meth:`SimulationGeometry.te_normal_criterion` method, which identifies all edges with
# two adjacent surfaces with unit normals with a dot product less than the specified
# criterion.
#
# As a word of caution, if you are too lenient with it, the wake will be shed from
# everywhere.


time_settings = pyvl.TimeSettings(12, 0.05)  # this now matters

settings = pyvl.SolverSettings(flow_conditions, model_settings, time_settings)

te_lines = sim_geo.te_normal_criterion(-0.5)  # -0.5 feels nice in my bones
ands, asur = sim_geo.line_adjecency_information(te_lines)


wake_model = pyvl.WakeModelLineExplicitUnsteady(
    ands,
    te_lines,
    asur,
    settings.model_settings.vortex_limit,
    -settings.time_settings.dt,
    10,
)

# %%
#
# Limiting Solver CPU Usage
# -------------------------
#
# Solver has to be run again, for multiple iterations. Since this will be a bit more
# computationally demanding than other examples so far, it is as good time as any
# to show how to control the solver's resources.
#
# The solver will use OpenMP to parallelize calculations of the induction matrix and
# use both :func:`scipy.linalg.lu_factor` and :func:`scipy.linalg.lu_solve` to invert
# the system. The thread usage of OpenMP can be limited by setting the value of the
# environment variable ``OMP_THREAD_NUM``, while controlling :mod:`scipy` depends on
# what exactly is used for it.
#
# The simplest and most straight-forward way to do this is using the
# ```threadpoolctl`` module<https://pypi.org/project/threadpoolctl/>`_.

from threadpoolctl import threadpool_limits  # noqa: E402

# For this case, use 4 threads
with threadpool_limits(limits=4):
    results = pyvl.run_solver(sim_geo, settings, wake_model, None)


# %%
#
# Show the Results Again
# ----------------------
#
# After running the post-processor again, the difference can be seen. Note that only
# the results of the last iteration are shown. Note again that ``threadpoolctl`` is
# once again used to limit the number of threads used for computing the induction
# inside the :func:`postprocess.compute_surface_dynamic_pressure` function.

with threadpool_limits(limits=4):
    pressures = pyvl.postprocess.compute_surface_dynamic_pressure(results)

plotter = pv.Plotter()

sg = sim_geo.polydata_at_time(settings.time_settings.output_times[-1])
sg.cell_data["Pressure"] = pressures[-1]
sg.set_active_scalars("Pressure")

wm = wake_model.as_polydata()
wm.set_active_scalars(None)

plotter.add_mesh(sg, label="Geometry")
plotter.add_mesh(wm, label="Wake")

plotter.add_axes()
plotter.show(interactive=False)
# %%
#
# We can now again plot the pressure at the two different sections of the wing.

pressures1 = pyvl.postprocess.compute_dynamic_pressure_variable(
    results, [plane1.points] * len(settings.time_settings.output_times)
)[-1]
pressures2 = pyvl.postprocess.compute_dynamic_pressure_variable(
    results, [plane2.points] * len(settings.time_settings.output_times)
)[-1]

max_p = np.max((np.abs(pressures1), np.abs(pressures2)))

plt.figure()

cplt1 = plt.tricontourf(
    plane1.points[:, 0],
    plane1.points[:, 2],
    pressures1,
    vmin=0,
    vmax=+max_p,
    cmap="magma",
)
plt.colorbar(cplt1)

plt.gca().set(
    title=f"Pressure Distribution at $y = {y0:g}$",
    aspect="equal",
    xlabel="$x$",
    ylabel="$z$",
)
plt.show(block=False)
plt.figure()

cplt2 = plt.tricontourf(
    plane2.points[:, 0],
    plane2.points[:, 2],
    pressures2,
    vmin=0,
    vmax=+max_p,
    cmap="magma",
)
plt.colorbar(cplt2)

plt.gca().set(
    title=f"Pressure Distribution at $y = {y1:g}$",
    aspect="equal",
    xlabel="$x$",
    ylabel="$z$",
)
plt.show(block=False)
