r"""Example 1: Single Vortex
========================

.. currentmodule:: pyvl

This example shows the concept at the core of any panel code: the simple vortex panel.
This is a closed loop vortex with constant circulation.
"""  # noqa: D205, D400

import numpy as np
import pyvista as pv
import pyvl

pv.set_plot_theme("document")
pv.set_jupyter_backend("html")

# %%
#
# Geometry Setup
# --------------
#
# Geometry of a flat plate is trivial to set up.

geo = pyvl.Geometry.from_polydata(
    label="panel",
    reference_frame=pyvl.ReferenceFrame(),
    pd=pv.PolyData.from_regular_faces(
        points=np.array(((-1, -1, 0), (+1, -1, 0), (+1, +1, 0), (-1, +1, 0)), np.double),
        faces=np.array(((0, 1, 2, 3),), np.intp),
    ),
)

plt = pv.Plotter(off_screen=True)
plt.add_mesh(geo.as_polydata())
plt.show(interactive=False)
plt.close()
del plt

sim_geo = pyvl.SimulationGeometry.from_geometries(geo)

# %%
#
# Prepare the Settings
# --------------------
#
# After the :class:`SimulationGeometry` is prepared, the simulation settings must be
# configured. This is done via the :class::`SolverSettings` object. This contains other
# sub-objects, which themselves contain settings related to different aspects of the
# solver.
#


# %%
#
# Last which will be discussed here is the :class:`ModelSettings`. This class contains
# settings related to the settings made by the solver when it comes to the models of the
# flow and phyisics.

# Specify the minimum distance before vortex has no more effect.
model_settings = pyvl.ModelSettings(vortex_limit=1e-6)

# %%
#
# These can now be combined togethere into the :class:`SolverSettings` object.
v_inf = 1.0
rho_inf = 1.0
settings = pyvl.SolverSettings(flow_velocity=(0, 0, v_inf), model_settings=model_settings)

# %%
#
# Running the Solver
# ------------------
#
# The solver can now be run by calling :func:`run_solver` and passing the
# :class:`SimulationGeometry`, :class:`SolverSettings`, and :class:`OutputSettings`.


results = pyvl.run_solver(sim_geo, settings, times=[0], initial_time=0, n_threads=1)

# %%
#
# Post-Processing
# ---------------
#
# Now that the results have been computed, post-processing can be done to obtain some
# more useful results. In this example, this is done by creating a :mod:`pyvista` mesh,
# then computing velocity at each point in the mesh, the computing streamlines.
#
# What is quit clearly shown from this streamline plot, is that the flow is being
# deflected away from the panel's center. There still is a single streamline which
# manages to pass through, as velocity is exactly zero only at the control point,
# which the numerical integrator does not pick up on.
#

mesh = pv.Plane(
    center=(0, 0, 0),
    direction=(0, 1, 0),
    i_resolution=101,
    j_resolution=101,
    i_size=5,
    j_size=5,
)

velocities = [
    pyvl.postprocess.compute_velocities(state, mesh.points) for state in results
]

for i, state in enumerate(results):
    plotter = pv.Plotter(off_screen=True)

    mesh.point_data["Velocity"] = np.nan_to_num(velocities[i])
    mesh.set_active_vectors("Velocity")
    sl = mesh.streamlines(
        vectors="Velocity",
        pointa=(mesh.points[:, 0].min(), 0, mesh.points[:, 2].min()),
        pointb=(mesh.points[:, 0].max(), 0, mesh.points[:, 2].min()),
    )

    sg = state.geometry.polydata_at_time(state.time).extract_all_edges()
    sg.cell_data["Circulation"] = state.circulation
    plotter.add_mesh(sg, label="Geometry", color="black")
    plotter.add_mesh(sl)

    plotter.view_xz()
    plotter.show(interactive=False)
    plotter.close()
    del plotter

# %%
#
# Another interesting quantity is the pressure. The pressure distribution is computed
# on the same plane. I can be seen that there is a pressure increase at the panel's
# center. This is simply the result of the flow coming to a stop, which for the
# incompressible flow here means an increase in pressure up to
# :math:`\frac{1}{2} \rho {v_\infty}^2`. Further away the pressure drop decreases.

pressure_fields = [
    pyvl.postprocess.compute_dynamic_pressure(state, mesh.points, density=rho_inf)
    for state in results
]

for field, state in zip(pressure_fields, results):
    max_pressure = 1 / 2 * rho_inf * v_inf**2
    sg = sim_geo.polydata_edges_at_time(state.time)
    mesh.point_data["Pressure"] = field / max_pressure
    contours = mesh.contour(isosurfaces=31)

    plotter = pv.Plotter(off_screen=True)

    plotter.add_mesh(mesh, label="Geometry", scalars="Pressure")
    plotter.add_mesh(contours, color="red", scalars=None)

    plotter.view_xz()
    plotter.show(interactive=False)
    plotter.close()
    del plotter
