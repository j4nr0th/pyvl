r"""Example 2: Adding a Wake Model
==============================

.. currentmodule:: pyvl

This example shows how a wake model can be used. A wake model can help
capture some of the viscous effects which are not able to be captured
by potential flow model on its own.
"""  # noqa: D205, D400

import numpy as np
import pyvista as pv
import pyvl

pv.set_plot_theme("document")
pv.set_jupyter_backend("html")
pv.global_theme.show_edges = True

# %%
#
# Simulation Setup
# ----------------
#
# For this example, the initial simulation setup is identical to the one used in
# :ref:`the first example <sphx_glr_auto_examples_plot_example_1.py>`, so it won't be
# commented on much.

plate = pv.Plane()
assert isinstance(plate, pv.PolyData)

geo = pyvl.Geometry.from_polydata(
    label="plate",
    reference_frame=pyvl.ReferenceFrame(),
    pd=plate,
)

sim_geo = pyvl.SimulationGeometry(geo)

alpha = np.radians(10)  # 10 degrees
v_inf = 10  # 10 m/s
flow_conditions = pyvl.FlowConditionsUniform(
    v_inf * np.cos(alpha), 0, v_inf * np.sin(alpha)
)
model_settings = pyvl.ModelSettings(vortex_limit=1e-6)

# %%
#
# For this example, an unsteady wake model will be used, so time settings
# are set to run the simulation for 20 time steps with 0.005 between each.

time_settings = pyvl.TimeSettings(20, 0.005)

settings = pyvl.SolverSettings(flow_conditions, model_settings, time_settings)

# %%
#
# Creating a Wake Model
# ---------------------
#
# In this case, the :class:`WakeModelLineExplicitUnsteady` will be used.
# This wake model requires the information about the wake shedding elements.

shedding_lines: list[int] = list()
pos = sim_geo.positions_at_time(0.0)
for i_line, ln in enumerate(sim_geo.mesh.line_data):
    if pos[ln[0], 0] == +0.5 and pos[ln[1], 0] == +0.5:
        shedding_lines.append(i_line)

nods, surfs = sim_geo.line_adjecency_information(shedding_lines)

wake_model = pyvl.WakeModelLineExplicitUnsteady(
    nods,
    np.array(shedding_lines),
    surfs,
    1e-6,
    0.0,
    15,
)

# %%
#
# Running the Solver
# ------------------
#
# The solver can now be run by calling :func:`run_solver` and passing the
# :class:`SimulationGeometry`, :class:`SolverSettings`, and
# :class:`WakeModelLineExplicitUnsteady`.


results = pyvl.run_solver(sim_geo, settings, wake_model, None)

# %%
#
# Post-Processing
# ---------------
#
# Post-processing works the exact same way, but now the presence of the wake can be
# observed.
#

mesh = pv.RectilinearGrid(
    np.linspace(-1, 1, 11),
    np.linspace(-1, 1, 11),
    np.linspace(-1, 1, 11),
)

velocities = pyvl.postprocess.compute_velocities(results, mesh.points)

for i, t in enumerate(results.settings.time_settings.output_times):
    plotter = pv.Plotter()

    mesh.point_data["Velocity"] = velocities[i, :, :]
    mesh.set_active_vectors("Velocity")
    wake_mod = results.wake_models[i]
    assert isinstance(wake_mod, pyvl.WakeModelLineExplicitUnsteady)
    wm = wake_mod.as_polydata()
    sg = sim_geo.polydata_at_time(0.0)
    circulation = results.circulations[i, :]
    sg.cell_data["Circulation"] = circulation
    sg.set_active_scalars("Circulation")
    wm.cell_data["Circulation"] = wake_mod.circulation.flatten()
    wm.set_active_scalars("Circulation")

    plotter.add_mesh(mesh.glyph(factor=0.01))
    plotter.add_mesh(sg, label="Geometry")
    plotter.add_mesh(wm, label="Wake")

    plotter.show(interactive=False)

# %%
#
# Forces
# ------
#
# Unlike as in :ref:`the first example <sphx_glr_auto_examples_plot_example_1.py>`, where
# there were not total forces due to all circulation being bound on the surface of the
# mesh, there is now a non-zero resultant force on the geometry due to circulation being
# shed into the wake.
#

forces = pyvl.postprocess.circulatory_forces(results)

for field in forces:
    sg = sim_geo.polydata_edges_at_time(0.0)
    sg.cell_data["Forces"] = field
    print(f"Total force: {np.sum(field, axis=0)} Newtons")
    plotter = pv.Plotter()
    plotter.add_mesh(sg.glyph(factor=1))
    plotter.add_mesh(sg, label="Geometry", color="Red")

    plotter.show(interactive=False)

# %%
#
# Notes About the Forces
# ----------------------
#
# Due to the nature of the solver, the flat plate representation of a lifting surface
# is not very accurate. This is because that is not what the solver is intended to do.
# In contrast to this, `AVL <http://web.mit.edu/drela/Public/web/avl/>`_ does exactly
# that by simulating wings as horseshoe elements, which have the lifting line at the
# quarter chord line and the control point at the three quarter point.
#
# Instead, the :mod:`pyvl` solver works by having the control point at the center and
# with closed vortex rings. As such, it work best with full airfoil profiles.
