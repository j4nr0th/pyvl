r"""Adding a Wake Model
===================

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

    wm = results.wake_models[i].as_polydata()
    sg = sim_geo.polydata_at_time(0.0)
    circulation = results.circulations[i, :]
    sg.cell_data["Circulation"] = circulation
    sg.set_active_scalars("Circulation")
    wm.cell_data["Circulation"] = results.wake_models[i].circulation.flatten()
    wm.set_active_scalars("Circulation")

    plotter.add_mesh(mesh.glyph(factor=0.01))
    plotter.add_mesh(sg, label="Geometry")
    plotter.add_mesh(wm, label="Wake")

    plotter.show(interactive=False)
