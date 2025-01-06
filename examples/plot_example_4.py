r"""Example 4: Simulating a Rotor
=============================

.. currentmodule:: pyvl

One of streinghts of potential flow is that it allows for fast computations of
simulations with complex and/or moving geometry. As such, this example will demonstrate
how a propeller can be simulated.
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
# Propeller is loaded from a mesh file.

RPM = 600
ang_vel = (RPM / 60) * 2 * np.pi

geo = pyvl.Geometry.from_meshio(
    label="prop",
    reference_frame=pyvl.RotorReferenceFrame(omega=(-ang_vel, 0, 0)),
    mesh=mio.read(examples.example_file_name("prop.msh")),
)

sim_geo = pyvl.SimulationGeometry(geo)


v_inf = 10
flow_conditions = pyvl.FlowConditionsUniform(v_inf, 0.0, 0.0)
model_settings = pyvl.ModelSettings(vortex_limit=1e-6)

# Plot it just to show what it looks like
sim_geo.polydata_at_time(0.0).plot(interactive=False)

dt = 10 / 360 / (RPM / 60)
time_settings = pyvl.TimeSettings(120, dt)

settings = pyvl.SolverSettings(flow_conditions, model_settings, time_settings)

te_lines = sim_geo.te_normal_criterion(-0.5)  # -0.5 feels nice in my bones
ands, asur = sim_geo.line_adjecency_information(te_lines)

wake_model = pyvl.WakeModelLineExplicitUnsteady(
    ands,
    te_lines,
    asur,
    settings.model_settings.vortex_limit,
    -settings.time_settings.dt,
    60,
)

# %%
#
# Running the Solver
# ------------------
#
# Running the solver is done exactly as before:

from threadpoolctl import threadpool_limits  # noqa: E402

# For this case, use 4 threads
with threadpool_limits(limits=4):
    results = pyvl.run_solver(sim_geo, settings, wake_model, None)
    pressures = pyvl.postprocess.compute_surface_dynamic_pressure(results)


# %%
#
# Visualize the Results
# ---------------------
#
# After running the post-processor again, the difference can be seen. Using `pyvista`
# the results are combined to create a

plotter = pv.Plotter(notebook=False, off_screen=True)
plotter.add_axes()

plotter.open_gif("output/example_4/propeller.gif")

for i, t in enumerate(time_settings.output_times):
    sg = sim_geo.polydata_at_time(t)
    sg.cell_data["Pressure"] = pressures[i]
    sg.set_active_scalars("Pressure")
    # sg.save(f"examples/output/example_4/mesh-{i:03d}.vtp")

    wmod = results.wake_models[i]
    assert isinstance(wmod, pyvl.WakeModelLineExplicitUnsteady)
    wm = wmod.as_polydata()
    wm.set_active_scalars(None)

    edges = wm.extract_all_edges()
    # edges.save(f"examples/output/example_4/wake-{i:03d}.vtk")
    plotter.add_mesh(edges, name="wake")
    plotter.add_mesh(sg, name="geo")
    plotter.write_frame()

plotter.close()
