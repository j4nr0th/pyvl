r"""Example 4: Simulating a Rotor
=============================

.. currentmodule:: pyvl

One of streinghts of potential flow is that it allows for fast computations of
simulations with complex and/or moving geometry. As such, this example will demonstrate
how a propeller can be simulated.
"""  # noqa: D205, D400

from pathlib import Path

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
    reference_frame=pyvl.ReferenceFrame(
        # Need to specify both rotation (rate) and theta (rotation angle)
        rotation=(-ang_vel, 0, 0),
        theta=lambda t: (-ang_vel * t, 0, 0),
    ),
    mesh=mio.read(examples.example_file_name("prop.msh")),
)

sim_geo = pyvl.SimulationGeometry.from_geometries(geo)


v_inf = 10
flow_conditions = pyvl.FlowConditionsUniform(v_inf, 0.0, 0.0)

# Plot it just to show what it looks like
sim_geo.polydata_at_time(0.0).plot(interactive=False)

dt = 10 / 360 / (RPM / 60)
time_settings = pyvl.TimeSettings(120, dt)


te_lines = sim_geo.te_normal_criterion(-0.5)  # -0.5 feels nice in my bones

model_settings = pyvl.ModelSettings(
    vortex_limit=1e-6,
    wake_settings=pyvl.WakeSettings(
        wake_shedder=pyvl.WakeShedderUniform(te_lines),
        wake_element_capacity=len(te_lines) * time_settings.nt,
    ),
)
settings = pyvl.SolverSettings(flow_conditions, model_settings, time_settings)


# %%
#
# Running the Solver
# ------------------
#
# Running the solver is done exactly as before:

results = pyvl.run_solver(sim_geo, settings, None, n_threads=4)
pressures = pyvl.postprocess.compute_surface_dynamic_pressure(results, n_threads=4)


# %%
#
# Visualize the Results
# ---------------------
#
# After running the post-processor again, the difference can be seen. Using `pyvista`
# the results are combined to create a

plotter = pv.Plotter(notebook=False, off_screen=True)
plotter.add_axes()

out_dir = Path(__file__).parent / "output" / "example_4"
out_dir.mkdir(exist_ok=True)
plotter.open_movie(out_dir / "propeller.mp4", framerate=10)

for i, t in enumerate(time_settings.output_times):
    sg = sim_geo.polydata_at_time(t)
    sg.cell_data["Pressure"] = pressures[i]
    sg.set_active_scalars("Pressure")
    plotter.add_mesh(sg, name="geo")

    wmod = results.wake_states[i]
    if wmod.quad_count > 0:
        wm = wmod.as_polydata()
        wm.set_active_scalars(None)
        edges = wm.extract_all_edges()
        plotter.add_mesh(edges, name="wake")

    plotter.write_frame()

plotter.close()
