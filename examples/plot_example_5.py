r"""Example 5: Simulating a Rotor
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
NT = 120
times = np.cumsum(np.full(NT, dt))


te_lines = sim_geo.te_normal_criterion(-0.5)  # -0.5 feels nice in my bones

model_settings = pyvl.ModelSettings(
    vortex_limit=1e-6,
)
wake_settings = pyvl.WakeSettings(
    wake_shedder=pyvl.WakeShedderUniform(te_lines),
    wake_element_capacity=len(te_lines) * NT,
)
settings = pyvl.SolverSettings(flow_conditions, model_settings, wake_settings)


# %%
#
# Running the Solver
# ------------------
#
# Running the solver is done exactly as before:

results = pyvl.run_solver(sim_geo, settings, times=times, n_threads=4)
pressures = [
    pyvl.postprocess.compute_surface_dynamic_pressure(state, n_threads=4)
    for state in results
]


# %%
#
# Visualize the Results
# ---------------------
#
# After running the post-processor again, the difference can be seen. Using `pyvista`
# the results are combined to create a

plotter = pv.Plotter(notebook=False, off_screen=True)
plotter.add_axes()

# out_dir = Path(__file__).parent / Path("output" / "example_5")
out_dir = Path("output", "example_5")
out_dir.mkdir(exist_ok=True, parents=True)
plotter.open_gif(out_dir / "propeller.gif", fps=10)
plotter.set_position((-4, 2, 2))
plotter.set_focus((+3, 0, 0))

for i, state in enumerate(results):
    sg = state.geometry.polydata_at_time(state.time)
    sg.cell_data["Pressure"] = pressures[i]
    sg.set_active_scalars("Pressure")
    plotter.add_mesh(sg, name="geo")

    if state.wake.quad_count > 0:
        wm = state.wake.as_polydata()
        wm.set_active_scalars(None)
        edges = wm.extract_all_edges()
        plotter.add_mesh(edges, name="wake")

    plotter.write_frame()

plotter.close()
# %%
#
# Visualize the Velocity
# ----------------------
#
# After running the post-processor again, the difference can be seen. Using `pyvista`
# the results are combined to create a


wake_len = v_inf * times[-1] * 1.2
wake_mid = wake_len / 2

NX = 101
NV = 41

plane = pv.Plane(
    center=(wake_mid, 0, 0),
    i_size=2 * wake_len,
    j_size=wake_len,
    i_resolution=NX,
    j_resolution=NV,
)


velocity_mag = [
    np.linalg.norm(
        pyvl.postprocess.compute_velocities(
            state,
            positions=plane.points,
            n_threads=4,
        ),
        axis=-1,
    )
    for state in results
]

max_mag = max(vm.max() for vm in velocity_mag)
min_mag = min(vm.min() for vm in velocity_mag)

plotter = pv.Plotter(notebook=False, off_screen=True)
plotter.open_gif(out_dir / "propeller-velocity.gif", fps=10)
plotter.set_position((wake_mid, 0, 6))
plotter.set_focus((wake_mid, 0, 0))

for vel, state in zip(velocity_mag, results):
    plane.point_data["velocity"] = vel
    plotter.add_mesh(plane, name="vel", scalars="velocity", clim=(min_mag, max_mag))
    plotter.add_mesh(state.geometry.polydata_at_time(state.time), name="geo")
    plotter.write_frame()
plotter.close()
