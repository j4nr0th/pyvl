r"""Example 6: Symmetry
===================

.. currentmodule:: pyvl

Symmetry can be used to enforce wall conditions for incompressible flow.
It is implemented in a way which removes the need to double the geometry. For
demonstration, the same propeller geometry from the previous example is used,
with a "wall" placed below it.
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
NT = 30
times = np.cumsum(np.full(NT, dt))


te_lines = sim_geo.te_normal_criterion(-0.5)  # -0.5 feels nice in my bones
PLANE_Y = 0.5  # y location of symmetry plane

model_settings = pyvl.ModelSettings(
    vortex_limit=1e-6,
    symmetry_plane=pyvl.TransformationPlane(
        origin=(0, PLANE_Y, 0),
        normal=(0, 1, 0),
    ),
)
wake_settings = pyvl.WakeSettings(
    wake_shedder=pyvl.WakeShedderUniform(te_lines),
    wake_element_capacity=len(te_lines) * NT * 2,
)
settings = pyvl.SolverSettings(flow_conditions, model_settings, wake_settings)


# %%
#
# Running the Solver
# ------------------
#
# Running the solver is done exactly as before:

wake_len = v_inf * times[-1]
wake_mid = wake_len / 2

NX = 51
NV = 51

out_dir = Path("output", "example_6")
out_dir.mkdir(exist_ok=True, parents=True)
plane = pv.Plane(
    center=(wake_mid, PLANE_Y, 0),
    i_size=2 * wake_len,
    j_size=2 * wake_len,
    i_resolution=NX,
    j_resolution=NV,
)


def plot_symmetry_plane_velocities(have_symmetry: bool, out_name: str):
    """Plot the example either with or without symmetry active."""
    if have_symmetry:
        model_settings = pyvl.ModelSettings(
            vortex_limit=1e-6,
            symmetry_plane=pyvl.TransformationPlane(
                origin=(0, PLANE_Y, 0),
                normal=(0, 1, 0),
            ),
        )
    else:
        model_settings = pyvl.ModelSettings(vortex_limit=1e-6)
    settings = pyvl.SolverSettings(flow_conditions, model_settings, wake_settings)
    results = pyvl.run_solver(sim_geo, settings, times=times, n_threads=4)

    velocities = [
        pyvl.postprocess.compute_velocities(
            state,
            positions=plane.points,
            n_threads=4,
        )
        for state in results
    ]

    velocity_mag = [np.linalg.norm(v, axis=-1) for v in velocities]

    max_mag = max(vm.max() for vm in velocity_mag)
    min_mag = min(vm.min() for vm in velocity_mag)

    plotter = pv.Plotter(notebook=False, off_screen=True)
    plotter.open_gif(out_dir / Path(out_name).with_suffix(".gif"), fps=10)
    plotter.set_position((wake_mid, PLANE_Y, 5))
    plotter.set_focus((wake_mid, PLANE_Y, 0))
    plotter.set_viewup((0, 1, 0))

    line = pv.Line(
        pointa=(plane.points[:, 0].min(), PLANE_Y, 0),
        pointb=(plane.points[:, 0].max(), PLANE_Y, 0),
    )

    for vel, vmag, state in zip(velocities, velocity_mag, results):
        sp = plane.copy()
        plane.point_data["velocity"] = vel
        sp.point_data["velocity"] = vel
        # zero out z component so they stay in the plane
        sp.point_data["velocity"][:, 2] = 0
        streamlines = sp.streamlines(
            vectors="velocity",
            pointa=(wake_mid, PLANE_Y - 1, 0),
            pointb=(wake_mid, PLANE_Y + 1, 0),
        )
        plotter.add_mesh(plane, name="vel", scalars="velocity", clim=(min_mag, max_mag))
        plotter.add_mesh(streamlines, name="str", color="black")
        plotter.add_mesh(
            line, color="red", label="symmetry line", name="line", line_width=5
        )
        plotter.add_mesh(state.geometry.polydata_at_time(state.time), name="geo")
        plotter.add_legend()
        plotter.write_frame()
    plotter.close()
    del plotter


plot_symmetry_plane_velocities(have_symmetry=True, out_name="with_symmetry")

# %%
#
# Now with no symmetry it is quite clear that the resulting streamlines pass over the
# symmetry plane. On the other hand, the simulation can run faster, as the solver does
# not need to recompute the system matrix.

plot_symmetry_plane_velocities(have_symmetry=False, out_name="without_symmetry")
