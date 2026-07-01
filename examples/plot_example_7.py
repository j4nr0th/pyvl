r"""Example 7: Propeller Refinement Study
    =====================================

.. currentmodule:: pyvl
"""  # noqa: D205, D400

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pyvista as pv
import pyvl
from pyvl import meshing

pv.set_plot_theme("document")
pv.set_jupyter_backend("html")
pv.global_theme.show_edges = False


# %%
#
# Creating the Geometry
# ---------------------
#
# A utility meshing tool is provided for creating simple vortex lattice meshes of
# propellers and rotors. :class:`VLBlade` is used to define a single blade of a propeller
# or rotor. This can then be used to create a full propeller or rotor geometry by rotating
# the blade.

R_OUTER = 0.200
R_INNER = 0.040
C_INNER = 0.050
C_OUTER = 0.100
blade_mesh = meshing.VLBlade(
    # Span is 0.2
    reference_line=R_OUTER - R_INNER,
    # Reference line is at the leading edge
    reference_chord_fraction=0.0,
    # Chord length increases linearly from C_INNER to C_OUTER
    chord_distribution=lambda s: C_INNER + (C_OUTER - C_INNER) * s,
    # Twist decreases linearly from 40 degrees to 0 degrees
    twist_distribution=lambda s: np.radians(40.0 * (1.0 - s)),
    # No camber
    camber_distribution=None,
)

N_BLADES = 3
N_SW = 40
N_CW = 15
base_geo = blade_mesh.mesh_geometry(
    spanwise_positions=N_SW, chordwise_positions=N_CW, label="blade1"
)
blades = []

RPS = 10  # 10 rev/s


def blade_orientation_function(t: float) -> tuple[float, float, float]:
    """Return the orientation of the blade at time t."""
    return (0.0, 0.0, 2 * np.pi * RPS * np.mod(t, 1))


base_rotating_rf = pyvl.ReferenceFrame(
    theta=blade_orientation_function, rotation=(0, 0, 2 * np.pi * RPS)
)
for i, theta in enumerate(np.linspace(0, 2 * np.pi, N_BLADES, endpoint=False)):
    # if i != 1:
    #     continue
    # Base geometry on the first blade, but rotate it around the z-axis
    blades.append(
        pyvl.Geometry(
            label=f"blade{i + 1}",
            positions=base_geo.positions,
            mesh=base_geo.msh,
            reference_frame=pyvl.ReferenceFrame(
                offset=(R_INNER * np.sin(theta), R_INNER * np.cos(theta), 0.0),
                theta=(0.0, 0.0, -theta),
                parent=base_rotating_rf,
            ),
        )
    )

sim_geo = pyvl.SimulationGeometry.from_geometries(*blades)
plotter = pv.Plotter()

plotter.add_mesh(sim_geo.polydata_at_time(0), color="lightblue")
plotter.add_axes_at_origin()
plotter.show()

# %%
#
# Preparing the Simulation Settings
# ---------------------------------
#
# With the geometry defined, we can set up the simulation settings.

v_inf = 0.5

shedding_lines: list[int] = []
line_offset = 0
lines_c = N_SW * (N_CW - 1)
lines_s = (N_SW - 1) * N_CW
for _ in range(N_BLADES):
    shedding_lines.extend(
        line_offset + np.arange(N_SW - 1, dtype=np.intp) * N_CW + (N_CW - 1) + lines_c
    )
    line_offset += lines_c + lines_s


# Plot the shedding lines
plotter = pv.Plotter()

pd = sim_geo.polydata_at_time(0)
plotter.add_mesh(pd, color="lightblue")
line_points = [sim_geo.mesh_joined.get_line_points(line) for line in shedding_lines]
plotter.add_lines(
    np.array(pd.points[np.concatenate(line_points)]), color="green", width=5
)

plotter.add_axes_at_origin()
plotter.show()


# Custom flow conditions


# Compute time steps based on angular velocity
N_PER_REV = 72
dt = 1 / (RPS * N_PER_REV)
t_stop = 5 / RPS  # Stop the flow after this many rotations


def flow_velocity(
    time: float,
    positions: npt.NDArray[np.double],
    out_array: npt.NDArray[np.double] | None = None,
) -> npt.NDArray[np.double]:
    """Override the get_velocity method to stop the flow after t_stop."""
    if time < t_stop:
        if out_array is None:
            return np.full_like(positions, fill_value=(0.0, 0.0, -v_inf))
        out_array[...] = (0.0, 0.0, -v_inf)
        return out_array

    if out_array is None:
        return np.zeros_like(positions)

    out_array[...] = 0
    return out_array


settings = pyvl.SolverSettings(
    flow_velocity=flow_velocity,
    model_settings=pyvl.ModelSettings(
        vortex_limit=1e-10,
        # pyvl.TransformationPlane(origin=(0, -0.5, 0), normal=(0, 0, 1)),
        symmetry_plane=None,
    ),
    wake_settings=pyvl.WakeSettings(
        wake_shedder=pyvl.WakeShedderUniform(shedding_lines),
        wake_element_capacity=10000,
    ),
)

# %%
#
# Prepare Output Directory
# ------------------------
#
# The simulation will be run for a number of time steps, and the results will be saved
# to an output directory. The output directory will be created if it does not already
# exist. Since we will have to load a callable, we have to have some way to serialize
# or deserialize it. For this example, we will use a predefined serializer that can
# handle the callable we are using.

serializer = pyvl.PredefinedSerializer(
    rotor_motion_fn=blade_orientation_function, flow_velocity=flow_velocity
)


out_dir = Path("output", "example_7")

output_settings = pyvl.OutputSettings.new_simple(
    naming_callback=lambda i, _: out_dir / f"step_{i:04d}.json",
    ftype="JSON",
    callable_serializer=serializer.serialize,
    callable_deserializer=serializer.deserialize,
)

# %%
#
# Running the Simulation
# ----------------------
#
# With the geometry and settings defined, we can now run the simulation. The simulation
# will be run for a number of time steps, and the results will be saved to the
# output directory.

# Do not run more than this many steps
MAX_STEPS = N_PER_REV * 5  # Run for 5 revolutions
THREAD_CNT = 6  # Use 6 threads for the simulation

if not out_dir.exists():
    out_dir.mkdir(exist_ok=True, parents=True)
    # Run the actual solver to steady state
    pyvl.run_solver_steady_state(
        geometry=sim_geo,
        settings=settings,
        dt=dt,
        max_steps=MAX_STEPS,
        output_settings=output_settings,
        n_threads=THREAD_CNT,
    )


# %%
#
# Post Processing
# -------------------
#
# With the simulation complete, we can now post process the results. We will load the
# results from the output directory, and then plot the results.

plotter = pv.Plotter(off_screen=True, window_size=(800, 800))
plotter.open_movie(out_dir / "rotor-smoothed.mp4", framerate=60)

plotter.set_position((1, 1, 0.5))
plotter.set_focus((0, 0, 0))

for res_file in out_dir.iterdir():
    if res_file.suffix != ".json":
        continue
    print(f"Loading results from {res_file}...", end="")
    result = pyvl.SolverState.load_from_file(
        res_file, deserializer=serializer.deserialize
    )
    print(f" Results loaded. Time: {result.time:.4f}.")
    point_circ = np.zeros(result.geometry.mesh_joined.n_points, dtype=np.double)
    for i_line, line_circ in enumerate(result.circulation):
        point_circ[np.array(result.geometry.mesh_joined.get_line_points(i_line))] += (
            line_circ
        )

    rotor_pd = result.geometry.polydata_at_time(result.time)
    rotor_pd.point_data["circulation"] = point_circ

    plotter.add_mesh(rotor_pd, color="lightblue", name="rotor", scalars="circulation")
    if result.wake.quad_count > 0:
        plotter.add_mesh(
            result.wake.as_polydata().extract_all_edges(), color="orange", name="wake"
        )
    plotter.write_frame()

plotter.show()
plotter.close()
del plotter
