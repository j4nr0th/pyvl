"""Implementation of the flow solver."""

from collections.abc import Mapping
from time import perf_counter

import numpy as np
import numpy.typing as npt
import scipy.linalg as la

from pydust.cdust import Mesh
from pydust.geometry import Geometry, SimulationGeometry
from pydust.settings import ModelSettings, SolverSettings


def run_solver(
    geometries: Mapping[str, Geometry], settings: SolverSettings
) -> tuple[SimulationGeometry, list[npt.NDArray[np.float64]]]:
    """Run the flow solver to obtain specified circulations."""
    out_list: list[npt.NDArray[np.float64]] = []
    geometry = SimulationGeometry(geometries)
    del geometries
    times: npt.NDArray[np.float64]
    if settings.time_settings is None:
        times = np.array((0,), np.float64)
    else:
        times = np.arange(settings.time_settings.nt) * settings.time_settings.dt

    n_elements = geometry.msh.n_surfaces
    n_lines = geometry.msh.n_lines
    system_matrix = np.empty((n_elements, n_elements))
    line_buffer = np.empty((n_elements, n_lines, 3))
    velocity = np.empty((n_elements, 3))
    rhs = np.empty((n_elements))

    for iteration, time in enumerate(times):
        iteration_begin_time = perf_counter()
        # Update the mesh
        geometry.time = time
        # Get mesh properties
        control_points = geometry._cpts
        normals = geometry._normals
        # Compute flow velocity (if necessary)
        velocity = settings.flow_conditions.get_velocity(time, control_points, velocity)
        # Compute flow penetration at control points
        rhs = np.vecdot(normals, -velocity, rhs, axis=1)  # type: ignore
        # Compute normal induction
        system_matrix = geometry.msh.induction_matrix3(
            settings.model_settings.vortex_limit,
            control_points,
            normals,
            system_matrix,
            line_buffer,
        )
        # Decompose the system matrix to allow for solving multiple times
        decomp = la.lu_factor(system_matrix, overwrite_a=True)
        # Solve the linear system
        # By setting overwrite_b=True, rhs is where the output is written to
        circulation = la.lu_solve(decomp, rhs, overwrite_b=True)
        # Adjust circulations
        geometry.adjust_circulations(circulation)
        iteration_end_time = perf_counter()
        if (
            settings.time_settings is None
            or (settings.time_settings.output_interval is None)
            or (iteration % settings.time_settings.output_interval == 0)
        ):
            out_list.append(np.array(circulation))  # Make sure to create a copy
        print(
            f"Finished iteration {iteration} out of {len(times)} in "
            f"{iteration_end_time - iteration_begin_time:g} seconds."
        )

    del system_matrix, line_buffer, velocity, rhs

    return (geometry, out_list)


def compute_induced_velocities(
    msh: Mesh,
    circulation: npt.NDArray[np.float64],
    model_setting: ModelSettings,
    positions: npt.NDArray,
) -> npt.NDArray:
    """Compute velocity induced by the mesh with circulation."""
    induction_matrix = msh.induction_matrix(model_setting.vortex_limit, positions)
    return np.linalg.vecdot(induction_matrix, circulation[None, :, None], axis=1)
