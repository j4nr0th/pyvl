"""Implementation of the flow solver."""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import numpy.typing as npt
import scipy.linalg as la

from pydust.cdust import Mesh
from pydust.geometry import SimulationGeometry
from pydust.settings import ModelSettings, SolverSettings


@dataclass(frozen=True)
class SolverResults:
    """Class which contains solver results."""

    circulations: npt.NDArray[np.float64]
    times: npt.NDArray[np.float64]


def run_solver(geometry: SimulationGeometry, settings: SolverSettings) -> SolverResults:
    """Run the flow solver to obtain specified circulations."""
    times: npt.NDArray[np.float64]
    if settings.time_settings is None:
        times = np.array((0,), np.float64)
    else:
        times = np.arange(settings.time_settings.nt) * settings.time_settings.dt

    i_out = 0

    out_circulaiton_array = np.empty((len(times), geometry.n_surfaces), np.float64)

    for iteration, time in enumerate(times):
        iteration_begin_time = perf_counter()
        # Update the mesh
        # TODO: cache these updates.
        meshes: list[Mesh] = []
        for geo_name in geometry:
            info = geometry[geo_name]
            new_rf = info.rf.at_time(time)
            new_pos = new_rf.to_global_with_offset(info.msh.positions)
            meshes.append(info.msh.copy(new_pos))

        merged = Mesh.merge_meshes(*meshes)

        # Get mesh properties
        control_points = merged.surface_centers
        normals = merged.surface_normals
        # point_positions = merged.positions
        # Compute flow velocity (if necessary)
        element_velocity = settings.flow_conditions.get_velocity(time, control_points)
        # Compute flow penetration at control points
        rhs = np.vecdot(normals, -element_velocity, axis=1)  # type: ignore
        # Compute normal induction
        system_matrix = merged.induction_matrix3(
            settings.model_settings.vortex_limit,
            control_points,
            normals,
        )
        # Decompose the system matrix to allow for solving multiple times
        decomp = la.lu_factor(system_matrix, overwrite_a=True)
        # Solve the linear system
        # By setting overwrite_b=True, rhs is where the output is written to
        circulation = la.lu_solve(decomp, rhs, overwrite_b=True)
        # Adjust circulations
        for geo_name in geometry:
            info = geometry[geo_name]
            if not info.closed:
                continue
            circulation[info.surfaces] -= np.mean(circulation[info.surfaces])

        iteration_end_time = perf_counter()
        if (
            settings.time_settings is None
            or (settings.time_settings.output_interval is None)
            or (iteration % settings.time_settings.output_interval == 0)
        ):
            out_circulaiton_array[i_out, :] = circulation
            i_out += 1
        print(
            f"Finished iteration {iteration} out of {len(times)} in "
            f"{iteration_end_time - iteration_begin_time:g} seconds."
        )

    # del system_matrix, line_buffer, velocity, rhs
    return SolverResults(
        out_circulaiton_array,
        times,
    )


def compute_induced_velocities(
    msh: Mesh,
    circulation: npt.NDArray[np.float64],
    model_setting: ModelSettings,
    positions: npt.NDArray,
) -> npt.NDArray:
    """Compute velocity induced by the mesh with circulation."""
    induction_matrix = msh.induction_matrix(model_setting.vortex_limit, positions)
    return np.linalg.vecdot(induction_matrix, circulation[None, :, None], axis=1)
