"""Implementation of the flow solver."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from time import perf_counter

import numpy as np
import numpy.typing as npt
import scipy.linalg as la

from pydust.cdust import ReferenceFrame
from pydust.geometry import SimulationGeometry
from pydust.settings import SolverSettings


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

    pos = np.empty((geometry.n_points, 3), np.float64)
    norm = np.empty((geometry.n_surfaces, 3), np.float64)
    cpts = np.empty((geometry.n_surfaces, 3), np.float64)

    previous_rfs: dict[str, None | ReferenceFrame] = {name: None for name in geometry}

    for iteration, time in enumerate(times):
        iteration_begin_time = perf_counter()
        updated = 0
        for geo_name in geometry:
            info = geometry[geo_name]
            new_rf = info.rf.at_time(time)
            if new_rf == previous_rfs[geo_name]:
                # The geometry of that particular part did not change.
                continue
            updated += 1
            previous_rfs[geo_name] = new_rf
            pos[info.points] = new_rf.to_global_with_offset(info.pos)
            cpts[info.surfaces] = new_rf.to_global_with_offset(
                info.msh.surface_average_vec3(info.pos)
            )
            norm[info.surfaces] = new_rf.to_global_without_offset(
                info.msh.surface_normal(info.pos)
            )

        # Compute flow velocity
        element_velocity = settings.flow_conditions.get_velocity(time, cpts)
        # Compute flow penetration at control points
        rhs = np.vecdot(norm, -element_velocity, axis=1)  # type: ignore
        # if updated != 0:
        # Compute normal induction
        system_matrix = geometry.mesh.induction_matrix3(
            settings.model_settings.vortex_limit, pos, cpts, norm
        )

        # Apply the wake model's effect
        if settings.wake_model is not None:
            settings.wake_model.apply_corrections(cpts, norm, system_matrix, rhs)

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

        # update the wake model
        if settings.wake_model is not None:
            settings.wake_model.update(
                time, geometry, pos, circulation, settings.flow_conditions
            )

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
    simulation_geometry: SimulationGeometry,
    settings: SolverSettings,
    results: SolverResults,
    positions: npt.NDArray,
    workers: int | None = None,
) -> npt.NDArray:
    """Compute velocity induced by the mesh with circulation."""
    out_vec = np.empty((len(results.times), positions.shape[0], 3), np.float64)

    def _velocity_compute_function(
        i, time: float, circulation: npt.NDArray[np.float64]
    ) -> None:
        """Compute velocity induction."""
        points = simulation_geometry.positions_at_time(time)
        ind_mat = simulation_geometry.mesh.induction_matrix(
            settings.model_settings.vortex_limit,
            points,
            positions,
        )
        np.vecdot(ind_mat, circulation[None, :, None], out=out_vec[i, ...], axis=1)  # type: ignore

    with ThreadPoolExecutor(workers) as executor:
        executor.map(
            _velocity_compute_function,
            range(results.times.size),
            results.times,
            results.circulations,
        )

    return out_vec
