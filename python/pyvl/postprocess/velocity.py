"""Velocity field reconstruction."""

from collections.abc import Iterable

import numpy as np
import numpy.typing as npt

from pyvl.solver import SolverResults, _compute_induced_velocity


def compute_velocities(
    results: SolverResults,
    positions: npt.NDArray,
    n_threads: int = 1,
    induced_only: bool = False,
) -> npt.NDArray[np.double]:
    """Compute velocity at the specified positions for each time step.

    Parameters
    ----------
    results : SolverResults
        Results of the solver.

    positions : (N, 3) array
        Array of positions where the velocity should be computed for all time steps.

    n_threads : int, default: 1
        Number of threads to use for computing the velocity.

    induced_only : bool, default: False
        When set, freestream velocity is not included and only velocity induced by
        the geometry and its wake is computed.

    Returns
    -------
    (M, N, 3) array
        Array of velocity vectors for each output step.
    """
    out_times = results.settings.time_settings.output_times
    cpts = np.ascontiguousarray(positions, dtype=np.double)
    if len(cpts.shape) != 2 or cpts.shape[1] != 3:
        raise ValueError("Positions must be an array of 3 component position vectors.")
    output_array = np.empty((out_times.size, cpts.shape[0], 3), np.double)
    for i, t in enumerate(out_times):
        circulation = results.circulations[i, :]
        line_circulations = results.geometry.dual_joined.line_circulations(circulation)
        pos = results.geometry.positions_at_time(t)
        wm = results.wake_states[i]
        _compute_induced_velocity(
            time=t,
            tol=results.settings.model_settings.vortex_limit,
            mesh=results.geometry.mesh_joined,
            positions=pos,
            line_circulation=line_circulations,
            wake=wm,
            flow_cond=results.settings.flow_conditions if not induced_only else None,
            target=cpts,
            n_threads=n_threads,
            out=output_array[i, ...],
        )

    return output_array


def compute_velocities_variable(
    results: SolverResults,
    positions: Iterable[npt.NDArray],
    n_threads: int = 1,
    induced_only: bool = False,
) -> list[npt.NDArray[np.double]]:
    """Compute velocity at the specified positions for each time step.

    Parameters
    ----------
    results : SolverResults
        Results of the solver.

    positions : Iterable of (N, 3) array
        Iterable which contains arrays of positions where the velocity should be computed
        for each time step.

    n_threads : int, default: 1
        Number of threads to use for computing the velocity.

    induced_only : bool, default: False
        When set, freestream velocity is not included and only velocity induced by
        the geometry and its wake is computed.

    Returns
    -------
    list of (N, 3) array
        List of velocity vectors for each output step.
    """
    out_times = results.settings.time_settings.output_times
    out_list: list[npt.NDArray[np.double]] = list()
    for i, (t, pts) in enumerate(zip(out_times, positions, strict=True)):
        cpts = np.ascontiguousarray(pts, dtype=np.double)
        if len(cpts.shape) != 2 or cpts.shape[1] != 3:
            raise ValueError(
                "Positions must be an array of 3 component position vectors."
            )
        circulation = results.circulations[i, :]
        line_circulations = results.geometry.dual_joined.line_circulations(circulation)
        pos = results.geometry.positions_at_time(t)
        wm = results.wake_states[i]
        total_velocity = _compute_induced_velocity(
            time=t,
            tol=results.settings.model_settings.vortex_limit,
            mesh=results.geometry.mesh_joined,
            positions=pos,
            line_circulation=line_circulations,
            wake=wm,
            flow_cond=results.settings.flow_conditions if not induced_only else None,
            target=cpts,
            n_threads=n_threads,
        )
        out_list.append(total_velocity)
    return out_list
