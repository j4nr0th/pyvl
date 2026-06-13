"""Functions related to computing dynamic pressure field on the mesh and surroundings."""

from collections.abc import Iterable

import numpy as np
import numpy.typing as npt

from pyvl.solver import SolverResults, _compute_induced_velocity


def compute_surface_dynamic_pressure(
    results: SolverResults, n_threads: int = 1
) -> list[npt.NDArray[np.double]]:
    """Compute dynamic pressure on the surface centers of the mesh.

    Parameters
    ----------
    results : SolverResults
        Results of the solver.

    n_threads : int, default: 1
        Number of threads to use for calculations.

    Returns
    -------
    list of M (N,) arrays
        List with array of pressure values for each output step.
    """
    out_times = results.settings.time_settings.output_times
    out_list: list[npt.NDArray[np.double]] = []
    for i, t in enumerate(out_times):
        circulation = results.circulations[i, :]
        msh = results.geometry.mesh
        pos, vel = results.geometry.geometry_at_time(t)
        cpts = msh.surface_average_vec3(pos)
        tol = results.settings.model_settings.vortex_limit
        circulation = results.circulations[i, :]
        line_circulations = results.geometry.dual.line_circulations(circulation)
        pos = results.geometry.positions_at_time(t)
        freestream_velocity = results.settings.flow_conditions.get_velocity(t, cpts)

        wm = results.wake_states[i]
        total_velocity = _compute_induced_velocity(
            time=t,
            tol=tol,
            mesh=msh,
            positions=pos,
            line_circulation=line_circulations,
            wake=wm,
            flow_cond=results.settings.flow_conditions,
            target=cpts,
            n_threads=n_threads,
        )

        pressure = np.sum(
            total_velocity * (0.5 * total_velocity + freestream_velocity), axis=-1
        )
        pressure = -results.settings.flow_conditions.get_density(t, cpts) * pressure
        out_list.append(pressure)

    return out_list


def compute_dynamic_pressure_variable(
    results: SolverResults, positions: Iterable[npt.ArrayLike], n_threads: int = 1
) -> list[npt.NDArray[np.double]]:
    """Compute dynamic pressure at the specified positions for each time step.

    Parameters
    ----------
    results : SolverResults
        Results of the solver.

    positions : Iterable of (N, 3) array_like
        Iterable which contains arrays of positions where the velocity should be computed
        for each time step.

    n_threads : int, default: 1
        Number of threads to use for calculations.

    Returns
    -------
    list of (N,) array
        List of pressure values for each output step.
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
        line_circulations = results.geometry.dual.line_circulations(circulation)
        pos = results.geometry.positions_at_time(t)
        freestream_velocity = results.settings.flow_conditions.get_velocity(t, cpts)

        wm = results.wake_states[i]
        total_velocity = _compute_induced_velocity(
            time=t,
            tol=results.settings.model_settings.vortex_limit,
            mesh=results.geometry.mesh,
            positions=pos,
            line_circulation=line_circulations,
            wake=wm,
            flow_cond=results.settings.flow_conditions,
            target=cpts,
            n_threads=n_threads,
        )

        pressure = np.sum(
            total_velocity * (0.5 * total_velocity + freestream_velocity), axis=-1
        )
        pressure = -results.settings.flow_conditions.get_density(t, cpts) * pressure
        out_list.append(pressure)
    return out_list
