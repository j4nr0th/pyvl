"""Functions related to computing dynamic pressure field on the mesh and surroundings."""

from collections.abc import Iterable

import numpy as np
import numpy.typing as npt

from pyvl.solver import SolverResults


def compute_surface_dynamic_pressure(
    results: SolverResults,
) -> list[npt.NDArray[np.float64]]:
    """Compute dynamic pressure on the surface centers of the mesh.

    Parameters
    ----------
    results : SolverResults
        Results of the solver.

    Returns
    -------
    list of M (N,) arrays
        List with array of pressure values for each output step.
    """
    out_times = results.settings.time_settings.output_times
    out_list: list[npt.NDArray[np.float64]] = []
    for i, t in enumerate(out_times):
        circulation = results.circulations[i, :]
        msh = results.geometry.mesh
        pos = results.geometry.positions_at_time(t)
        cpts = results.geometry.mesh.surface_average_vec3(pos)
        ind_mat = msh.induction_matrix(
            results.settings.model_settings.vortex_limit,
            pos,
            cpts,
        )
        induced_velocity: npt.NDArray[np.float64] = np.vecdot(
            ind_mat, circulation[None, :, None], axis=1
        )  # type: ignore
        vel = results.geometry.velocity_at_time(t)
        cp_vel = results.geometry.mesh.surface_average_vec3(vel)
        induced_velocity -= cp_vel
        freestream_velocity = results.settings.flow_conditions.get_velocity(t, cpts)

        wm = results.wake_models[i]
        if wm is not None:
            induced_velocity += wm.get_velocity(cpts)

        pressure = np.vecdot(
            induced_velocity, 0.5 * induced_velocity + freestream_velocity, axis=-1
        )  # type: ignore
        pressure = -results.settings.flow_conditions.get_density(t, cpts) * pressure
        out_list.append(pressure)

    return out_list


def compute_dynamic_pressure_variable(
    results: SolverResults, positions: Iterable[npt.NDArray]
) -> list[npt.NDArray[np.float64]]:
    """Compute dynamic pressure at the specified positions for each time step.

    Parameters
    ----------
    results : SolverResults
        Results of the solver.
    positions : Iterable of (N, 3) array
        Iterable which contains arrays of positions where the velocity should be computed
        for each time step.

    Returns
    -------
    list of (N,) array
        List of velocity vectors for each output step.
    """
    out_times = results.settings.time_settings.output_times
    out_list: list[npt.NDArray[np.float64]] = list()
    for i, (t, pts) in enumerate(zip(out_times, positions, strict=True)):
        cpts = np.ascontiguousarray(pts, dtype=np.float64)
        if len(cpts.shape) != 2 or cpts.shape[1] != 3:
            raise ValueError(
                "Positions must be an array of 3 component position vectors."
            )
        circulation = results.circulations[i, :]
        msh = results.geometry.mesh
        pos = results.geometry.positions_at_time(t)
        ind_mat = msh.induction_matrix(
            results.settings.model_settings.vortex_limit,
            pos,
            cpts,
        )
        induced_velocity: npt.NDArray[np.float64] = np.vecdot(
            ind_mat, circulation[None, :, None], axis=1
        )  # type: ignore
        freestream_velocity = results.settings.flow_conditions.get_velocity(t, cpts)

        wm = results.wake_models[i]
        if wm is not None:
            induced_velocity += wm.get_velocity(cpts)

        pressure = np.vecdot(
            induced_velocity, 0.5 * induced_velocity + freestream_velocity, axis=-1
        )  # type: ignore
        pressure = -results.settings.flow_conditions.get_density(t, cpts) * pressure
        out_list.append(pressure)
    return out_list
