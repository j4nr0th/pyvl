"""Velocity field reconstruction."""

from collections.abc import Iterable

import numpy as np
import numpy.typing as npt

from pyvl.solver import SolverResults


def compute_velocities(
    results: SolverResults, positions: npt.NDArray
) -> npt.NDArray[np.float64]:
    """Compute velocity at the specified positions for each time step.

    Parameters
    ----------
    results : SolverResults
        Results of the solver.
    positions : (N, 3) array
        Array of positions where the velocity should be computed for all time steps.

    Returns
    -------
    (M, N, 3) array
        Array of velocity vectors for each output step.
    """
    out_times = results.settings.time_settings.output_times
    cpts = np.ascontiguousarray(positions, dtype=np.float64)
    if len(cpts.shape) != 2 or cpts.shape[1] != 3:
        raise ValueError("Positions must be an array of 3 component position vectors.")
    output_array = np.empty((out_times.size, cpts.shape[0], 3), np.float64)
    for i, t in enumerate(out_times):
        circulation = results.circulations[i, :]
        msh = results.geometry.mesh
        pos = results.geometry.positions_at_time(t)
        ind_mat = msh.induction_matrix(
            results.settings.model_settings.vortex_limit,
            pos,
            cpts,
        )
        np.vecdot(ind_mat, circulation[None, :, None], axis=1, out=output_array[i, :, :])  # type: ignore
        output_array[i, :, :] += results.settings.flow_conditions.get_velocity(t, cpts)
        wm = results.wake_models[i]
        if wm is not None:
            output_array[i, :, :] += wm.get_velocity(cpts)
    return output_array


def compute_velocities_variable(
    results: SolverResults, positions: Iterable[npt.NDArray]
) -> list[npt.NDArray[np.float64]]:
    """Compute velocity at the specified positions for each time step.

    Parameters
    ----------
    results : SolverResults
        Results of the solver.
    positions : Iterable of (N, 3) array
        Iterable which contains arrays of positions where the velocity should be computed
        for each time step.

    Returns
    -------
    list of (N, 3) array
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
        out_list.append(np.vecdot(ind_mat, circulation[None, :, None], axis=1))  # type: ignore
        out_list[-1] += results.settings.flow_conditions.get_velocity(t, cpts)
        wm = results.wake_models[i]
        if wm is not None:
            out_list[-1] += wm.get_velocity(cpts)
    return out_list
