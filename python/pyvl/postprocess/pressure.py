"""Functions related to computing dynamic pressure field on the mesh and surroundings."""

from collections.abc import Iterable

import numpy as np
import numpy.typing as npt

from pyvl.solver import SolverResults


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
    out_list: list[npt.NDArray[np.double]] = []
    for i, state in enumerate(results):
        pos = state.geometry.positions_at_time(state.time)
        cpts = state.geometry.mesh_joined.surface_average_vec3(pos)
        total_velocity = state.compute_velocity(
            positions=cpts, induced_only=True, n_threads=n_threads
        )

        pressure = np.sum(total_velocity**2, axis=-1)
        pressure = (
            -results.settings.flow_conditions.get_density(state.time, cpts) * pressure / 2
        )
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
    out_list: list[npt.NDArray[np.double]] = list()
    for i, (state, pts) in enumerate(zip(results, positions, strict=True)):
        cpts = np.ascontiguousarray(pts, dtype=np.double)
        if len(cpts.shape) != 2 or cpts.shape[1] != 3:
            raise ValueError(
                "Positions must be an array of 3 component position vectors."
            )
        total_velocity = state.compute_velocity(
            positions=cpts, induced_only=False, n_threads=n_threads
        )

        pressure = np.sum(
            state.settings.flow_conditions.get_velocity(time=state.time, positions=cpts)
            ** 2,
            axis=-1,
        ) - np.sum(total_velocity**2, axis=-1)
        pressure = (
            results.settings.flow_conditions.get_density(state.time, cpts) * pressure / 2
        )
        out_list.append(pressure)
    return out_list
