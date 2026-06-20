"""Functions related to computing dynamic pressure field on the mesh and surroundings."""

import numpy as np
import numpy.typing as npt

from pyvl.solver import SolverState


def compute_surface_dynamic_pressure(
    state: SolverState, n_threads: int = 1
) -> npt.NDArray[np.double]:
    """Compute dynamic pressure on the surface centers of the mesh.

    Parameters
    ----------
    state : SolverState
        Results of the solver.

    n_threads : int, default: 1
        Number of threads to use for calculations.

    Returns
    -------
    npt.NDArray[np.double]
        Array of pressure values for the surface centers of the mesh.
    """
    pos = state.geometry.positions_at_time(state.time)
    cpts = state.geometry.mesh_joined.surface_average_vec3(pos)
    total_velocity = state.compute_velocity(
        positions=cpts, induced_only=True, n_threads=n_threads
    )

    pressure = np.sum(total_velocity**2, axis=-1)
    pressure = (
        -state.settings.flow_conditions.get_density(state.time, cpts) * pressure / 2
    )
    return pressure


def compute_dynamic_pressure(
    state: SolverState, positions: npt.ArrayLike, n_threads: int = 1
) -> npt.NDArray[np.double]:
    """Compute dynamic pressure at the specified positions for each time step.

    Parameters
    ----------
    state : SolverState
        Results of the solver.

    positions : (N, 3) array_like
        Array of positions where the velocity should be computed.

    n_threads : int, default: 1
        Number of threads to use for calculations.

    Returns
    -------
    npt.NDArray[np.double]
        Array of pressure values for the specified positions.
    """
    cpts = np.ascontiguousarray(positions, dtype=np.double)
    if len(cpts.shape) != 2 or cpts.shape[1] != 3:
        raise ValueError("Positions must be an array of 3 component position vectors.")
    total_velocity = state.compute_velocity(
        positions=cpts, induced_only=False, n_threads=n_threads
    )

    pressure = np.sum(
        state.settings.flow_conditions.get_velocity(time=state.time, positions=cpts) ** 2,
        axis=-1,
    ) - np.sum(total_velocity**2, axis=-1)
    pressure = state.settings.flow_conditions.get_density(state.time, cpts) * pressure / 2
    return pressure
