"""Velocity field reconstruction."""

import numpy as np
import numpy.typing as npt

from pyvl.solver import SolverState


def compute_velocities(
    state: SolverState,
    positions: npt.NDArray,
    n_threads: int = 1,
    induced_only: bool = False,
) -> npt.NDArray[np.double]:
    """Compute velocity at the specified positions for each time step.

    Parameters
    ----------
    state : SolverState
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
    (N, 3) array
        Array of velocity vectors for the specified positions.
    """
    return state.compute_velocity(
        positions=positions,
        induced_only=induced_only,
        n_threads=n_threads,
    )
