"""Post-processing related to forces."""

import numpy as np
import numpy.typing as npt

from pyvl.solver import SolverResults


# TODO: correct for wake
def circulatory_forces(
    results: SolverResults, n_threads: int = 1
) -> list[npt.NDArray[np.double]]:
    """Compute forces resulting from the mesh circulation."""
    out: list[npt.NDArray[np.double]] = list()
    for i, state in enumerate(results):
        positions = state.geometry.positions_at_time(state.time)
        velocity = state.compute_velocity(
            positions=positions, induced_only=False, n_threads=n_threads
        )
        # Copy circulations and set the circulation of shed lines to zero.
        circ = state.circulation.copy()
        circ[state.shed_lines] = 0

        forces = results.geometry.mesh_joined.line_forces(circ, positions, velocity)

        out.append(forces)

    return out
