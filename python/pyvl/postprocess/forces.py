"""Post-processing related to forces."""

import numpy as np
import numpy.typing as npt

from pyvl.solver import SolverResults


# TODO: correct for wake
def circulatory_forces(results: SolverResults) -> list[npt.NDArray[np.double]]:
    """Compute forces resulting from the mesh circulation."""
    out: list[npt.NDArray[np.double]] = list()
    for i, t in enumerate(results.settings.time_settings.output_times):
        line_circ = results.geometry.dual_joined.line_circulations(
            results.circulations[i, :] / (2 * np.pi)
        )
        positions, motion = results.geometry.geometry_at_time(t)

        freestream = results.settings.flow_conditions.get_velocity(t, positions)
        tol = results.settings.model_settings.vortex_limit
        ind_mat = results.geometry.mesh_joined.induction_matrix(tol, positions, positions)
        induced = np.sum(ind_mat * (results.circulations[i, :])[None, :, None], axis=1)

        wm = results.wake_states[i]
        induced += wm.induced_velocity(tol, positions)

        forces = results.geometry.mesh_joined.line_forces(
            line_circ, positions, freestream + induced - motion
        )

        out.append(forces)

    return out
