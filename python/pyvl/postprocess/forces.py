"""Post-processing related to forces."""

import numpy as np
import numpy.typing as npt

from pyvl.cvl import Mesh
from pyvl.solver import SolverResults


def circulatory_forces(results: SolverResults) -> list[npt.NDArray[np.float64]]:
    """Compute forces resulting from the mesh circulation."""
    out: list[npt.NDArray[np.float64]] = list()
    for i, t in enumerate(results.settings.time_settings.output_times):
        reduced_c = results.circulations[i, :] / (2 * np.pi)
        positions = results.geometry.positions_at_time(t)
        freestream = results.settings.flow_conditions.get_velocity(t, positions)
        ind_mat = results.geometry.mesh.induction_matrix(
            results.settings.model_settings.vortex_limit, positions, positions
        )
        induced = np.vecdot(ind_mat, (results.circulations[i, :])[None, :, None], axis=1)  # type: ignore
        motion = results.geometry.velocity_at_time(t)
        wm = results.wake_models[i]
        if wm is not None:
            induced += wm.get_velocity(positions)
        forces = Mesh.line_forces(
            results.geometry.mesh,
            results.geometry.dual,
            reduced_c,
            positions,
            freestream + induced - motion,
        )
        if wm is not None:
            wm.correct_forces(
                forces,
                results.geometry,
                positions,
                results.circulations[i, :],
                results.settings.flow_conditions,
            )
        out.append(forces)

    return out
