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
        forces = Mesh.line_forces(
            results.geometry.mesh, results.geometry.dual, reduced_c, positions, freestream
        )
        out.append(forces)

    return out
