"""Aerodynamic element support."""

from typing import Self

import numpy as np
import numpy.typing as npt

from pyvl.fio.io_common import HirearchicalMap
from pyvl.geometry import SimulationGeometry


class ImplicitElements:
    """State of the solver at a specific moment."""

    positions: npt.NDArray[np.float64]
    normals: npt.NDArray[np.float64]
    control_points: npt.NDArray[np.float64]
    cp_velocity: npt.NDArray[np.float64]
    circulation: npt.NDArray[np.float64]
    geometry: SimulationGeometry
    induction_matrix: npt.NDArray[np.float64]

    def __init__(
        self,
        geometry: SimulationGeometry,
    ) -> None:
        self.geometry = geometry
        self.positions = np.empty((geometry.n_points, 3), np.float64)
        self.normals = np.empty((geometry.n_surfaces, 3), np.float64)
        self.control_points = np.empty((geometry.n_surfaces, 3), np.float64)
        self.cp_velocity = np.empty((geometry.n_surfaces, 3), np.float64)
        self.circulation = np.empty((geometry.n_surfaces,), np.float64)

    def save(self) -> HirearchicalMap:
        """Serialize current state to a HirearchicalMap."""
        out = HirearchicalMap()
        out.insert_array("positions", self.positions)
        out.insert_array("normals", self.normals)
        out.insert_array("control_points", self.control_points)
        out.insert_array("cp_velocity", self.cp_velocity)
        out.insert_array("circulation", self.circulation)
        out.insert_hirearchycal_map("simulation_geometry", self.geometry.save())
        return out

    @classmethod
    def load(cls, hmap: HirearchicalMap) -> Self:
        """Deserialize current state from a HirearchicalMap."""
        geometry = SimulationGeometry.load(
            hmap.get_hirearchical_map("simulation_geometry")
        )
        self = cls(geometry=geometry)
        self.positions[:] = hmap.get_array("positions")
        self.normals[:] = hmap.get_array("normals")
        self.control_points[:] = hmap.get_array("control_points")
        self.circulation[:] = hmap.get_array("circulation")
        self.cp_velocity[:] = hmap.get_array("cp_velocity")

        return self

    def update_points(
        self,
        point_indices: slice,
        element_indices: slice,
        new_positions: npt.ArrayLike,
        new_velocities,
    ) -> None:
        """Update points."""
        # Update positions
        self.positions[point_indices] = new_positions
        # Update control points
        self.control_points[element_indices] = self.geometry.mesh.surface_average_vec3()
        ...
