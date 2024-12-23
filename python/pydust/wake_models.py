"""Implementations of some basic wake models."""

from typing import Self

import h5py
import numpy as np
import pyvista as pv
from numpy import typing as npt

from pydust.flow_conditions import FlowConditions
from pydust.geometry import INVALID_ID, Mesh, SimulationGeometry, mesh_to_polydata_faces
from pydust.wake import WakeModel


class WakeModelNone(WakeModel):
    """Wake model, which does nothing."""

    def __init__(self, *args, **kwargs) -> None:
        del args, kwargs
        pass

    def update(self, *args, **kwargs) -> None:
        """Returh the model's (no) contribution."""
        del args, kwargs
        return None


class WakeModelLineExplicitUnsteady(WakeModel):
    """Models wake as lines moving with induced velocity of the mesh.

    The model forms wake panels, which advect with the velocity induced by the free-stream
    and the mesh.
    """

    bordering_nodes: npt.NDArray[np.uint]
    adjacent_surfaces: npt.NDArray[np.uint]
    shedding_lines: npt.NDArray[np.uint]
    current_time: float
    step_count: int
    line_rows: int
    wake_mesh: Mesh
    wake_positions: npt.NDArray[np.float64]
    vortex_tol: float
    circulation: npt.NDArray[np.float64]

    @staticmethod
    def _create_wake_mesh(line_rows: int, n_lines: int) -> Mesh:
        """Create wake mesh to represent vortex panels."""
        nodes_per_row = line_rows + 1
        ring_connectivity = np.empty((n_lines, line_rows, 4), np.uint32)
        ring_connectivity[:, :, 0] = np.arange(line_rows, dtype=np.uint32)
        ring_connectivity[:, :, 1] = np.arange(line_rows, dtype=np.uint32) + nodes_per_row
        ring_connectivity[:, :, 2] = (
            np.arange(line_rows, dtype=np.uint32) + nodes_per_row + 1
        )
        ring_connectivity[:, :, 3] = np.arange(line_rows, dtype=np.uint32) + 1
        ring_connectivity[...] += (
            2 * nodes_per_row * np.arange(n_lines, dtype=np.uint32)
        )[:, None, None]  # type: ignore
        return Mesh(
            2 * nodes_per_row * n_lines,
            ring_connectivity.reshape((-1, 4)),  # type: ignore
        )

    def __init__(
        self,
        bordering_nodes: npt.NDArray[np.uint],
        shedding_lines: npt.NDArray[np.uint],
        adjacent_surfaces: npt.NDArray[np.uint],
        vortex_tol: float,
        time: float,
        line_rows: int,
    ) -> None:
        if len(shedding_lines) == 0:
            raise RuntimeError("The geometry has no edges which would shed any wake.")
        self.shedding_lines = np.array(shedding_lines, np.uint)
        self.wake_positions = np.empty(
            (self.shedding_lines.size, 2, line_rows + 1, 3), np.float64
        )

        self.line_rows = line_rows
        self.bordering_nodes = np.array(bordering_nodes, np.uint)
        self.adjacent_surfaces = np.array(adjacent_surfaces)
        self.vortex_tol = vortex_tol
        circulation_shape = (self.shedding_lines.size, line_rows)
        self.circulation = np.zeros(circulation_shape, np.float64)
        # Create the wake mesh
        self.wake_mesh = WakeModelLineExplicitUnsteady._create_wake_mesh(
            line_rows, self.shedding_lines.size
        )
        self.current_time = time
        self.step_count = 0

    def update(
        self,
        time: float,
        geometry: SimulationGeometry,
        positions: npt.NDArray[np.float64],
        circulation: npt.NDArray[np.float64],
        flow: FlowConditions,
    ) -> None:
        """Update the wake model."""
        dt = time - self.current_time
        self.current_time = time
        if self.step_count == 0:
            # First step needs more setup
            self.wake_positions[:, 0, 0, :] = positions[self.bordering_nodes[:, 0], :]
            self.wake_positions[:, 1, 0, :] = positions[self.bordering_nodes[:, 1], :]
            self.step_count = 1

        # Update positions of shed line's endpoints
        complete_rows = min(self.step_count, self.line_rows)
        for i in reversed(range(complete_rows)):
            old_pos = self.wake_positions[:, :, i, :]
            flat = np.array(old_pos.reshape((-1, 3)), np.float64)
            vel = flow.get_velocity(time, flat)
            ind_mat = geometry.mesh.induction_matrix(self.vortex_tol, positions, flat)
            vel += np.vecdot(ind_mat, circulation[None, :, None], axis=1)  # type: ignore
            new_pos = old_pos + vel.reshape(old_pos.shape) * dt
            self.wake_positions[:, :, i + 1, :] = new_pos

        # Make sure the first nodes stay at the trailing edge
        self.wake_positions[:, 0, 0, :] = positions[self.bordering_nodes[:, 0], :]
        self.wake_positions[:, 1, 0, :] = positions[self.bordering_nodes[:, 1], :]

        if complete_rows != self.line_rows:
            self.wake_positions[:, :, complete_rows + 1 :, :] = (
                self.wake_positions[:, :, complete_rows, :]
            )[:, :, None, :]

        # Shift the circulations
        self.circulation[:, 1:] = self.circulation[:, :-1]
        # Get circulations of each panel row
        # TODO: check if this needs a minus
        self.circulation[:, 0] = 0
        # Check for any un-adjacent surfaces
        mask = self.adjacent_surfaces != INVALID_ID
        self.circulation[mask[:, 1], 0] += circulation[
            self.adjacent_surfaces[mask[:, 1], 1]
        ]
        self.circulation[mask[:, 0], 0] -= circulation[
            self.adjacent_surfaces[mask[:, 0], 0]
        ]
        # self.circulation[:, 0] = (
        #     circulation[self.adjacent_surfaces[:, 1]]
        #     - circulation[self.adjacent_surfaces[:, 0]]
        # )

        self.step_count += 1

    def get_velocity(
        self,
        positions: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Compute velocity induced by wake at requested positions."""
        ind_mat = self.wake_mesh.induction_matrix(
            self.vortex_tol,
            self.wake_positions.reshape((-1, 3)),
            positions,
        )
        return np.vecdot(ind_mat, self.circulation[None, :, None], axis=1)  # type: ignore

    def apply_corrections(
        self,
        control_pts: npt.NDArray[np.float64],
        normals: npt.NDArray[np.float64],
        mat_in: npt.NDArray[np.float64],
        rhs_in: npt.NDArray[np.float64],
    ) -> None:
        """Return implicit and explicit corrections for the no-prenetration conditions."""
        ind_mat = self.wake_mesh.induction_matrix3(
            self.vortex_tol, self.wake_positions.reshape((-1, 3)), control_pts, normals
        ).reshape((control_pts.shape[0], self.shedding_lines.size, self.line_rows, 3))
        # Split the induction matrix
        #   First row is always implicit
        implicit = ind_mat[:, :, 0, :]
        #   Rest are explicit
        explicit = ind_mat[:, :, 1:, :].reshape((control_pts.shape[0], -1, 3))

        # Apply implicit correction to adjacent surfaces
        for i_line in range(self.shedding_lines.size):
            correction = implicit[:, i_line, :]
            mat_in[:, self.adjacent_surfaces[:, 1], :] += correction
            mat_in[:, self.adjacent_surfaces[:, 0], :] -= correction

        # Compute explicit correction
        circ = self.circulation.reshape((self.shedding_lines.size, self.line_rows))[
            :, 1:
        ].reshape((-1,))
        ev = np.vecdot(np.vecdot(explicit, circ[None, :, None], axis=1), normals, axis=-1)  # type: ignore
        rhs_in[:] += ev

    def as_polydata(self) -> pv.PolyData:
        """Return the visual representation of the wake model."""
        pd = pv.PolyData.from_irregular_faces(
            self.wake_positions.reshape((-1, 3)), mesh_to_polydata_faces(self.wake_mesh)
        )
        pd.cell_data["circulation"] = self.circulation.flatten()
        return pd

    def save(self, group: h5py.Group) -> None:
        """Serialize state into HDF5 group."""
        group["bordering_nodes"] = self.bordering_nodes
        group["adjacent_surfaces"] = self.adjacent_surfaces
        group["shedding_lines"] = self.shedding_lines
        group["current_time"] = self.current_time
        group["step_count"] = self.step_count
        group["line_rows"] = self.line_rows
        group["wake_mesh"] = self.wake_mesh
        group["wake_position"] = self.wake_positions
        group["vortex_tol"] = self.vortex_tol
        group["circulation"] = self.circulation

    @classmethod
    def load(cls, group: h5py.Group) -> Self:
        """Create instance of the wake model from the data in the HDF5 group."""
        bordering_nodes = group["bordering_nodes"]
        shedding_lines = group["shedding_lines"]
        adjacent_surfaces = group["adjacent_surfaces"]
        vortex_tol = group["vortex_tol"]
        time = group["current_time"]
        line_rows = group["line_rows"]
        circulation = group["circulation"]
        wake_positions = group["wake_positions"]
        step_count = group["step_count"]

        assert isinstance(bordering_nodes, h5py.Dataset)
        assert isinstance(shedding_lines, h5py.Dataset)
        assert isinstance(adjacent_surfaces, h5py.Dataset)
        assert isinstance(vortex_tol, h5py.Dataset)
        assert isinstance(time, h5py.Dataset)
        assert isinstance(line_rows, h5py.Dataset)
        assert isinstance(circulation, h5py.Dataset)
        assert isinstance(wake_positions, h5py.Dataset)
        assert isinstance(step_count, h5py.Dataset)

        out = cls(
            bordering_nodes[()],
            shedding_lines[()],
            adjacent_surfaces[()],
            vortex_tol[()],
            time[()],
            line_rows[()],
        )

        out.circulation[...] = circulation[()]
        out.wake_positions[...] = wake_positions[()]
        out.step_count = step_count[()]

        return out
