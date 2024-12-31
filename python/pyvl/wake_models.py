"""Implementations of some basic wake models."""

from __future__ import annotations

from typing import Self

import numpy as np
import pyvista as pv
from numpy import typing as npt

from pyvl.fio.io_common import HirearchicalMap
from pyvl.flow_conditions import FlowConditions
from pyvl.geometry import INVALID_ID, Mesh, SimulationGeometry, mesh_to_polydata_faces
from pyvl.wake import WakeModel


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
        """Update the wake model.

        .. deprecated:: 0.0.1

            The ``positions`` argument will be removed, since information is already
            passed through the ``geometry`` argument.

        Parameters
        ----------
        time : float
            The time at which the wake model should now be at.
        geometry : SimulationGeometry
            The state of the :class:`SimulationGeometry` at the current time step.
        positions : (N, 3) array
            Positions of the mesh points.
        circulation : (N,) array
            Circulation values of vortex ring elements.
        flow : FlowConditions
            Flow conditions of the simulation.
        """
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
        """Compute velocity induced by wake at requested positions.

        Parameters
        ----------
        positions : (N, 3) array
            Array of positions where the velocity should be computed.
        """
        ind_mat = self.wake_mesh.induction_matrix(
            self.vortex_tol,
            self.wake_positions.reshape((-1, 3)),
            positions,
        )
        return np.vecdot(ind_mat, self.circulation.reshape((1, -1, 1)), axis=1)  # type: ignore

    def apply_corrections(
        self,
        control_pts: npt.NDArray[np.float64],
        normals: npt.NDArray[np.float64],
        mat_in: npt.NDArray[np.float64],
        rhs_in: npt.NDArray[np.float64],
    ) -> None:
        """Return implicit and explicit corrections for the no-prenetration conditions.

        Parameters
        ----------
        control_pts : (N, 3) array
            Positions where the no-penetration condition will be applied.
        normals : (N, 3) array
            Surface normals at the control points.
        mat_in : (N, N) array
            Left-hand side of the circulation equation. Describes normal velocity
            induction at the control points due to unknown circulations.
        rhs_in : (N,) array
            Right-hand side of the circulation equation. Describes the normal velocity
            induction at the control points due to know sources and free-stream.
        """
        ind_mat = self.wake_mesh.induction_matrix3(
            self.vortex_tol, self.wake_positions.reshape((-1, 3)), control_pts, normals
        ).reshape((control_pts.shape[0], self.shedding_lines.size, self.line_rows))
        # Split the induction matrix
        #   First row is always implicit
        implicit = ind_mat[:, :, 0]
        #   Rest are explicit
        explicit = ind_mat[:, :, 1:].reshape((control_pts.shape[0], -1))

        # Apply implicit correction to adjacent surfaces
        for i_line in range(self.shedding_lines.size):
            correction = implicit[:, i_line]
            s1 = self.adjacent_surfaces[i_line, 1]
            if s1 != INVALID_ID:
                mat_in[:, s1] -= correction
            s2 = self.adjacent_surfaces[i_line, 0]
            if s2 != INVALID_ID:
                mat_in[:, s2] += correction

        # Compute explicit correction
        circ = self.circulation.reshape((self.shedding_lines.size, self.line_rows))[
            :, 1:
        ].reshape((-1,))
        ev = np.vecdot(explicit, circ[None, :], axis=1)  # type: ignore
        rhs_in[:] += ev

    def as_polydata(self) -> pv.PolyData:
        """Return the visual representation of the wake model."""
        pd = pv.PolyData.from_irregular_faces(
            self.wake_positions.reshape((-1, 3)), mesh_to_polydata_faces(self.wake_mesh)
        )
        pd.cell_data["circulation"] = self.circulation.flatten()
        return pd

    def save(self) -> HirearchicalMap:
        """Serialize the object into a HirearchicalMap.

        Returns
        -------
        HirearchicalMap
            Serialized state of the :class:`WakeModelLineExplicitUnsteady` object.
        """
        out = HirearchicalMap()
        out.insert_array("bordering_nodes", self.bordering_nodes)
        out.insert_array("adjacent_surfaces", self.adjacent_surfaces)
        out.insert_array("shedding_lines", self.shedding_lines)
        out.insert_scalar("current_time", self.current_time)
        out.insert_int("step_count", self.step_count)
        out.insert_int("line_rows", self.line_rows)
        out.insert_array("wake_positions", self.wake_positions)
        out.insert_scalar("vortex_tol", self.vortex_tol)
        out.insert_array("circulation", self.circulation)
        return out

    @classmethod
    def load(cls, group: HirearchicalMap) -> Self:
        """Deserialize the object from a HirearchicalMap.

        Parameters
        ----------
        hmap : HirearchicalMap
            Serialized state of the :class:`WakeModelLineExplicitUnsteady`
            object created by a call to :meth:`WakeModelLineExplicitUnsteady.save`.

        Returns
        -------
        Self
            Deserialized :class:`WakeModelLineExplicitUnsteady` object.
        """
        bordering_nodes = group.get_array("bordering_nodes")
        shedding_lines = group.get_array("shedding_lines")
        adjacent_surfaces = group.get_array("adjacent_surfaces")
        vortex_tol = group.get_scalar("vortex_tol")
        time = group.get_scalar("current_time")
        line_rows = group.get_int("line_rows")
        circulation = group.get_array("circulation")
        wake_positions = group.get_array("wake_positions")
        step_count = group.get_int("step_count")

        out = cls(
            bordering_nodes,
            shedding_lines,
            adjacent_surfaces,
            vortex_tol,
            time,
            line_rows,
        )

        out.circulation[...] = circulation
        out.wake_positions[...] = wake_positions
        out.step_count = step_count

        return out

    def correct_forces(
        self,
        line_forces: npt.NDArray[np.float64],
        geometry: SimulationGeometry,
        positions: npt.NDArray[np.float64],
        circulation: npt.NDArray[np.float64],
        flow: FlowConditions,
    ) -> None:
        """Apply correction to force vectors at different mesh lines.

        Parameters
        ----------
        line_forces : (N, 3) array
            Array of force vectors at different mesh lines. Corrections should be
            added or subtracted in-place.
        geometry : SimulationGeometry
            The state of the :class:`SimulationGeometry` at the current time step.
        positions : (N, 3) array
            Positions of the mesh points.
        circulation : (N,) array
            Circulation values of vortex ring elements.
        flow : FlowConditions
            Flow conditions of the simulation.
        """
        tmp_mesh = WakeModelLineExplicitUnsteady._create_wake_mesh(
            1, self.shedding_lines.size
        )
        dual = tmp_mesh.compute_dual()
        cpts = np.ascontiguousarray(self.wake_positions[:, :, 0:2, :].reshape(-1, 3))
        circ = np.ascontiguousarray(self.circulation[:, 0] / (2 * np.pi))
        freestream = flow.get_velocity(self.current_time, cpts)
        ind_mat = geometry.mesh.induction_matrix(self.vortex_tol, positions, cpts)
        ind_v = np.vecdot(ind_mat, circulation[None, :, None], axis=1)  # type: ignore
        self_v = self.get_velocity(cpts)
        forces = Mesh.line_forces(tmp_mesh, dual, circ, cpts, freestream + ind_v + self_v)
        i_line: np.uint32
        for i, i_line in enumerate(self.shedding_lines):
            line_forces[i_line, :] -= forces[4 * i + 1, :]
