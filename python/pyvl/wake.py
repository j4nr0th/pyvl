"""Definitions of what a wake model ought to be and how it should work."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

import numpy as np
import numpy.typing as npt
import pyvista as pv

from pyvl.cvl import TransformationPlane, line_induction, line_normal_induction
from pyvl.fio.io_common import HirearchicalMap


@dataclass(frozen=True)
class WakeState:
    """State of the wake at a given time step."""

    line_positions: npt.NDArray[np.double]
    line_circulations: npt.NDArray[np.double]
    line_count: int
    next_insertion_index: int

    @property
    def positions(self) -> npt.NDArray[np.double]:
        """The positions of the lines in the wake."""
        return self.line_positions[: self.line_count]

    @property
    def circulations(self) -> npt.NDArray[np.double]:
        """The circulations of the lines in the wake."""
        return self.line_circulations[: self.line_count]

    @property
    def capacity(self) -> int:
        """The maximum number of lines that can be stored in the wake state."""
        return self.line_positions.shape[0]

    def __post_init__(self) -> None:
        """Validate the input data."""
        if self.line_circulations.ndim != 1:
            raise ValueError("Circulations must be a 1D array.")
        if (
            self.line_positions.ndim != 3
            or self.line_positions.shape[1] != 2
            or self.line_positions.shape[2] != 3
        ):
            raise ValueError("Positions must be a 3D array with shape (N, 2, 3).")

        if self.line_positions.shape[0] != self.line_circulations.size:
            raise ValueError("The number of positions and circulations must be the same.")

        if self.line_count < 0 or self.line_count > self.line_positions.shape[0]:
            raise ValueError(
                "The line count must be a non-negative integer and less than or equal to "
                "the capacity of the wake."
            )

        if (
            self.next_insertion_index < 0
            or self.next_insertion_index > self.line_positions.shape[0]
        ):
            raise ValueError(
                "The next insertion index must be a non-negative integer and less than or"
                " equal to the capacity of the wake."
            )

    @classmethod
    def empty(cls, capacity: int) -> Self:
        """Create an empty wake state with the given capacity."""
        return cls(
            line_positions=np.zeros((capacity, 2, 3), dtype=np.double),
            line_circulations=np.zeros(capacity, dtype=np.double),
            line_count=0,
            next_insertion_index=0,
        )

    def induced_velocity(
        self,
        tol: float,
        positions: npt.NDArray[np.double],
        symmetry_plane: TransformationPlane | None = None,
        out_velocity: npt.NDArray[np.double] | None = None,
        n_threads: int = 1,
    ) -> npt.NDArray[np.double]:
        """Compute the velocity induced by the wake at the given positions."""
        if self.line_count == 0:
            if out_velocity is not None:
                out_velocity[:] = 0.0
                return out_velocity
            return np.zeros_like(positions, dtype=np.double)
        return line_induction(
            tol=tol,
            line_positions=self.positions,
            line_circulations=self.circulations,
            target_positions=positions,
            symmetry_plane=symmetry_plane,
            out_velocity=out_velocity,
            n_threads=n_threads,
        )

    def induced_normal_velocity(
        self,
        tol: float,
        control_pts: npt.NDArray[np.double],
        normals: npt.NDArray[np.double],
        symmetry_plane: TransformationPlane | None = None,
        out_velocity: npt.NDArray[np.double] | None = None,
        n_threads: int = 1,
    ) -> npt.NDArray[np.double]:
        """Compute the normal velocity induced by the wake at the given control points."""
        if self.line_count == 0:
            if out_velocity is not None:
                out_velocity[:] = 0.0
                return out_velocity
            return np.zeros(control_pts.shape[0], dtype=np.double)
        return line_normal_induction(
            tol=tol,
            line_positions=self.positions,
            line_circulations=self.circulations,
            target_positions=control_pts,
            target_normals=normals,
            symmetry_plane=symmetry_plane,
            out_velocity=out_velocity,
            n_threads=n_threads,
        )

    def add_lines(
        self,
        new_positions: npt.NDArray[np.double],
        new_circulations: npt.NDArray[np.double],
        out_state: WakeState | None = None,
    ) -> WakeState:
        """Add new lines to the wake state.

        If not there is not enough space to add all the new lines, oldest ones will be
        replaced.

        Parameters
        ----------
        new_positions : (N, 2, 3) array
            Positions of the new lines to be added.

        new_circulations : (N,) array
            Circulations of the new lines to be added.

        out_state : WakeState, optional
            The wake state to write the updated state to. If not provided, new wake state
            will be created and returned.

        Returns
        -------
        WakeState
            Updated wake state. If the output wake state is provided, this will be the
            reference to the same object, otherwise a new wake state object will be
            returned.
        """
        # Validate input shapes
        if new_circulations.ndim != 1:
            raise ValueError("Circulations must be a 1D array.")
        new_lines = new_circulations.size
        if new_positions.shape != (new_lines, 2, 3):
            raise ValueError("New positions must have shape (N, 2, 3).")
        # Ensure we have an output state
        if out_state is None:
            out_positions = self.line_positions.copy()
            out_circulations = self.line_circulations.copy()
        else:
            out_positions = out_state.line_positions
            out_circulations = out_state.line_circulations

        # Check how many we can add to the end of the array
        append_count = min(new_lines, self.capacity - self.next_insertion_index)
        out_positions[
            self.next_insertion_index : self.next_insertion_index + append_count
        ] = new_positions[:append_count]
        out_circulations[
            self.next_insertion_index : self.next_insertion_index + append_count
        ] = new_circulations[:append_count]
        out_insertion_index = self.next_insertion_index + append_count
        if remainder := new_lines - append_count:
            out_positions[:remainder] = new_positions[append_count:]
            out_circulations[:remainder] = new_circulations[append_count:]
            out_insertion_index = remainder

        out_line_count = min(self.line_count + new_lines, self.capacity)

        return WakeState(
            line_positions=out_positions,
            line_circulations=out_circulations,
            line_count=out_line_count,
            next_insertion_index=out_insertion_index,
        )

    def update_wake(
        self,
        dt: float,
        velocities: npt.NDArray[np.double],
        out_state: WakeState | None = None,
    ) -> WakeState:
        """Update the wake state to the given time step.

        This method is used to advect the wake elements based on the
        velocities at the positions of the corners of the quads.

        Parameters
        ----------
        dt : float
            The time step for the update.

        velocities : (N, 3) array
            Velocities at the positions of the corners of the quads.

        out_state : WakeState, optional
            The wake state to write the updated state to. If not provided, new wake state
            will be created and returned.

        Returns
        -------
        WakeState
            Updated wake state. If the output wake state is provided, this will be the
            reference to the same object, otherwise a new wake state object will be
            returned.
        """
        if velocities.shape != (self.line_count, 2, 3):
            raise ValueError("Velocities must have the same shape as line positions.")

        out_positions = (
            out_state.line_positions
            if out_state is not None
            else np.empty_like(self.line_positions)
        )
        out_positions[: self.line_count] = (
            self.line_positions[: self.line_count] + velocities * dt
        )

        return WakeState(
            line_positions=out_positions,
            line_circulations=self.line_circulations.copy(),
            line_count=self.line_count,
            next_insertion_index=self.next_insertion_index,
        )

    def as_polydata(self) -> pv.PolyData:
        """Convert the wake state to a PyVista PolyData object for visualization."""
        if self.line_count == 0:
            return pv.PolyData()

        # Create a PolyData object with the line vertices
        points = self.line_positions[: self.line_count].reshape(-1, 3)
        lines = np.astype(np.arange(self.line_count * 2).reshape(-1, 2), int)
        polydata = pv.PolyData.from_regular_faces(points, lines)

        # Add circulations as cell data
        polydata.cell_data["circulation"] = self.line_circulations[: self.line_count]

        return polydata

    def save(self) -> HirearchicalMap:
        """Serialize the wake state into a HirearchicalMap."""
        hmap = HirearchicalMap()
        hmap.insert_array("line_positions", self.positions)
        hmap.insert_array("line_circulations", self.circulations)
        hmap.insert_int("line_count", self.line_count)
        hmap.insert_int("line_capacity", self.capacity)
        hmap.insert_int("next_insertion_index", self.next_insertion_index)
        return hmap

    @classmethod
    def load(cls, hmap: HirearchicalMap) -> Self:
        """Deserialize the wake state from a HirearchicalMap."""
        capacity = hmap.get_int("line_capacity")
        line_positions = np.zeros((capacity, 2, 3), dtype=np.double)
        line_circulations = np.zeros(capacity, dtype=np.double)
        line_positions[: hmap.get_int("line_count"), ...] = hmap.get_array(
            "line_positions"
        ).reshape(-1, 2, 3)
        line_circulations[: hmap.get_int("line_count"), ...] = hmap.get_array(
            "line_circulations"
        ).reshape(-1)
        return cls(
            line_positions=line_positions,
            line_circulations=line_circulations,
            line_count=hmap.get_int("line_count"),
            next_insertion_index=hmap.get_int("next_insertion_index"),
        )
