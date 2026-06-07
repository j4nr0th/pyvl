"""Definitions of what a wake model ought to be and how it should work."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

import numpy as np
import numpy.typing as npt
import pyvista as pv

from pyvl.cvl import quad_induction, quad_normal_induction
from pyvl.fio.io_common import HirearchicalMap


@dataclass(frozen=True)
class WakeState:
    """State of the wake at a given time step."""

    quad_positions: npt.NDArray[np.double]
    quad_circulations: npt.NDArray[np.double]
    quad_count: int
    next_insertion_index: int

    @property
    def positions(self) -> npt.NDArray[np.double]:
        """Return the positions of the quads in the wake."""
        return self.quad_positions[: self.quad_count]

    def circulations(self) -> npt.NDArray[np.double]:
        """Return the circulations of the quads in the wake."""
        return self.quad_circulations[: self.quad_count]

    @property
    def capacity(self) -> int:
        """Return the maximum number of quads that can be stored in the wake state."""
        return self.quad_positions.shape[0]

    def __post_init__(self) -> None:
        """Validate the input data."""
        if self.quad_circulations.ndim != 1:
            raise ValueError("Circulations must be a 1D array.")
        if (
            self.quad_positions.ndim != 3
            or self.quad_positions.shape[1] != 4
            or self.quad_positions.shape[2] != 3
        ):
            raise ValueError("Positions must be a 3D array with shape (N, 4, 3).")

        if self.quad_positions.shape[0] != self.quad_circulations.size:
            raise ValueError("The number of positions and circulations must be the same.")

        if self.quad_count > 0 or self.quad_count > self.quad_positions.shape[0]:
            raise ValueError(
                "The quad count must be a non-negative integer and less than or equal to "
                "the capacity of the wake."
            )

        if (
            self.next_insertion_index < 0
            or self.next_insertion_index > self.quad_positions.shape[0]
        ):
            raise ValueError(
                "The next insertion index must be a non-negative integer and less than or"
                " equal to the capacity of the wake."
            )

    @classmethod
    def empty(cls, capacity: int) -> Self:
        """Create an empty wake state with the given capacity."""
        return cls(
            quad_positions=np.zeros((capacity, 4, 3), dtype=np.double),
            quad_circulations=np.zeros(capacity, dtype=np.double),
            quad_count=0,
            next_insertion_index=0,
        )

    def induced_velocity(
        self,
        tol: float,
        positions: npt.NDArray[np.double],
        out_velocity: npt.NDArray[np.double] | None = None,
    ) -> npt.NDArray[np.double]:
        """Compute the velocity induced by the wake at the given positions."""
        if self.quad_count == 0:
            if out_velocity is not None:
                out_velocity[:] = 0.0
                return out_velocity
            return np.zeros_like(positions, dtype=np.double)
        return quad_induction(
            tol=tol,
            quad_positions=self.quad_positions[: self.quad_count],
            quad_circulations=self.quad_circulations[: self.quad_count],
            target_positions=positions,
            out_velocity=out_velocity,
        )

    def induced_normal_velocity(
        self,
        tol: float,
        control_pts: npt.NDArray[np.double],
        normals: npt.NDArray[np.double],
        out_velocity: npt.NDArray[np.double] | None = None,
    ) -> npt.NDArray[np.double]:
        """Compute the normal velocity induced by the wake at the given control points."""
        if self.quad_count == 0:
            if out_velocity is not None:
                out_velocity[:] = 0.0
                return out_velocity
            return np.zeros(control_pts.shape[0], dtype=np.double)
        return quad_normal_induction(
            tol=tol,
            quad_positions=self.quad_positions[: self.quad_count],
            quad_circulations=self.quad_circulations[: self.quad_count],
            target_positions=control_pts,
            target_normals=normals,
            out_velocity=out_velocity,
        )

    def add_quads(
        self,
        new_positions: npt.NDArray[np.double],
        new_circulations: npt.NDArray[np.double],
        out_state: WakeState | None = None,
    ) -> WakeState:
        """Add new quads to the wake state.

        If not there is not enough space to add all the new quads, oldest ones will be
        replaced.

        Parameters
        ----------
        new_positions : (N, 4, 3) array
            Positions of the new quads to be added.

        new_circulations : (N,) array
            Circulations of the new quads to be added.

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
        new_quads = new_circulations.size
        if new_positions.shape != (new_quads, 4, 3):
            raise ValueError("New positions must have shape (N, 4, 3).")
        # Ensure we have an output state
        if out_state is None:
            out_positions = np.empty_like(self.quad_positions)
            out_circulations = np.empty_like(self.quad_circulations)
        else:
            out_positions = out_state.quad_positions
            out_circulations = out_state.quad_circulations

        # Check how many we can add to the end of the array
        append_count = min(new_quads, self.capacity - self.next_insertion_index)
        out_positions[
            self.next_insertion_index : self.next_insertion_index + append_count
        ] = new_positions[:append_count]
        out_circulations[
            self.next_insertion_index : self.next_insertion_index + append_count
        ] = new_circulations[:append_count]
        out_insertion_index = self.next_insertion_index + append_count
        if remainder := new_quads - append_count:
            out_positions[:remainder] = new_positions[append_count:]
            out_circulations[:remainder] = new_circulations[append_count:]
            out_insertion_index = remainder

        out_quad_count = min(self.quad_count + new_quads, self.capacity)

        return WakeState(
            quad_positions=out_positions,
            quad_circulations=out_circulations,
            quad_count=out_quad_count,
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
        if velocities.shape != (self.quad_count, 4, 3):
            raise ValueError("Velocities must have the same shape as quad positions.")

        out_positions = (
            out_state.quad_positions
            if out_state is not None
            else np.empty_like(self.quad_positions)
        )
        out_positions[: self.quad_count] = (
            self.quad_positions[: self.quad_count] + velocities * dt
        )
        if out_state is not None:
            return out_state

        return WakeState(
            quad_positions=out_positions,
            quad_circulations=self.quad_circulations.copy(),
            quad_count=self.quad_count,
            next_insertion_index=self.next_insertion_index,
        )

    def as_polydata(self) -> pv.PolyData:
        """Convert the wake state to a PyVista PolyData object for visualization."""
        if self.quad_count == 0:
            return pv.PolyData()

        # Create a PolyData object with the quad vertices
        points = self.quad_positions[: self.quad_count].reshape(-1, 3)
        quads = np.astype(np.arange(self.quad_count * 4).reshape(-1, 4), int)
        polydata = pv.PolyData.from_regular_faces(points, quads)

        # Add circulations as cell data
        polydata.cell_data["circulation"] = self.quad_circulations[: self.quad_count]

        return polydata

    def save(self) -> HirearchicalMap:
        """Serialize the wake state into a HirearchicalMap."""
        hmap = HirearchicalMap()
        hmap.insert_array("quad_positions", self.quad_positions)
        hmap.insert_array("quad_circulations", self.quad_circulations)
        hmap.insert_int("quad_count", self.quad_count)
        hmap.insert_int("next_insertion_index", self.next_insertion_index)
        return hmap

    @classmethod
    def load(cls, hmap: HirearchicalMap) -> Self:
        """Deserialize the wake state from a HirearchicalMap."""
        return cls(
            quad_positions=hmap.get_array("quad_positions"),
            quad_circulations=hmap.get_array("quad_circulations"),
            quad_count=hmap.get_int("quad_count"),
            next_insertion_index=hmap.get_int("next_insertion_index"),
        )
