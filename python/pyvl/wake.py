"""Definitions of what a wake model ought to be and how it should work."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Self

import numpy as np
from numpy import typing as npt

from pyvl.fio.io_common import HirearchicalMap
from pyvl.flow_conditions import FlowConditions
from pyvl.geometry import SimulationGeometry


class WakeModel(ABC):
    """A model of the wake, which can be used for simulation using the VLM.

    Requirements for a Wake Model
    -----------------------------

    The wake model must be capable of performing the following functions:

    - RWM1: Shall allow for saving and loading its current state.
    - RWM2: Shall be updated each iteration.
    - RWM3: Shall provide corrections to the induction system:
        - RWM3.1: Correct the system matrix (implicit correction).
        - RWM3.2: Correct the induced velocity (explicit correction).
    - RWM4: Shall provide a way to visualize its effects with PyVista
    - RWM5: Shall allow for a way to compute induced velocity (connected with RWM3.2).
    """

    @abstractmethod
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
        ...

    @abstractmethod
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
        ...

    @abstractmethod
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
        ...

    @abstractmethod
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
        ...

    @abstractmethod
    def save(self) -> HirearchicalMap:
        """Serialize the object into a HirearchicalMap.

        Returns
        -------
        HirearchicalMap
            Serialized state of the :class:`WakeModel` object.
        """
        ...

    @classmethod
    @abstractmethod
    def load(cls, group: HirearchicalMap) -> Self:
        """Deserialize the object from a HirearchicalMap.

        Parameters
        ----------
        hmap : HirearchicalMap
            Serialized state of the :class:`WakeModel` object created by a call
            to :meth:`WakeModel.save`.

        Returns
        -------
        Self
            Deserialized :class:`WakeModel` object.
        """
        ...
