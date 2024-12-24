"""Definitions of what a wake model ought to be and how it should work."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Self

import numpy as np
from numpy import typing as npt

from pydust.flow_conditions import FlowConditions
from pydust.geometry import SimulationGeometry
from pydust.io_common import HirearchicalMap


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
        """Update the wake model."""
        ...

    def apply_corrections(
        self,
        control_pts: npt.NDArray[np.float64],
        normals: npt.NDArray[np.float64],
        mat_in: npt.NDArray[np.float64],
        rhs_in: npt.NDArray[np.float64],
    ) -> None:
        """Return implicit and explicit corrections for the no-prenetration conditions."""
        ...

    @abstractmethod
    def get_velocity(
        self,
        positions: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Compute velocity induced by wake at requested positions."""
        ...

    @abstractmethod
    def save(self) -> HirearchicalMap:
        """Save current state into a HirearchicalMap."""
        ...

    @classmethod
    @abstractmethod
    def load(cls, group: HirearchicalMap) -> Self:
        """Load the wake model state from a HirearchicalMap."""
        ...


def _load_wake_model(group: HirearchicalMap) -> WakeModel:
    """Load the wake model from a HirearchicalMap."""
    cls: type[WakeModel] = group.get_type("type")
    # Create the type and pass the state to construct it from
    data = group.get_hirearchical_map("data")
    return cls.load(data)


def _store_wake_model(wm: WakeModel) -> HirearchicalMap:
    """Store the wake model into a HirearchicalMap."""
    out = HirearchicalMap()
    out.insert_type("type", type(wm))
    # Create the type and pass the state to construct it from
    data_group = wm.save()
    out.insert_hirearchycal_map("data", data_group)
    return out
