"""Definitions of what a wake model ought to be and how it should work."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Self

import h5py
import numpy as np
from numpy import typing as npt

from pydust.flow_conditions import FlowConditions
from pydust.geometry import SimulationGeometry


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

    @abstractmethod
    def get_velocity(
        self,
        positions: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Compute velocity induced by wake at requested positions."""
        ...

    @abstractmethod
    def save(self, group: h5py.Group) -> None:
        """Save current state into a h5py group."""
        ...

    @classmethod
    @abstractmethod
    def load(cls, group: h5py.Group) -> Self:
        """Load the wake model state from the group."""
        ...


def _load_wake_model(group: h5py.Group) -> WakeModel:
    """Load the wake model from the HDF5 group."""
    # Load the fully qualified type name and parse it
    type_data = group["type"]
    assert isinstance(type_data, h5py.Dataset)
    type_name = str(type_data[()])
    mname, tname = type_name.rsplit(".", 1)
    # Import the module and the type
    mod = __import__(mname, fromlist=[tname])
    cls = getattr(mod, tname)
    # Create the type and pass the state to construct it from
    data = group["data"]
    assert isinstance(data, h5py.Group)
    return cls.load(data)


def _store_wake_model(wm: WakeModel, group: h5py.Group) -> None:
    """Store the wake model in the HDF5 group."""
    # Load the fully qualified type name and parse it
    t = type(wm)
    group["type"] = t.__module__ + "." + t.__qualname__
    # Create the type and pass the state to construct it from
    data_group = group.create_group("data")
    wm.save(data_group)
