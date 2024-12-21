"""Definitions of what a wake model ought to be and how it should work."""

from abc import ABC

import numpy as np
from numpy import typing as npt

from pydust.geometry import SimulationGeometry
from pydust.settings import FlowConditions


class WakeModel(ABC):
    """A model of the wake, which can be used for simulation using the VLM."""

    def __init__(self, geometry: SimulationGeometry, *args, **kwargs) -> None: ...

    def update(
        self,
        time: float,
        flow: FlowConditions,
        *args,
        **kwargs,
    ) -> npt.NDArray[np.float64] | None:
        """Compute the wake contribution to the right hand side of the VLM system."""
        ...
