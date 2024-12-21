"""Implementations of some basic wake models."""

import numpy as np
from numpy import typing as npt

from pydust.geometry import SimulationGeometry
from pydust.settings import FlowConditions
from pydust.wake import WakeModel


class WakeModelNone(WakeModel):
    """Wake model, which does nothing."""

    def __init__(self, geometry: SimulationGeometry) -> None:
        del geometry
        pass

    def update(self, time: float, flow: FlowConditions) -> None:
        """Returh the model's (no) contribution."""
        del time, flow
        return None


class WakeModelLineFrozen(WakeModel):
    """Models wake as lines moving with fixed velocity (frozen wake)."""

    shedding_elements: npt.NDArray[np.uint]
    wake_velocity: npt.NDArray[np.float64]
    current_time: float
    wake_positions: npt.NDArray[np.float64]

    def __init__(
        self,
        geometry: SimulationGeometry,
        wake_velocity: npt.ArrayLike,
        line_rows: int,
        dp_crit: float = 0.0,
    ) -> None:
        self.wake_velocity = np.array(wake_velocity, np.float64).reshape((3,))
        normals = np.empty((geometry.n_surfaces, 3), np.float64)
        for geo_name in geometry:
            geo = geometry[geo_name]
            normals[geo.surfaces] = geo.msh.surface_normal(geo.pos)
        self.shedding_elements = geometry.dual.dual_normal_criterion(dp_crit, normals)
