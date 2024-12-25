"""Base class and implementations for the FlowConditions."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Self

import numpy as np
import numpy.typing as npt

from pydust.fio.io_common import HirearchicalMap


class FlowConditions(ABC):
    """Base class for flow field information."""

    @abstractmethod
    def get_velocity(
        self,
        time: float,
        positions: npt.NDArray[np.float64],
        out_array: npt.NDArray[np.float64] | None = None,
    ) -> npt.NDArray[np.float64]:
        """Return velocity at the specified positions at given time."""
        ...

    @abstractmethod
    def save(self) -> HirearchicalMap:
        """Serialize itself into a HirearchicalMap."""
        ...

    @classmethod
    @abstractmethod
    def load(cls, hmap: HirearchicalMap) -> Self:
        """Deserialize from a HirearchicalMap."""
        ...


@dataclass(frozen=True)
class FlowConditionsUniform(FlowConditions):
    """Flow field with uniform velocity."""

    vx: float
    vy: float
    vz: float

    def get_velocity(
        self,
        time: float,
        positions: npt.NDArray[np.float64],
        out_array: npt.NDArray[np.float64] | None = None,
    ) -> npt.NDArray[np.float64]:
        """Return the uniform velocity."""
        del time
        v = np.array((self.vx, self.vy, self.vz), np.float64)
        if out_array is not None:
            out_array[:, :] = v[None, :]
            return out_array
        return np.full_like(positions, v)

    def save(self) -> HirearchicalMap:
        """Serialize itself into a HirearchicalMap."""
        hm = HirearchicalMap()
        hm.insert_array("velocity", (self.vx, self.vy, self.vz))
        return hm

    @classmethod
    def load(cls, hmap: HirearchicalMap) -> Self:
        """Deserialize from a HirearchicalMap."""
        return cls(*hmap.get_array("velocity"))


@dataclass(frozen=True)
class FlowConditionsRotating(FlowConditions):
    """Rotating flow field."""

    center_x: float
    center_y: float
    center_z: float

    omega_x: float
    omega_y: float
    omega_z: float

    def get_velocity(
        self,
        time: float,
        positions: npt.NDArray[np.float64],
        out_array: npt.NDArray[np.float64] | None = None,
    ) -> npt.NDArray[np.float64]:
        """Return the uniform velocity."""
        del time, out_array
        pos = positions - np.array(
            ((self.center_x, self.center_y, self.center_z),), np.float64
        )
        omg = np.array((self.omega_x, self.omega_y, self.omega_z), np.float64)
        v = np.linalg.cross(pos, omg)
        return v

    def save(self) -> HirearchicalMap:
        """Serialize itself into a HirearchicalMap."""
        hm = HirearchicalMap()
        hm.insert_array("center", (self.center_x, self.center_y, self.center_z))
        hm.insert_array("omega", (self.omega_x, self.omega_y, self.omega_z))
        return hm

    @classmethod
    def load(cls, hmap: HirearchicalMap) -> Self:
        """Deserialize from a HirearchicalMap."""
        return cls(*(*hmap.get_array("center"), *hmap.get_array("omega")))
