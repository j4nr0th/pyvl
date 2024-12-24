"""Implementation of various different types of ReferenceFrames."""

from __future__ import annotations

from collections.abc import Callable
from typing import Self

import numpy as np
import numpy.typing as npt
from scipy.integrate import quad

from pydust._typing import VecLike3
from pydust.cdust import ReferenceFrame
from pydust.io_common import HirearchicalMap


class TranslatingReferenceFrame(ReferenceFrame):
    """Reference frame moving with constant linear velocity."""

    vel: npt.NDArray[np.float64]
    omega: np.float64
    axis_of_rotaiton: npt.NDArray[np.float64]
    time: float

    def __new__(
        cls,
        offset: VecLike3 = (0, 0, 0),
        theta: VecLike3 = (0, 0, 0),
        velocity: VecLike3 = (0, 0, 0),
        parent: ReferenceFrame | None = None,
        time: float = 0.0,
    ) -> Self:
        """Create a new TranslatingReferenceFrame."""
        self = super().__new__(cls, offset=offset, theta=theta, parent=parent)
        v = np.array(velocity, np.float64).reshape((3,))

        self.vel = v
        self.time = float(time)
        return self

    def at_time(self, t: float) -> TranslatingReferenceFrame:
        """Return a reference frame with a changed position at a different time."""
        dt = t - self.time

        return TranslatingReferenceFrame(
            offset=self.offset + dt * self.vel,
            velocity=self.vel,
            time=t,
            parent=self.parent.at_time(t) if self.parent is not None else None,
            theta=self.angles,
        )

    def save(self, group: HirearchicalMap) -> None:
        """Serialze the reference frame into a HirearchicalMap."""
        group.insert_array("velocity", self.vel)
        group.insert_scalar("time", self.time)
        return super().save(group)

    @classmethod
    def load(cls, group: HirearchicalMap, parent: ReferenceFrame | None = None) -> Self:
        """Deserialize the reference frame from a HirearchicalMap."""
        offset = group.get_array("offset")
        angles = group.get_array("angles")
        velocity = group.get_array("velocity")
        time = group.get_scalar("time")

        return cls(offset, angles, velocity, parent, time)


class RotorReferenceFrame(ReferenceFrame):
    """Reference frame designed for rotors.

    Rotates around the z-axis by a rotation rate given by a callable.
    """

    rotation: Callable[[float], float]
    time: float

    def __new__(
        cls,
        offset: VecLike3 = (0, 0, 0),
        theta: VecLike3 = (0, 0, 0),
        parent: ReferenceFrame | None = None,
        rotation: Callable[[float], float] | None = None,
        time: float = 0.0,
    ) -> Self:
        """Create a new RotorReferenceFrame."""
        self = super().__new__(
            cls,
            offset=offset,
            theta=theta,
            parent=parent,
        )
        self.rotation = rotation if rotation is not None else lambda t: 0.0 * t
        self.time = time
        return self

    def at_time(self, t: float) -> RotorReferenceFrame:
        """Return the reference frame at a specifice time."""
        offset = self.offset
        theta = self.angles
        theta[2] += quad(self.rotation, self.time, t)[0]
        return RotorReferenceFrame(
            offset=offset,
            theta=theta,
            parent=self.parent.at_time(t) if self.parent is not None else None,
            time=t,
            rotation=self.rotation,
        )

    def __repr__(self) -> str:
        """Representation of the reference frame."""
        offsets = self.offset
        angles = self.angles
        return (
            f"{type(self).__name__}({offsets[0]:g}, {offsets[1]:g}, {offsets[2]:g}"
            f", {angles[0]:g}, {angles[1]:g}, {angles[2]:g}, {self.parent}, "
            f"time={self.time:g}, rotation={self.rotation})"
        )

    def save(self, group: HirearchicalMap) -> None:
        """Serialze the reference frame into the HDF5 group."""
        raise NotImplementedError

    @classmethod
    def load(cls, group: HirearchicalMap, parent: ReferenceFrame | None = None) -> Self:
        """Deserialize the reference frame into the HDF5 group."""
        raise NotImplementedError
