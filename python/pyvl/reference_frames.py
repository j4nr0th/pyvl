"""Implementation of various different types of ReferenceFrames."""

from __future__ import annotations

from typing import Self

import numpy as np
import numpy.typing as npt

from pyvl._typing import VecLike3
from pyvl.cvl import ReferenceFrame
from pyvl.fio.io_common import HirearchicalMap


class TranslatingReferenceFrame(ReferenceFrame):
    r"""Reference frame moving with constant linear velocity.

    If the rotation of the reference frame is represented by :math:`\mathbf{R}`, then the
    transformation of this reference frame can be written as:

    .. math::

        \vec{r}^\prime(t) = \mathbf{R} \vec{r} + \vec{d} + \vec{v} (t - t_0),

    where the initial time is :math:`t_0` and the velocity is :math:`\vec{v}`.

    Parameters
    ----------
    offset : VecLike3, default: (0, 0, 0)
        Initial offset of the reference frame :math:`\vec{d}`

    theta : VecLike3, default: (0, 0, 0)
        Rotations of the reference frame around its axis.

    velocity : VecLike3, default: (0, 0, 0)
        Velocity of the reference frame :math:`\vec{v}`.

    parent : ReferenceFrame, optional
        Parent to which the position of the reference frame is relative to.

    time : float, default: 0.0
        Initial time :math:`t_0` at which the position of the origin is at
        :math:`\vec{d}`.
    """

    vel: npt.NDArray[np.float64]
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
        """Compute reference frame at the given time.

        This is used when the reference frame is moving or rotating in space.

        Parameters
        ----------
        t : float
            Time at which the reference frame is needed.

        Returns
        -------
        TranslatingReferenceFrame
            New reference frame at the given time.
        """
        dt = t - self.time

        return TranslatingReferenceFrame(
            offset=self.offset + dt * self.vel,
            velocity=self.vel,
            time=t,
            parent=self.parent.at_time(t) if self.parent is not None else None,
            theta=self.angles,
        )

    def save(self, group: HirearchicalMap) -> None:
        """Serialize the ReferenceFrame into a HirearchicalMap.

        Parameters
        ----------
        hmap: HirearchicalMap
            :class:`HirearchicalMap` in which to save the reference frame into.
        """
        group.insert_array("velocity", self.vel)
        group.insert_scalar("time", self.time)
        return super().save(group)

    @classmethod
    def load(cls, group: HirearchicalMap, parent: ReferenceFrame | None = None) -> Self:
        """Load the ReferenceFrame from a HirearchicalMap.

        Parameters
        ----------
        hmap : HirearchicalMap
            A :class:`HirearchicalMap`, which was created with a call to
            :meth:`ReferenceFrame.save`.
        parent : ReferenceFrame, optional
            Parent of the reference frame.

        Returns
        -------
        Self
            Deserialized :class:`ReferenceFrame`.
        """
        offset = group.get_array("offset")
        angles = group.get_array("angles")
        velocity = group.get_array("velocity")
        time = group.get_scalar("time")

        return cls(offset, angles, velocity, parent, time)

    def add_velocity(
        self, positions: npt.NDArray[np.float64], velocity: npt.NDArray[np.float64], /
    ) -> None:
        """Add the velocity at the specified positions.

        This method exists to account for the motion of the mesh from non-stationary
        reference frames.

        Parameters
        ----------
        positions : (N, 3) array
            Array of :math:`N` position vectors specifying the positions where the
            velocity should be updated.

        velocity : (N, 3) array
            Array to which the velocity vectors at the specified positions should be added
            to. These values should be added to and not just overwritten.
        """
        # Do it just to enforce the types.
        super().add_velocity(positions, velocity)
        # Add a constant velocity
        velocity[:, :] += self.vel[None, :]


class RotorReferenceFrame(ReferenceFrame):
    r"""Reference frame designed for rotors.

    Rotates around the given axis by a constant angular velocity. The velocity
    of a point with the position vector :math:`\vec{r}` is described by the equation:

    .. math::

        \frac{d \vec{r}}{d t} = \vec{r} \times \vec{\omega}

    Parameters
    ----------
    offset : VecLike3, default: (0, 0, 0)
        Initial offset of the reference frame :math:`\vec{d}`

    theta : VecLike3, default: (0, 0, 0)
        Rotations of the reference frame around its axis.

    velocity : VecLike3, default: (0, 0, 0)
        Velocity of the reference frame :math:`\vec{v}`.

    parent : ReferenceFrame, optional
        Parent to which the position of the reference frame is relative to.

    omega : VecLike3, default: (0, 0, 0)
        Angular velocity vector. Both direction and magnitude matter. If it
        is too small in magnitude, then ``(0, 0, 1)`` will be used as the
        direction.

    time : float, default: 0.0
        Initial time :math:`t_0` at which there is no rotation.
    """

    omega: npt.NDArray[np.float64]
    time: float

    def __new__(
        cls,
        offset: VecLike3 = (0, 0, 0),
        theta: VecLike3 = (0, 0, 0),
        parent: ReferenceFrame | None = None,
        omega: VecLike3 = (0, 0, 0),
        time: float = 0.0,
    ) -> Self:
        """Create a new RotorReferenceFrame."""
        self = super().__new__(
            cls,
            offset=offset,
            theta=theta,
            parent=parent,
        )
        self.omega = np.array(omega, np.float64).reshape((3,))
        self.time = time
        return self

    @staticmethod
    def _rotate(
        omega: npt.NDArray[np.float64], r: npt.NDArray[np.float64], dt: float
    ) -> npt.NDArray[np.float64]:
        """Rotate a vector around the origin with angular velocity omega for time dt.

        Parameters
        ----------
        omega : (3,) array
            Angular velocity vector.
        r : (3,) array
            Position vector to rotate.
        dt : float
            Amount of time to pass.

        Returns
        -------
        (3,) array
            Rotated position vector.
        """
        angular_velocity = np.linalg.norm(omega)
        if angular_velocity == 0:
            return np.array(r)
        direction = omega / angular_velocity
        rp = np.dot(r, direction) * direction
        v1 = r - rp
        v2 = np.cross(r, direction)
        phi = np.arctan2(v2, v1)
        rad = np.hypot(v1, v2)
        dr = rad * np.cos(phi + angular_velocity * dt)
        return rp + dr

    def at_time(self, t: float) -> RotorReferenceFrame:
        """Compute reference frame at the given time.

        This is used when the reference frame is moving or rotating in space.

        Parameters
        ----------
        t : float
            Time at which the reference frame is needed.

        Returns
        -------
        TranslatingReferenceFrame
            New reference frame at the given time.
        """
        rot_mat = self.rotation_matrix
        rot_mat[0, :] = RotorReferenceFrame._rotate(
            self.omega, rot_mat[0, :], t - self.time
        )
        rot_mat[1, :] = RotorReferenceFrame._rotate(
            self.omega, rot_mat[1, :], t - self.time
        )
        rot_mat[2, :] = RotorReferenceFrame._rotate(
            self.omega, rot_mat[2, :], t - self.time
        )
        new_angles = ReferenceFrame.angles_from_rotation(rot_mat)
        return RotorReferenceFrame(
            self.offset,
            new_angles,
            self.parent.at_time(t) if self.parent else None,
            self.omega,
            t,
        )

    def __repr__(self) -> str:
        """Representation of the reference frame."""
        offsets = self.offset
        angles = self.angles
        return (
            f"{type(self).__name__}({offsets[0]:g}, {offsets[1]:g}, {offsets[2]:g}"
            f", {angles[0]:g}, {angles[1]:g}, {angles[2]:g}, {self.parent}, "
            f"time={self.time:g}, omega={self.omega})"
        )

    def save(self, group: HirearchicalMap) -> None:
        """Serialize the reference frame into a HirearchicalMap."""
        group.insert_array("offset", self.offset)
        group.insert_array("angles", self.angles)
        group.insert_array("omega", self.omega)
        group.insert_scalar("time", self.time)

    @classmethod
    def load(cls, group: HirearchicalMap, parent: ReferenceFrame | None = None) -> Self:
        """Deserialize the reference frame from a HirearchicalMap."""
        return cls(
            offset=group.get_array("offset"),
            theta=group.get_array("angles"),
            parent=parent,
            omega=group.get_array("omega"),
            time=group.get_scalar("time"),
        )

    def add_velocity(
        self, positions: npt.NDArray[np.float64], velocity: npt.NDArray[np.float64], /
    ) -> None:
        """Add the velocity at the specified positions.

        This method exists to account for the motion of the mesh from non-stationary
        reference frames.

        Parameters
        ----------
        positions : (N, 3) array
            Array of :math:`N` position vectors specifying the positions where the
            velocity should be updated.

        velocity : (N, 3) array
            Array to which the velocity vectors at the specified positions should be added
            to. These values should be added to and not just overwritten.
        """
        # Do it just to enforce the types.
        super().add_velocity(positions, velocity)

        # Add a rotation velocity
        v = np.cross(positions, self.omega[None, :])
        velocity[:, :] += v
