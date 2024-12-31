"""Base class and implementations for the FlowConditions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Self

import numpy as np
import numpy.typing as npt

from pyvl.fio.io_common import HirearchicalMap


class FlowConditions(ABC):
    """Base class for flow field information.

    Must implement methods for computing free-stream velocity, as well
    as serialization and de-serialization.
    """

    @abstractmethod
    def get_velocity(
        self,
        time: float,
        positions: npt.NDArray[np.float64],
        out_array: npt.NDArray[np.float64] | None = None,
    ) -> npt.NDArray[np.float64]:
        """Return velocity at the specified positions at given time.

        Parameters
        ----------
        time : float
            Time at which the velocity field should be computed.

        positions : (N, 3) array
            Array of positions where the flow field should be computed.

        out_array : (N, 3) array, optional
            If specified, this array should receive the output values, along with being
            returned by the function.

        Returns
        -------
        (N, 3) array
            Array of :math:`N` velocity vectors at the specified positions and time. If
            the parameter ``out_array`` was specified, it should also be the return value
            of this function.
        """
        ...

    @abstractmethod
    def get_density(
        self,
        time: float,
        positions: npt.NDArray[np.float64],
        out_array: npt.NDArray[np.float64] | None = None,
    ) -> npt.NDArray[np.float64]:
        """Return density at the specified positions at given time.

        Parameters
        ----------
        time : float
            Time at which the density field should be computed.

        positions : (N, 3) array
            Array of positions where the flow field should be computed.

        out_array : (N,) array, optional
            If specified, this array should receive the output values, along with being
            returned by the function.

        Returns
        -------
        (N,) array
            Array of :math:`N` density values at the specified positions and time. If
            the parameter ``out_array`` was specified, it should also be the return value
            of this function.
        """
        ...

    @abstractmethod
    def save(self) -> HirearchicalMap:
        """Serialize itself into a HirearchicalMap.

        Returns
        -------
        HirearchicalMap
            Serialized state of the :class:`FlowConditions` object.
        """
        ...

    @classmethod
    @abstractmethod
    def load(cls, hmap: HirearchicalMap) -> Self:
        """Deserialize from a HirearchicalMap.

        Parameters
        ----------
        hmap : HirearchicalMap
            Serialized state of the :class:`FlowConditions` object created by a call to
            :meth:`FlowConditions.save`.

        Returns
        -------
        Self
            Deserialized :class:`FlowConditions` object.
        """
        ...


@dataclass(frozen=True)
class FlowConditionsUniform(FlowConditions):
    """Flow field with uniform velocity.

    Parameters
    ----------
    vx : float
        Velocity in the x direction.

    vy : float
        Velocity in the y direction.

    vz : float
        Velocity in the z direction.

    rho : float, default: 1.225
        Constant density of the fluid.

    Examples
    --------
    To visualize how this velocity field looks like, let's plot it with :mod:`pyvista`.

    .. jupyter-execute::

        >>> import numpy as np
        >>> import pyvista as pv
        >>> from pyvl import FlowConditionsUniform
        >>>
        >>> pv.set_plot_theme("document")
        >>> pv.set_jupyter_backend("html")

    Now make the grid and flow conditions object.

    .. jupyter-execute::

        >>> rg = pv.RectilinearGrid(
        ...     np.linspace(-1, +1, 6),
        ...     np.linspace(-1, +1, 6),
        ...     np.linspace(-1, +1, 6),
        ... )
        >>> fc = FlowConditionsUniform(0.2, -0.4, 0.3)

    It can now be plotted:

    .. jupyter-execute::

        >>> rg.point_data["velocity"] = fc.get_velocity(0.0, rg.points)
        >>> rg.glyph(factor=0.5).plot(interactive=False)
    """

    vx: float
    vy: float
    vz: float
    rho: float = 1.225

    def get_velocity(
        self,
        time: float,
        positions: npt.NDArray[np.float64],
        out_array: npt.NDArray[np.float64] | None = None,
    ) -> npt.NDArray[np.float64]:
        """Return velocity at the specified positions at given time.

        Parameters
        ----------
        time : float
            Time at which the velocity field should be computed.

        positions : (N, 3) array
            Array of positions where the flow field should be computed.

        out_array : (N, 3) array, optional
            If specified, this array should receive the output values, along with being
            returned by the function.

        Returns
        -------
        (N, 3) array
            Array of :math:`N` velocity vectors at the specified positions and time. If
            the parameter ``out_array`` was specified, it should also be the return value
            of this function.
        """
        del time
        v = np.array((self.vx, self.vy, self.vz), np.float64)
        if out_array is not None:
            out_array[:, :] = v[None, :]
            return out_array
        return np.full_like(positions, v)

    def get_density(
        self,
        time: float,
        positions: npt.NDArray[np.float64],
        out_array: npt.NDArray[np.float64] | None = None,
    ) -> npt.NDArray[np.float64]:
        """Return density at the specified positions at given time.

        Parameters
        ----------
        time : float
            Time at which the density field should be computed.

        positions : (N, 3) array
            Array of positions where the flow field should be computed.

        out_array : (N,) array, optional
            If specified, this array should receive the output values, along with being
            returned by the function.

        Returns
        -------
        (N,) array
            Array of :math:`N` density values at the specified positions and time. If
            the parameter ``out_array`` was specified, it should also be the return value
            of this function.
        """
        del time
        if out_array is not None:
            out_array[:, :] = self.rho
            return out_array
        return np.full_like(positions, self.rho)

    def save(self) -> HirearchicalMap:
        """Serialize itself into a HirearchicalMap.

        Returns
        -------
        HirearchicalMap
            Serialized state of the :class:`FlowConditionsUniform` object.
        """
        hm = HirearchicalMap()
        hm.insert_array("velocity", (self.vx, self.vy, self.vz))
        return hm

    @classmethod
    def load(cls, hmap: HirearchicalMap) -> Self:
        """Deserialize from a HirearchicalMap.

        Parameters
        ----------
        hmap : HirearchicalMap
            Serialized state of the :class:`FlowConditionsUniform` object created by a
            call to :meth:`FlowConditionsUniform.save`.

        Returns
        -------
        Self
            Deserialized :class:`FlowConditionsUniform` object.
        """
        return cls(*hmap.get_array("velocity"))


@dataclass(frozen=True)
class FlowConditionsRotating(FlowConditions):
    r"""Rotating flow field.

    The velocity this field represents a rotation around a point :math:`\vec{r}_0` with a
    constant angular velocity :math:`\vec{\omega}`, such that:

    .. math::

        \vec{v}_{\infty}\left(\vec{r}\right) = \left(\vec{r} - \vec{r}_0\right) \times
        \vec{\omega}

    Parameters
    ----------
    center_x : float
        The x coordinate of the center of rotation.

    center_y : float
        The y coordinate of the center of rotation.

    center_z : float
        The z coordinate of the center of rotation.

    omega_x : float
        The x component of the angular velocity vector.

    omega_y : float
        The y component of the angular velocity vector.

    omega_z : float
        The z component of the angular velocity vector.

    rho : float, default: 1.225
        Constant density of the fluid.

    Examples
    --------
    To visualize how this velocity field looks like, let's plot it with :mod:`pyvista`.

    .. jupyter-execute::

        >>> import numpy as np
        >>> import pyvista as pv
        >>> from pyvl import FlowConditionsRotating
        >>>
        >>> pv.set_plot_theme("document")
        >>> pv.set_jupyter_backend("html")

    Now make the grid and flow conditions object.

    .. jupyter-execute::

        >>> rg = pv.RectilinearGrid(
        ...     np.linspace(-1, +1, 6),
        ...     np.linspace(-1, +1, 6),
        ...     np.linspace(-1, +1, 6),
        ... )
        >>> fc = FlowConditionsRotating(0.0, 0.5, 0.5, 0.5, 0.2, 0.0)

    It can now be plotted:

    .. jupyter-execute::

        >>> rg.point_data["velocity"] = fc.get_velocity(0.0, rg.points)
        >>> rg.glyph(factor=0.5).plot(interactive=False)
    """

    center_x: float
    center_y: float
    center_z: float

    omega_x: float
    omega_y: float
    omega_z: float

    rho: float = 1.225

    def get_velocity(
        self,
        time: float,
        positions: npt.NDArray[np.float64],
        out_array: npt.NDArray[np.float64] | None = None,
    ) -> npt.NDArray[np.float64]:
        """Return velocity at the specified positions at given time.

        Parameters
        ----------
        time : float
            Time at which the velocity field should be computed.

        positions : (N, 3) array
            Array of positions where the flow field should be computed.

        out_array : (N, 3) array, optional
            If specified, this array should receive the output values, along with being
            returned by the function.

        Returns
        -------
        (N, 3) array
            Array of :math:`N` velocity vectors at the specified positions and time. If
            the parameter ``out_array`` was specified, it should also be the return value
            of this function.
        """
        del time, out_array
        pos = positions - np.array(
            ((self.center_x, self.center_y, self.center_z),), np.float64
        )
        omg = np.array((self.omega_x, self.omega_y, self.omega_z), np.float64)
        v = np.linalg.cross(pos, omg)
        return v

    def get_density(
        self,
        time: float,
        positions: npt.NDArray[np.float64],
        out_array: npt.NDArray[np.float64] | None = None,
    ) -> npt.NDArray[np.float64]:
        """Return density at the specified positions at given time.

        Parameters
        ----------
        time : float
            Time at which the density field should be computed.

        positions : (N, 3) array
            Array of positions where the flow field should be computed.

        out_array : (N,) array, optional
            If specified, this array should receive the output values, along with being
            returned by the function.

        Returns
        -------
        (N,) array
            Array of :math:`N` density values at the specified positions and time. If
            the parameter ``out_array`` was specified, it should also be the return value
            of this function.
        """
        del time
        if out_array is not None:
            out_array[:, :] = self.rho
            return out_array
        return np.full_like(positions, self.rho)

    def save(self) -> HirearchicalMap:
        """Serialize itself into a HirearchicalMap.

        Returns
        -------
        HirearchicalMap
            Serialized state of the :class:`FlowConditionsRotating` object.
        """
        hm = HirearchicalMap()
        hm.insert_array("center", (self.center_x, self.center_y, self.center_z))
        hm.insert_array("omega", (self.omega_x, self.omega_y, self.omega_z))
        return hm

    @classmethod
    def load(cls, hmap: HirearchicalMap) -> Self:
        """Deserialize from a HirearchicalMap.

        Parameters
        ----------
        hmap : HirearchicalMap
            Serialized state of the :class:`FlowConditionsRotating` object created by a
            call to :meth:`FlowConditionsRotating.save`.

        Returns
        -------
        Self
            Deserialized :class:`FlowConditionsRotating` object.
        """
        return cls(*(*hmap.get_array("center"), *hmap.get_array("omega")))
