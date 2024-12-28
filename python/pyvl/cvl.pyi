"""Typing file for C implemented functions/objects."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Self, final

import numpy as np
from numpy import typing as npt

from pyvl._typing import VecLike3
from pyvl.fio.io_common import HirearchicalMap

INVALID_ID: int = ...
"""Value of ID indicating an invalid object.

A line with a point with this ID does not have that end.
This occurs in dual meshes of open surfaces, where not all
lines are contained in two surfaces, thus their duals will
have only one valid dual point id in them.

Similarly, a line with this ID being in a surface indicates that
it is missing.

"""

@final
class GeoID:
    """Class used to refer to topological objects with orientation."""

    def __new__(cls, index: int, orientation: object = False) -> Self: ...
    @property
    def orientation(self) -> bool:
        """True if orientation of object is reversed."""
        ...

    @orientation.setter
    def orientation(self, o) -> None: ...
    @property
    def index(self) -> int:
        """Index of the object."""
        ...

    @index.setter
    def index(self) -> int: ...
    def __eq__(self, value) -> bool: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

@final
class Line:
    """Class which describes a connection between two points."""

    def __new__(cls, begin: int | GeoID, end: int | GeoID) -> Self: ...

    begin: int
    """Start point of the line."""

    end: int
    """End point of the line."""

    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...
    def __eq__(self, value) -> bool: ...

@final
class Surface:
    """Surface bound by a set of lines.

    If the end point of a line ``i`` is not the start point of the line
    ``i + 1``, a :class:`ValueError` will be raised.
    """

    def __new__(cls, lines: Sequence[Line]) -> Self: ...
    @property
    def n_lines(self) -> int:
        """Return the number of lines that make up the surface."""
        ...

    @property
    def lines(self) -> tuple[Line, ...]:
        """Return tuple of lines that make up the surface."""
        ...

    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

@final
class Mesh:
    """Object describing a discretization of a surface."""

    def __new__(
        cls,
        n_points: int,
        connectivity: Sequence[Sequence[int] | npt.ArrayLike],
    ) -> Self: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...
    @property
    def n_points(self) -> int:
        """Number of points in the mesh."""
        ...

    @property
    def n_lines(self) -> int:
        """Number of lines in the mesh."""
        ...

    @property
    def n_surfaces(self) -> int:
        """Number of surfaces in the mesh."""
        ...

    def get_line(self, i: int) -> Line:
        """Get the line from the mesh."""
        ...

    def get_surface(self, i: int) -> Surface:
        """Get the surface from the mesh."""
        ...

    def to_element_connectivity(
        self,
    ) -> tuple[npt.NDArray[np.uint64], npt.NDArray[np.uint64]]:
        """Convert mesh connectivity to arrays list of element lengths and indices."""
        ...

    def compute_dual(self) -> Mesh:
        """Create dual to the mesh."""
        ...

    def surface_normal(
        self, positions: npt.ArrayLike, out: npt.NDArray[np.float64] | None = None, /
    ) -> npt.NDArray[np.float64]:
        """Compute normals to surfaces based on point positions."""
        ...

    def surface_average_vec3(
        self, vectors: npt.ArrayLike, out: npt.NDArray[np.float64] | None = None, /
    ) -> npt.NDArray[np.float64]:
        """Compute average vec3 for each surface based on point values."""
        ...

    def induction_matrix(
        self,
        tol: float,
        positions: npt.NDArray[np.float64],
        control_points: npt.NDArray[np.float64],
        out: npt.NDArray[np.float64] | None = None,
        line_buffer: npt.NDArray[np.float64] | None = None,
        /,
    ) -> npt.NDArray[np.float64]:
        """Compute an induction matrix for the mesh."""
        ...

    def induction_matrix2(
        self,
        tol: float,
        positions: npt.NDArray[np.float64],
        control_points: npt.NDArray[np.float64],
        out: npt.NDArray[np.float64] | None = None,
        line_buffer: npt.NDArray[np.float64] | None = None,
        /,
    ) -> npt.NDArray[np.float64]:
        """Compute an induction matrix for the mesh using OpenACC."""
        ...

    def induction_matrix3(
        self,
        tol: float,
        positions: npt.NDArray[np.float64],
        control_points: npt.NDArray[np.float64],
        normals: npt.NDArray[np.float64],
        out: npt.NDArray[np.float64] | None = None,
        line_buffer: npt.NDArray[np.float64] | None = None,
        /,
    ) -> npt.NDArray[np.float64]:
        """Compute an induction matrix with normals included."""
        ...

    def line_velocities_from_point_velocities(
        self,
        point_velocities: npt.NDArray[np.float64],
        out: npt.NDArray[np.float64],
    ) -> None:
        """Compute line velocities by averaging velocities at its end nodes."""
        ...

    @classmethod
    def merge_meshes(cls, *meshes: Mesh) -> Self:
        """Merge multiple meshes into a single mesh."""
        ...

    def line_gradient(
        self,
        point_array: npt.NDArray[np.float64],
        line_array: npt.NDArray[np.float64] | None = None,
        /,
    ) -> npt.NDArray[np.float64]:
        """Compute line gradient from point values."""
        ...

    def dual_normal_criterion(
        self, crit: float, normals: npt.NDArray[np.float64], /
    ) -> npt.NDArray[np.uint]:
        """Find edges satisfying neighbouring normal dot product criterion."""
        ...

    def dual_free_edges(self, /) -> npt.NDArray[np.uint]:
        """Find edges with invalid nodes (dual free edges)."""
        ...

    @classmethod
    def from_lines(cls, n_points: int, connectivity: npt.ArrayLike) -> Self:
        """Create line-only mesh from line connectivity."""
        ...

    def line_induction_matrix(
        self,
        tol: float,
        positions: npt.NDArray[np.float64],
        control_points: npt.NDArray[np.float64],
        out: npt.NDArray[np.float64] | None = None,
        /,
    ) -> npt.NDArray[np.float64]:
        """Compute an induction matrix for the mesh based on line circulations."""
        ...

    @staticmethod
    def line_forces(
        primal: Mesh,
        dual: Mesh,
        circulation: npt.NDArray[np.float64],
        positions: npt.NDArray[np.float64],
        freestream: npt.NDArray[np.float64],
        out: npt.NDArray[np.float64] | None = None,
    ) -> npt.NDArray[np.float64]:
        r"""Compute forces due to reduced circulation filaments.

        Parameters
        ----------
        primal : Mesh
            Primal mesh.
        dual : Mesh
            Dual mesh, computed from the ``primal`` by a call to
            :meth:`Mesh.compute_dual()`.
        circulation : (N,) in_array
            Array of surface circulations divided by :math:`2 \pi`.
        positions : (M, 3) in_array
            Positions of the primal mesh nodes.
        freestream : (M, 3) in_array
            Free-stream velocity at the mesh nodes.
        out : (K, 3) out_array, optional
            Optional array where to write the results to. Assumed it does not alias memory
            from any other
            arrays.

        Returns
        -------
        (K, 3) out_array
            If ``out`` was given, it is returned as well. If not, the returned value is a
            newly allocated array of the correct size.
        """
        ...

class ReferenceFrame:
    r"""Class which is used to define position and orientation of geometry.

    This class represents a translation, followed by and orthonormal rotation. This
    transformation from a position vector :math:`\vec{r}` in the parent reference
    frame to a vector :math:`\vec{r}^\prime` in child reference frame can
    be written in four steps:

    .. math::

        \vec{r}_1 = \begin{bmatrix} 1 & 0 & 0 \\\ 0 & \cos\theta_x & \sin\theta_x \\\
        0 & -\sin\theta_x & \cos\theta_x \end{bmatrix} \vec{r}

    .. math::

        \vec{r}_2 = \begin{bmatrix} -\sin\theta_y & 0 & \cos\theta_y \\\ 0 & 1 & 0 \\\
        \cos\theta_y & 0 & \sin\theta_y \end{bmatrix} \vec{r}_1

    .. math::

        \vec{r}^3 = \begin{bmatrix} \cos\theta_z & \sin\theta_z & 0 \\\
        -\sin\theta_z & \cos\theta_z & 0 \\\ 0 & 0 & 1 \end{bmatrix} \vec{r}_2

    .. math::

        \vec{r}^\prime = \vec{r}_3 + \vec{d}

    Parameters
    ----------
    offset : VecLike3, default: (0, 0, 0)
        Position of the reference frame's origin expressed in the parent's reference
        frame.

    theta : VecLike3, default: (0, 0, 0)
        Rotation of the reference frame relative to its parent. The rotations are applied
        around the x, y, and z axis in that order.
    """

    def __new__(
        cls,
        offset: VecLike3 = (0, 0, 0),
        theta: VecLike3 = (0, 0, 0),
        parent: ReferenceFrame | None = None,
    ) -> Self: ...
    @property
    def rotation_matrix(self) -> npt.NDArray[np.float64]:
        """Matrix representing rotation of the reference frame."""
        ...
    @property
    def rotation_matrix_inverse(self) -> npt.NDArray[np.float64]:
        """Matrix representing rotation of the reference frame."""
        ...

    @property
    def angles(self) -> npt.NDArray[np.float64]:
        """Vector determining the rotations around axis in parent's frame."""
        ...

    @property
    def offset(self) -> npt.NDArray[np.float64]:
        """Vector determining the offset of the reference frame in parent's frame."""
        ...

    @property
    def parent(self) -> ReferenceFrame | None:
        """What frame it is relative to."""
        ...

    @property
    def parents(self) -> tuple[ReferenceFrame, ...]:
        """Tuple of all parents of this reference frame."""
        ...

    def from_parent_with_offset(
        self, r: npt.ArrayLike, out: npt.NDArray[np.float64] | None = None, /
    ) -> npt.NDArray[np.float64]:
        r"""Map position vector from parent reference frame to the child reference frame.

        Parameters
        ----------
        x : (N, 3) array
            Array of :math:`N` vectors in :math:`\mathbb{R}^3` in parent reference frame.
        out : (N, 3) array, optional"
            Array which receives the mapped vectors. Must have the exact shape of ``x``.
            It must also have the :class:`dtype` for :class:`numpy.double`, as well as be
            aligned, C-contiguous, and writable.

        Returns
        -------
        (N, 3) array
            Position vectors mapped to the child reference frame. If the ``out`` parameter
            was specified, this return value will be the same object. If ``out`` was not
            specified, then a new array will be allocated.
        """
        ...

    def from_parent_without_offset(
        self, r: npt.ArrayLike, out: npt.NDArray[np.float64] | None = None, /
    ) -> npt.NDArray[np.float64]:
        r"""Map direction vector from parent reference frame to the child reference frame.

        Parameters
        ----------
        x : (N, 3) array
            Array of :math:`N` vectors in :math:`\mathbb{R}^3` in parent reference frame.
        out : (N, 3) array, optional
            Array which receives the mapped vectors. Must have the exact shape of ``x``.
            It must also have the :class:`dtype` for :class:`numpy.double`, as well as be
            aligned, C-contiguous, and writable.

        Returns
        -------
        (N, 3) array
            Direction vectors mapped to the child reference frame. If the ``out``
            parameter was specified, this return value will be the same object. If ``out``
            was not specified, then a new array will be allocated.
        """
        ...

    def to_parent_with_offset(
        self, r: npt.ArrayLike, out: npt.NDArray[np.float64] | None = None, /
    ) -> npt.NDArray[np.float64]:
        r"""Map position vector from child reference frame to the parent reference frame.

        Parameters
        ----------
        x : (N, 3) array
            Array of :math:`N` vectors in :math:`\mathbb{R}^3` in child reference frame.
        out : (N, 3) array, optional"
            Array which receives the mapped vectors. Must have the exact shape of ``x``.
            It must also have the :class:`dtype` for :class:`numpy.double`, as well as be
            aligned, C-contiguous, and writable.

        Returns
        -------
        (N, 3) array
            Position vectors mapped to the parent reference frame. If the ``out``
            parameter was specified, this return value will be the same object. If
            ``out`` was not specified, then a new array will be allocated.
        """
        ...

    def to_parent_without_offset(
        self, r: npt.ArrayLike, out: npt.NDArray[np.float64] | None = None, /
    ) -> npt.NDArray[np.float64]:
        r"""Map direction vector from child reference frame to the parent reference frame.

        Parameters
        ----------
        x : (N, 3) array
            Array of :math:`N` vectors in :math:`\mathbb{R}^3` in child reference frame.
        out : (N, 3) array, optional
            Array which receives the mapped vectors. Must have the exact shape of ``x``.
            It must also have the :class:`dtype` for :class:`numpy.double`, as well as be
            aligned, C-contiguous, and writable.

        Returns
        -------
        (N, 3) array
            Direction vectors mapped to the parent reference frame. If the ``out``
            parameter was specified, this return value will be the same object. If ``out``
            was not specified, then a new array will be allocated.
        """
        ...

    def from_global_with_offset(
        self, r: npt.ArrayLike, out: npt.NDArray[np.float64] | None = None, /
    ) -> npt.NDArray[np.float64]:
        r"""Map position vector from global reference frame to the child reference frame.

        Parameters
        ----------
        x : (N, 3) array
            Array of :math:`N` vectors in :math:`\mathbb{R}^3` in global reference frame.
        out : (N, 3) array, optional"
            Array which receives the mapped vectors. Must have the exact shape of ``x``.
            It must also have the :class:`dtype` for :class:`numpy.double`, as well as be
            aligned, C-contiguous, and writable.

        Returns
        -------
        (N, 3) array
            Position vectors mapped to the child reference frame. If the ``out``
            parameter was specified, this return value will be the same object. If ``out``
            was not specified, then a new array will be allocated."
        """
        ...

    def from_global_without_offset(
        self, r: npt.ArrayLike, out: npt.NDArray[np.float64] | None = None, /
    ) -> npt.NDArray[np.float64]:
        r"""Map direction vector from global reference frame to the child reference frame.

        Parameters
        ----------
        x : (N, 3) array
            Array of :math:`N` vectors in :math:`\mathbb{R}^3` in global reference frame.
        out : (N, 3) array, optional
            Array which receives the mapped vectors. Must have the exact shape of ``x``.
            It must also have the :class:`dtype` for :class:`numpy.double`, as well as be
            aligned, C-contiguous, and writable.

        Returns
        -------
        (N, 3) array
            Direction vectors mapped to the child reference frame. If the ``out``
            parameter was specified, this return value will be the same object. If ``out``
            was not specified, then a new array will be allocated."
        """
        ...

    def to_global_with_offset(
        self, r: npt.ArrayLike, out: npt.NDArray[np.float64] | None = None, /
    ) -> npt.NDArray[np.float64]:
        r"""Map position vector from child reference frame to the parent reference frame.

        Parameters
        ----------
        x : (N, 3) array
            Array of :math:`N` vectors in :math:`\mathbb{R}^3` in child reference frame.
        out : (N, 3) array, optional"
            Array which receives the mapped vectors. Must have the exact shape of ``x``.
            It must also have the :class:`dtype` for :class:`numpy.double`, as well as be
            aligned, C-contiguous, and writable.

        Returns
        -------
        (N, 3) array
            Position vectors mapped to the global reference frame. If the ``out``
            parameter was specified, this return value will be the same object. If ``out``
            was not specified, then a new array will be allocated."
        """
        ...

    def to_global_without_offset(
        self, r: npt.ArrayLike, out: npt.NDArray[np.float64] | None = None, /
    ) -> npt.NDArray[np.float64]:
        r"""Map direction vector from child reference frame to the global reference frame.

        Parameters
        ----------
        x : (N, 3) array
            Array of :math:`N` vectors in :math:`\mathbb{R}^3` in child reference frame.
        out : (N, 3) array, optional
            Array which receives the mapped vectors. Must have the exact shape of ``x``.
            It must also have the :class:`dtype` for :class:`numpy.double`, as well as be
            aligned, C-contiguous, and writable.

        Returns
        -------
        (N, 3) array
            Direction vectors mapped to the global reference frame. If the ``out``
            parameter was specified, this return value will be the same object. If ``out``
            was not specified, then a new array will be allocated."
        """
        ...

    def rotate_x(self, theta: float) -> ReferenceFrame:
        """Create a copy of the frame rotated around the x-axis.

        Parameters
        ----------
        theta_x : float
            Angle by which to rotate the reference frame by.

        Returns
        -------
        Self
            Reference frame rotated around the x-axis by the specified angle.
        """
        ...

    def rotate_y(self, theta: float) -> ReferenceFrame:
        """Create a copy of the frame rotated around the y-axis.

        Parameters
        ----------
        theta_y : float
            Angle by which to rotate the reference frame by.

        Returns
        -------
        Self
            Reference frame rotated around the y-axis by the specified angle.
        """
        ...

    def rotate_z(self, theta: float) -> ReferenceFrame:
        """Create a copy of the frame rotated around the z-axis.

        Parameters
        ----------
        theta_z : float
            Angle by which to rotate the reference frame by.

        Returns
        -------
        Self
            Reference frame rotated around the z-axis by the specified angle.
        """
        ...

    def with_offset(self, new_offset: npt.ArrayLike) -> ReferenceFrame:
        """Create a copy of the frame with different offset value.

        Parameters
        ----------
        offset : VecLike3
            Offset to add to the reference frame relative to its parent.

        Returns
        -------
        ReferenceFrame
            A copy of itself which is translated by the value of ``offset`` in
            the parent's reference frame.
        """
        ...

    def at_time(self, t: float) -> Self:
        """Compute reference frame at the given time.

        This is used when the reference frame is moving or rotating in space.

        Parameters
        ----------
        t : float
            Time at which the reference frame is needed.

        Returns
        -------
        Self
            New reference frame at the given time.
        """
        ...

    @staticmethod
    def angles_from_rotation(rotation_matrix: npt.ArrayLike) -> npt.NDArray[np.double]:
        """Compute rotation angles from a transformation matrix.

        Parameters
        ----------
        mat : (3, 3) array
            Rotation matrix to convert to the rotation angles. This is done assuming that
            the matrix is orthogonal.

        Returns
        -------
        (3,) array"
            Rotation angles around the x-, y-, and z-axis which result in a transformation
            with equal rotation matrix.
        """
        ...

    def save(self, hmap: HirearchicalMap, /) -> None:
        """Serialize the ReferenceFrame into a HirearchicalMap.

        Parameters
        ----------
        hmap: HirearchicalMap
            :class:`HirearchicalMap` in which to save the reference frame into.
        """
        ...

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
        ...
