"""Typing file for C implemented functions/objects."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Self, final

import numpy as np
from numpy import typing as npt

from pyvl._typing import CallableDeserializer, CallableSerializer, Vec3Callable, VecLike3
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
    def __hash__(self) -> int: ...
    def __neg__(self) -> GeoID: ...

_GeoIDLike = GeoID | int

@final
class TransformationPlane:
    """Type used to describe a plane used for a transformation.

    Parameters
    ----------
    origin : VecLike3 or Callable, default: (0, 0, 0)
        Origin of the plane. Can be a constant vector or a callable returning the origin
        at time t.

    normal : VecLike3 or Callable, default: (0, 0, 1)
        Normal of the plane. Can be a constant vector or a callable returning the normal
        at time t. Does not need to be normalized, but it will be internally.
    """

    def __new__(
        cls,
        origin: VecLike3 | Vec3Callable = (0, 0, 0),
        normal: VecLike3 | Vec3Callable = (0, 0, 1),
    ) -> Self: ...
    def origin(self, t: float = 0) -> npt.NDArray[np.double]:
        """Get the origin of the plane at the given time.

        Parameters
        ----------
        t : float, default: 0
            Time at which to evaluate the origin.

        Returns
        -------
        (3,) array
            Origin vector at the given time.
        """
        ...

    def normal(self, t: float = 0) -> npt.NDArray[np.double]:
        """Get the normal of the plane at the given time.

        Parameters
        ----------
        t : float, default: 0
            Time at which to evaluate the normal.

        Returns
        -------
        (3,) array
            Normal vector at the given time.
        """
        ...

    def reflect(
        self, x: npt.ArrayLike, t: float = 0, out: npt.NDArray[np.double] | None = None
    ) -> npt.NDArray[np.double]:
        """Reflect points across the plane.

        Parameters
        ----------
        x : array
            Array of points to reflect. Must be an aligned, continuous (N, 3) array,
            where N is the number of points.

        t : float, default: 0
            Time at which to evaluate the plane's position and orientation.

        out : array, optional
            Array used to store the output. If not given or ``None``, a new array will
            be created.

        Returns
        -------
        array
            Reflected points. If ``out`` was not ``None``, a reference to it is returned,
        otherwise a new array is returned.
        """
        ...

    def at_time(self, t: float) -> TransformationPlane:
        """Get the plane at the given time.

        For planes with constant origin and normal, this will return the same plane. For
        planes with time-dependent origin and/or normal, this will return a new plane
        with the origin and normal evaluated at the given time.

        Parameters
        ----------
        t : float
            Time at which to evaluate the plane's position and orientation.

        Returns
        -------
        TransformationPlane
            New plane at the given time.
        """
        ...

@final
class Mesh:
    """Object describing a discretization of a surface."""

    def __new__(
        cls,
        n_points: int,
        connectivity: Sequence[Sequence[_GeoIDLike] | npt.ArrayLike],
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

    def get_line_points(self, i: _GeoIDLike) -> tuple[int, int]:
        """Get the indices of points that make up the line from the mesh.

        Parameters
        ----------
        i : GeoID or int
            ID of the line to get the points of. If an int is given, negative value
            means a reverse orientation.

        Returns
        -------
        int
            Index of the point at the start of the line.

        int
            Index of the point at the end of the line.
        """
        ...

    def get_surface_lines(self, i: _GeoIDLike) -> tuple[GeoID, ...]:
        """Get IDs of lines that make up the surface from the mesh.

        Parameters
        ----------
        i : GeoID or int
            ID of the surface to get the lines of. If an int is given, negative value
            means a reverse orientation.

        Returns
        -------
        tuple[GeoID, ...]
            Tuple of IDs of lines that make up the surface. When reversed orientation is
            requested, the order and orientation of the lines is reversed as well.
        """
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
        self, positions: npt.ArrayLike, out: npt.NDArray[np.double] | None = None, /
    ) -> npt.NDArray[np.double]:
        """Compute normals to surfaces based on point positions."""
        ...

    def surface_average_vec3(
        self, vectors: npt.ArrayLike, out: npt.NDArray[np.double] | None = None, /
    ) -> npt.NDArray[np.double]:
        """Compute average vec3 for each surface based on point values."""
        ...

    def induction_matrix(
        self,
        tol: float,
        positions: npt.NDArray[np.double],
        control_points: npt.NDArray[np.double],
        symmetry_plane: TransformationPlane | None = None,
        out: npt.NDArray[np.double] | None = None,
        line_buffer: npt.NDArray[np.double] | None = None,
        thread_count: int = 1,
    ) -> npt.NDArray[np.double]:
        """Compute an induction matrix for the mesh."""
        ...

    def induction_matrix3(
        self,
        tol: float,
        positions: npt.NDArray[np.double],
        control_points: npt.NDArray[np.double],
        normals: npt.NDArray[np.double],
        symmetry_plane: TransformationPlane | None = None,
        out: npt.NDArray[np.double] | None = None,
        line_buffer: npt.NDArray[np.double] | None = None,
        thread_count: int = 1,
    ) -> npt.NDArray[np.double]:
        """Compute an induction matrix with normals included."""
        ...

    def line_circulations(
        self,
        circulation: npt.NDArray[np.double],
        out: npt.NDArray[np.double] | None = None,
        n_threads: int = 1,
    ) -> npt.NDArray[np.double]:
        """Compute circulations based of lines using the mesh.

        Parameters
        ----------
        circulation : array
            Array of surface circulation values. Must match the number of points in the
            mesh.

        out : array, optional
            Array used to store the output. If not given or ``None``, a new array will
            be created.

        n_threads : int, default: 1
            Number of threads to use for this calculation.

        Returns
        -------
        array
            Line circulation values. If ``out`` was not ``None``, a reference to it is
            returned, otherwise a new array is returned.
        """
        ...

    def induction_velocity(
        self,
        tol: float,
        positions: npt.NDArray[np.double],
        control_points: npt.NDArray[np.double],
        line_circulation: npt.NDArray[np.double],
        symmetry_plane: TransformationPlane | None = None,
        out: npt.NDArray[np.double] | None = None,
        n_threads: int = 1,
    ) -> npt.NDArray[np.double]:
        """Compute velocity induced by mesh circulation.

        Parameters
        ----------
        tol : float
            Minimum distance before the induced velocity is clamped to zero.

        positions : array
            Positions of the geometry points. Must be an aligned, continuous (N, 3) array,
            where N is the number of points.

        control_points : array
            An (M, 3) array, which specifies the positions of M points.

        line_circulation : array
            Array of circulations for each of the lines.

        out : array, optional
            An array with enough space for M velocity vectors, one for
            each of the control points.

        symmetry_plane : TransformationPlane, optional
            A plane of symmetry to consider in the induction calculation.

        n_threads : int, default: 1
            Number of threads to use for computing the induction.

        Returns
        -------
        array
            Resulting induction vectors in an array. If ``out`` was given, the result is
            written to it and another reference to it returned, otherwise a new array is
            created.
        """
        ...

    def line_velocities_from_point_velocities(
        self,
        point_velocities: npt.NDArray[np.double],
        out: npt.NDArray[np.double],
    ) -> None:
        """Compute line velocities by averaging velocities at its end nodes."""
        ...

    @classmethod
    def merge_meshes(cls, *meshes: Mesh) -> Self:
        """Merge multiple meshes into a single mesh."""
        ...

    def line_gradient(
        self,
        point_array: npt.NDArray[np.double],
        line_array: npt.NDArray[np.double] | None = None,
        /,
    ) -> npt.NDArray[np.double]:
        """Compute line gradient from point values."""
        ...

    def dual_normal_criterion(
        self, crit: float, normals: npt.NDArray[np.double], /
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

    def line_forces(
        self,
        line_circulation: npt.NDArray[np.double],
        positions: npt.NDArray[np.double],
        velocity: npt.NDArray[np.double],
        out: npt.NDArray[np.double] | None = None,
    ) -> npt.NDArray[np.double]:
        r"""Compute forces due to reduced circulation filaments.

        Parameters
        ----------
        line_circulation : (N,) in_array
            Array of line circulations divided by :math:`2 \pi`.

        positions : (M, 3) in_array
            Positions of the primal mesh nodes.

        velocity : (M, 3) in_array
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

    @property
    def line_data(self) -> npt.NDArray[np.uint]:
        """Line connectivity of the mesh."""
        ...

    @classmethod
    def make_quad2d_plane(cls, n1: int, n2: int, /) -> Self:
        """Make a simple quad mesh of a 2D plane topology.

        Parameters
        ----------
        n1 : int
            Number of quads in the first direction.

        n2 : int
            Number of quads in the second direction.

        Returns
        -------
        Self
            Newly created mesh instance.
        """
        ...

@final
class ReferenceFrame:
    r"""Class which is used to define position and orientation of geometry.

    Each of the position, velocity, orientation, and rotation can be either a constant
    vector or a callable with signature ``(float) -> (float, float, float)``. Callables
    are evaluated at the given time to determine the current transformation.

    While denoting the position of the reference frame as :math:`\vec{r}(t)`, velocity as
    :math:`\vec{v}(t)`, the orientation matrix as :math:`\mathbf{T}(t)`, and its angular
    velocity as :math:`\vec{\omega}(t)`, the position and velocity relative to its parent,
    denoted by :math:`\vec{r}_\mathrm{parent}(t)` and :math:`\vec{v}_\mathrm{parent}(t)`,
    for a point at :math:`\vec{r}_P` with velocity :math:`\vec{v}_P` are given by

    .. math::

        \vec{r}_\mathrm{parent}(t) = \mathbf{T}(t) \left( \vec{r}_P + \vec{r}(t) \right)

    and

    .. math::

        \vec{v}_\mathrm{parent}(t) = \mathbf{T}(t) \left( \vec{v}_P + \vec{v} + \vec{r}(t)
        \times \vec{\omega}(t) \right)

    Of course it is also possible to transform any other quantity just with the
    orientation matrix :math:`\mathbf{T}`.

    Parameters
    ----------
    offset : VecLike3 or Callable, default: (0, 0, 0)
        Position of the reference frame's origin expressed in the parent's reference
        frame. Can be a constant vector or a callable returning the position at time t.

    theta : VecLike3 or Callable, default: (0, 0, 0)
        Rotation of the reference frame relative to its parent. The rotations are applied
        around the x, y, and z axis in that order. Can be a constant vector or a callable
        returning the orientation (Euler angles) at time t.

    velocity : VecLike3 or Callable, default: (0, 0, 0)
        Linear velocity of the reference frame. Can be a constant vector or a callable
        returning the velocity at time t.

    rotation : VecLike3 or Callable, default: (0, 0, 0)
        Angular velocity (rotation rate) of the reference frame. Can be a constant vector
        or a callable returning the rotation rate at time t.

    parent : ReferenceFrame, optional
        Parent reference frame.
    """

    def __new__(
        cls,
        offset: VecLike3 | Vec3Callable = (0, 0, 0),
        theta: VecLike3 | Vec3Callable = (0, 0, 0),
        velocity: VecLike3 | Vec3Callable = (0, 0, 0),
        rotation: VecLike3 | Vec3Callable = (0, 0, 0),
        parent: ReferenceFrame | None = None,
    ) -> Self: ...
    @property
    def parent(self) -> ReferenceFrame | None:
        """What frame it is relative to."""
        ...

    @property
    def parents(self) -> tuple[ReferenceFrame, ...]:
        """Tuple of all parents of this reference frame."""
        ...

    def offset_at(self, t: float = 0.0, /) -> npt.NDArray[np.double]:
        """Get the position of the reference frame at the given time.

        Parameters
        ----------
        t : float, default: 0.0
            Time at which to evaluate the position.

        Returns
        -------
        (3,) array
            Position vector at the given time.
        """
        ...

    def velocity_at(self, t: float = 0.0, /) -> npt.NDArray[np.double]:
        """Get the linear velocity of the reference frame at the given time.

        Parameters
        ----------
        t : float, default: 0.0
            Time at which to evaluate the velocity.

        Returns
        -------
        (3,) array
            Velocity vector at the given time.
        """
        ...

    def angles_at(self, t: float = 0.0, /) -> npt.NDArray[np.double]:
        """Get the orientation (Euler angles) of the reference frame at the given time.

        Parameters
        ----------
        t : float, default: 0.0
            Time at which to evaluate the orientation.

        Returns
        -------
        (3,) array
            Euler angles at the given time.
        """
        ...

    def rotation_at(self, t: float = 0.0, /) -> npt.NDArray[np.double]:
        """Get the angular velocity at the given time.

        Parameters
        ----------
        t : float, default: 0.0
            Time at which to evaluate the rotation.

        Returns
        -------
        (3,) array
            Angular velocity vector at the given time.
        """
        ...

    def rotation_matrix_at(self, t: float = 0.0, /) -> npt.NDArray[np.double]:
        """Get the rotation matrix of the reference frame at the given time.

        Parameters
        ----------
        t : float, default: 0.0
            Time at which to evaluate the rotation matrix.

        Returns
        -------
        (3, 3) array
            Rotation matrix at the given time.
        """
        ...

    def rotation_matrix_inverse_at(self, t: float = 0.0, /) -> npt.NDArray[np.double]:
        """Get the inverse rotation matrix of the reference frame at the given time.

        Parameters
        ----------
        t : float, default: 0.0
            Time at which to evaluate the inverse rotation matrix.

        Returns
        -------
        (3, 3) array
            Inverse rotation matrix at the given time.
        """
        ...

    def from_parent_position(
        self,
        x: npt.ArrayLike,
        time: float = 0.0,
        out: npt.NDArray[np.double] | None = None,
    ) -> npt.NDArray[np.double]:
        r"""Map position vector from parent reference frame to the child reference frame.

        Parameters
        ----------
        x : (N, 3) array
            Array of :math:`N` vectors in :math:`\mathbb{R}^3` in parent reference frame.
        time : float, default: 0.0
            Time at which to evaluate the transformation.
        out : (N, 3) array, optional
            Array which receives the mapped vectors. Must have the exact shape of ``x``.
            It must also have the :class:`dtype` for :class:`numpy.double`, as well as be
            aligned, C-contiguous, and writable.

        Returns
        -------
        (N, 3) array
            Position vectors mapped to the child reference frame. If the ``out``
            parameter was specified, this return value will be the same object. If ``out``
            was not specified, then a new array will be allocated.
        """
        ...

    def from_parent_velocity(
        self,
        position: npt.ArrayLike,
        velocity: npt.ArrayLike,
        time: float = 0.0,
        out_position: npt.NDArray[np.double] | None = None,
        out_velocity: npt.NDArray[np.double] | None = None,
    ) -> tuple[npt.NDArray[np.double], npt.NDArray[np.double]]:
        r"""Map velocity vectors from parent reference frame to the local reference frame.

        Parameters
        ----------
        position : (N, 3) array
            Array of :math:`N` position vectors in :math:`\mathbb{R}^3` in parent
            reference frame.

        velocity : (N, 3) array
            Array of :math:`N` velocity vectors in :math:`\mathbb{R}^3` in parent
            reference frame.

        time : float, default: 0.0
            Time at which to evaluate the transformation.

        out_position : (N, 3) array, optional
            Array which receives the mapped position vectors. Must have the exact shape of
            ``position``. It must also have the :class:`dtype` for :class:`numpy.double`,
            as well as be aligned, C-contiguous, and writable.

        out_velocity : (N, 3) array, optional
            Array which receives the mapped velocity vectors. Must have the exact shape of
            ``velocity``. It must also have the :class:`dtype` for :class:`numpy.double`,
            as well as be aligned, C-contiguous, and writable.

        Returns
        -------
        (N, 3) array
            Position vectors mapped to the local reference frame. If the ``out_position``
            parameter was specified, this return value will be the same object. If
            ``out_position`` was not specified, then a new array will be allocated.

        (N, 3) array
            Velocity vectors mapped to the local reference frame. If the ``out_velocity``
            parameter was specified, this return value will be the same object. If
            ``out_velocity`` was not specified, then a new array will be allocated.
        """
        ...

    def from_parent_vector(
        self,
        x: npt.ArrayLike,
        time: float = 0.0,
        out: npt.NDArray[np.double] | None = None,
    ) -> npt.NDArray[np.double]:
        r"""Map direction vector from parent reference frame to the child reference frame.

        Parameters
        ----------
        x : (N, 3) array
            Array of :math:`N` vectors in :math:`\mathbb{R}^3` in parent reference frame.
        time : float, default: 0.0
            Time at which to evaluate the transformation.
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

    def to_parent_position(
        self,
        x: npt.ArrayLike,
        time: float = 0.0,
        out: npt.NDArray[np.double] | None = None,
    ) -> npt.NDArray[np.double]:
        r"""Map position vector from child reference frame to the parent reference frame.

        Parameters
        ----------
        x : (N, 3) array
            Array of :math:`N` vectors in :math:`\mathbb{R}^3` in child reference frame.
        time : float, default: 0.0
            Time at which to evaluate the transformation.
        out : (N, 3) array, optional
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

    def to_parent_velocity(
        self,
        position: npt.ArrayLike,
        velocity: npt.ArrayLike,
        time: float = 0.0,
        out_position: npt.NDArray[np.double] | None = None,
        out_velocity: npt.NDArray[np.double] | None = None,
    ) -> tuple[npt.NDArray[np.double], npt.NDArray[np.double]]:
        r"""Map velocity vectors from local reference frame to the parent reference frame.

        Parameters
        ----------
        position : (N, 3) array
            Array of :math:`N` position vectors in :math:`\mathbb{R}^3` in local reference
            frame.

        velocity : (N, 3) array
            Array of :math:`N` velocity vectors in :math:`\mathbb{R}^3` in local reference
            frame.

        time : float, default: 0.0
            Time at which to evaluate the transformation.

        out_position : (N, 3) array, optional
            Array which receives the mapped position vectors. Must have the exact shape of
            ``position``. It must also have the :class:`dtype` for :class:`numpy.double`,
            as well as be aligned, C-contiguous, and writable.

        out_velocity : (N, 3) array, optional
            Array which receives the mapped velocity vectors. Must have the exact shape of
            ``velocity``. It must also have the :class:`dtype` for :class:`numpy.double`,
            as well as be aligned, C-contiguous, and writable.

        Returns
        -------
        (N, 3) array
            Position vectors mapped to the parent reference frame. If the ``out_position``
            parameter was specified, this return value will be the same object. If
            ``out_position`` was not specified, then a new array will be allocated.

        (N, 3) array
            Velocity vectors mapped to the parent reference frame. If the ``out_velocity``
            parameter was specified, this return value will be the same object. If
            ``out_velocity`` was not specified, then a new array will be allocated.
        """
        ...

    def to_parent_vector(
        self,
        x: npt.ArrayLike,
        time: float = 0.0,
        out: npt.NDArray[np.double] | None = None,
    ) -> npt.NDArray[np.double]:
        r"""Map direction vector from child reference frame to the parent reference frame.

        Parameters
        ----------
        x : (N, 3) array
            Array of :math:`N` vectors in :math:`\mathbb{R}^3` in child reference frame.
        time : float, default: 0.0
            Time at which to evaluate the transformation.
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

    def from_global_position(
        self,
        x: npt.ArrayLike,
        time: float = 0.0,
        out: npt.NDArray[np.double] | None = None,
    ) -> npt.NDArray[np.double]:
        r"""Map position vector from global reference frame to the child reference frame.

        Parameters
        ----------
        x : (N, 3) array
            Array of :math:`N` vectors in :math:`\mathbb{R}^3` in global reference frame.
        time : float, default: 0.0
            Time at which to evaluate the transformation.
        out : (N, 3) array, optional
            Array which receives the mapped vectors. Must have the exact shape of ``x``.
            It must also have the :class:`dtype` for :class:`numpy.double`, as well as be
            aligned, C-contiguous, and writable.

        Returns
        -------
        (N, 3) array
            Position vectors mapped to the child reference frame. If the ``out``
            parameter was specified, this return value will be the same object. If ``out``
            was not specified, then a new array will be allocated.
        """
        ...

    def from_global_velocity(
        self,
        position: npt.ArrayLike,
        velocity: npt.ArrayLike,
        time: float = 0.0,
        out_position: npt.NDArray[np.double] | None = None,
        out_velocity: npt.NDArray[np.double] | None = None,
    ) -> tuple[npt.NDArray[np.double], npt.NDArray[np.double]]:
        r"""Map velocity vectors from global reference frame to the local reference frame.

        Parameters
        ----------
        position : (N, 3) array
            Array of :math:`N` position vectors in :math:`\mathbb{R}^3` in global
            reference frame.

        velocity : (N, 3) array
            Array of :math:`N` velocity vectors in :math:`\mathbb{R}^3` in global
            reference frame.

        time : float, default: 0.0
            Time at which to evaluate the transformation.

        out_position : (N, 3) array, optional
            Array which receives the mapped position vectors. Must have the exact shape of
            ``position``. It must also have the :class:`dtype` for :class:`numpy.double`,
            as well as be aligned, C-contiguous, and writable.

        out_velocity : (N, 3) array, optional
            Array which receives the mapped velocity vectors. Must have the exact shape of
            ``velocity``. It must also have the :class:`dtype` for :class:`numpy.double`,
            as well as be aligned, C-contiguous, and writable.

        Returns
        -------
        (N, 3) array
            Position vectors mapped to the local reference frame. If the ``out_position``
            parameter was specified, this return value will be the same object. If
            ``out_position`` was not specified, then a new array will be allocated.

        (N, 3) array
            Velocity vectors mapped to the local reference frame. If the ``out_velocity``
            parameter was specified, this return value will be the same object. If
            ``out_velocity`` was not specified, then a new array will be allocated.
        """
        ...

    def from_global_vector(
        self,
        x: npt.ArrayLike,
        time: float = 0.0,
        out: npt.NDArray[np.double] | None = None,
    ) -> npt.NDArray[np.double]:
        r"""Map direction vector from global reference frame to the child reference frame.

        Parameters
        ----------
        x : (N, 3) array
            Array of :math:`N` vectors in :math:`\mathbb{R}^3` in global reference frame.
        time : float, default: 0.0
            Time at which to evaluate the transformation.
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

    def to_global_position(
        self,
        x: npt.ArrayLike,
        time: float = 0.0,
        out: npt.NDArray[np.double] | None = None,
    ) -> npt.NDArray[np.double]:
        r"""Map position vector from child reference frame to the global reference frame.

        Parameters
        ----------
        x : (N, 3) array
            Array of :math:`N` vectors in :math:`\mathbb{R}^3` in child reference frame.
        time : float, default: 0.0
            Time at which to evaluate the transformation.
        out : (N, 3) array, optional
            Array which receives the mapped vectors. Must have the exact shape of ``x``.
            It must also have the :class:`dtype` for :class:`numpy.double`, as well as be
            aligned, C-contiguous, and writable.

        Returns
        -------
        (N, 3) array
            Position vectors mapped to the global reference frame. If the ``out``
            parameter was specified, this return value will be the same object. If ``out``
            was not specified, then a new array will be allocated.
        """
        ...

    def to_global_velocity(
        self,
        position: npt.ArrayLike,
        velocity: npt.ArrayLike,
        time: float = 0.0,
        out_position: npt.NDArray[np.double] | None = None,
        out_velocity: npt.NDArray[np.double] | None = None,
    ) -> tuple[npt.NDArray[np.double], npt.NDArray[np.double]]:
        r"""Map velocity vectors from local reference frame to the global reference frame.

        Parameters
        ----------
        position : (N, 3) array
            Array of :math:`N` position vectors in :math:`\mathbb{R}^3` in local reference
            frame.

        velocity : (N, 3) array
            Array of :math:`N` velocity vectors in :math:`\mathbb{R}^3` in local reference
            frame.

        time : float, default: 0.0
            Time at which to evaluate the transformation.

        out_position : (N, 3) array, optional
            Array which receives the mapped position vectors. Must have the exact shape of
            ``position``. It must also have the :class:`dtype` for :class:`numpy.double`,
            as well as be aligned, C-contiguous, and writable.

        out_velocity : (N, 3) array, optional
            Array which receives the mapped velocity vectors. Must have the exact shape of
            ``velocity``. It must also have the :class:`dtype` for :class:`numpy.double`,
            as well as be aligned, C-contiguous, and writable.

        Returns
        -------
        (N, 3) array
            Position vectors mapped to the global reference frame. If the ``out_position``
            parameter was specified, this return value will be the same object. If
            ``out_position`` was not specified, then a new array will be allocated.

        (N, 3) array
            Velocity vectors mapped to the global reference frame. If the ``out_velocity``
            parameter was specified, this return value will be the same object. If
            ``out_velocity`` was not specified, then a new array will be allocated.
        """
        ...

    def to_global_vector(
        self,
        x: npt.ArrayLike,
        time: float = 0.0,
        out: npt.NDArray[np.double] | None = None,
    ) -> npt.NDArray[np.double]:
        r"""Map direction vector from child reference frame to the global reference frame.

        Parameters
        ----------
        x : (N, 3) array
            Array of :math:`N` vectors in :math:`\mathbb{R}^3` in child reference frame.
        time : float, default: 0.0
            Time at which to evaluate the transformation.
        out : (N, 3) array, optional
            Array which receives the mapped vectors. Must have the exact shape of ``x``.
            It must also have the :class:`dtype` for :class:`numpy.double`, as well as be
            aligned, C-contiguous, and writable.

        Returns
        -------
        (N, 3) array
            Direction vectors mapped to the global reference frame. If the ``out``
            parameter was specified, this return value will be the same object. If ``out``
            was not specified, then a new array will be allocated.
        """
        ...

    def rotate_x(self, theta: float) -> ReferenceFrame:
        """Create a copy of the frame rotated around the x-axis.

        Only for constant orientation. Raises TypeError if time-varying.

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

        Only for constant orientation. Raises TypeError if time-varying.

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

        Only for constant orientation. Raises TypeError if time-varying.

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

        Only works for constant position. Raises TypeError if position is time-varying.

        Parameters
        ----------
        offset : VecLike3
            Offset to set for the reference frame relative to its parent.

        Returns
        -------
        ReferenceFrame
            A copy of itself with the specified offset in the parent's reference frame.
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
        (3,) array
            Rotation angles around the x-, y-, and z-axis which result in a transformation
            with equal rotation matrix.
        """
        ...

    def save(self, hmap: HirearchicalMap, serializer: CallableSerializer) -> None:
        """Serialize the ReferenceFrame into a HirearchicalMap.

        Parameters
        ----------
        hmap: HirearchicalMap
            :class:`HirearchicalMap` in which to save the reference frame into.

        serializer: CallableSerializer
            Callable that is used to convert input callables into strings.
        """
        ...

    @classmethod
    def load(
        cls,
        group: HirearchicalMap,
        deserializer: CallableDeserializer,
        parent: ReferenceFrame | None = None,
    ) -> Self:
        """Load the ReferenceFrame from a HirearchicalMap.

        Parameters
        ----------
        hmap : HirearchicalMap
            A :class:`HirearchicalMap`, which was created with a call to
            :meth:`ReferenceFrame.save`.

        deserializer: CallableDeserializer
            Callable that is used to convert strings into callables.

        parent : ReferenceFrame, optional
            Parent of the reference frame.

        Returns
        -------
        Self
            Deserialized :class:`ReferenceFrame`.
        """
        ...

    @classmethod
    def transform_position(
        cls,
        x: npt.ArrayLike,
        start: ReferenceFrame | None = None,
        end: ReferenceFrame | None = None,
        time: float = 0.0,
        out: npt.NDArray[np.double] | None = None,
    ) -> npt.NDArray[np.double]:
        """Transform position vectors from one reference frame to another.

        Parameters
        ----------
        x : array_like
            Position vectors to transform.

        start : ReferenceFrame, optional
            Reference frame the position vectors are given in. If not given, the global
            reference frame is assumed.

        end : ReferenceFrame, optional
            Reference frame the resulting vectors should be given in. If not given, the
            global reference frame is assumed.

        time : float, default: 0.0
            What time the transformations should be taken at. Only relevant if the
            reference frames have time-dependant motion.

        out : array, optional
            Output array to write the output to. If not given a new one is created.

        Returns
        -------
        array
            Array of position vectors. If ``out`` was given, then the reference to it
            is returned.
        """
        ...

    @classmethod
    def transform_velocity(
        cls,
        position: npt.ArrayLike,
        velocity: npt.ArrayLike,
        start: ReferenceFrame | None = None,
        end: ReferenceFrame | None = None,
        time: float = 0.0,
        out_position: npt.NDArray[np.double] | None = None,
        out_velocity: npt.NDArray[np.double] | None = None,
    ) -> tuple[npt.NDArray[np.double], npt.NDArray[np.double]]:
        r"""Transform position and velocity vectors from one reference frame to another.

        Parameters
        ----------
        position : array_like
            Position vectors to transform.

        velocity : array_like
            Velocity vectors to transform.

        start : ReferenceFrame, optional
            Reference frame the position vectors are given in. If not given, the global
            reference frame is assumed.

        end : ReferenceFrame, optional
            Reference frame the resulting vectors should be given in. If not given, the
            global reference frame is assumed.

        time : float, default: 0.0
            What time the transformations should be taken at. Only relevant if the
            reference frames have time-dependant motion.

        out_position : array, optional
            Output array to write the output positions to. If not given a new one is
            created.

        out_velocity : array, optional
            Output array to write the output velocity to. If not given a new one is
            created.

        Returns
        -------
        array
            Array of position vectors. If ``out_position`` was given, then the reference
            to it is returned.

        array
            Array of velocity vectors. If ``out_velocity`` was given, then the reference
            to it is returned.
        """
        ...

    @classmethod
    def transform_vector(
        cls,
        x: npt.ArrayLike,
        start: ReferenceFrame | None = None,
        end: ReferenceFrame | None = None,
        time: float = 0.0,
        out: npt.NDArray[np.double] | None = None,
    ) -> npt.NDArray[np.double]:
        """Transform vectors from one reference frame to another.

        Parameters
        ----------
        x : array_like
            Vectors to transform.

        start : ReferenceFrame, optional
            Reference frame the vectors are given in. If not given, the global
            reference frame is assumed.

        end : ReferenceFrame, optional
            Reference frame the resulting vectors should be given in. If not given, the
            global reference frame is assumed.

        time : float, default: 0.0
            What time the transformations should be taken at. Only relevant if the
            reference frames have time-dependant motion.

        out : array, optional
            Output array to write the output to. If not given a new one is created.

        Returns
        -------
        array
            Array of vectors. If ``out`` was given, then the reference to it is returned.
        """
        ...

    @property
    def is_moving(self) -> bool:
        """True if either the reference frame or its ancestors are moving."""
        ...

    def moved_relative_to(
        self, other: ReferenceFrame | None, t_start: float, t_end: float, tol: float
    ) -> bool:
        """Check if the reference frame had motion relative to another.

        This function is intended to be used to determine if relative induction matrices
        need to be recomputed.

        The motion is determined by computing the relative transformation between the two
        reference frames at these two times. From there, two things are considered:

        - Does the difference in relative offset at the two times have the magnitude
          below ``tol``?
        - Does the largest value of the relative orientation angles have the absolute
          value below ``tol``?

        If any of these criteria is met, the reference frames are considered to have
        moved.

        Parameters
        ----------
        other : ReferenceFrame or None
            Reference frame to compare it to. ``None`` corresponds to the global reference
            frame.

        t_start : float
            First time to compare to.

        t_end : float
            Second time to compare to.

        tol : float
            How much difference is allowed for the two reference frames to not
            be considered moving.

        Returns
        -------
        bool
            Indication if the two reference frames have moved with respect to one another.
        """
        ...

    def common_ancestor(self, other: ReferenceFrame | None) -> ReferenceFrame | None:
        """Find the first common ancestor with another reference frame.

        This function is intended to find the shortest transformation needed by the
        two reference frames.

        Parameters
        ----------
        other : ReferenceFrame or None
            The reference frame to find the ancestor with. ``None`` corresponds to the
            global reference frame.

        Returns
        -------
        ReferenceFrame of None
            The nearest common ancestor of the two reference frames.
        """
        ...

def quad_induction(
    tol: float,
    quad_positions: npt.ArrayLike,
    quad_circulations: npt.ArrayLike,
    target_positions: npt.ArrayLike,
    symmetry_plane: TransformationPlane | None = None,
    out_velocity: npt.NDArray[np.double] | None = None,
    n_threads: int = 1,
) -> npt.NDArray[np.double]:
    """Compute the influence of quadrilateral circulation filaments at input positions.

    This is mainly used for computing the influence of the wake, which contains quads,
    which are considered separate (hence no mesh).

    Parameters
    ----------
    tol : float
        Distance at which the induced velocity is set to zero due to being too
        close to the vortex line.

    quad_positions : (M, 4, 3) array
        Array of positions of the corners of the quadrilateral filaments. The first
        dimension corresponds to the filaments, while the second dimension corresponds to
        the corners of each filament.

    quad_circulations : (M,) array
        Array of circulations for each quadrilateral filament.

    target_positions : (K, 3) array
        Array of positions at which to compute the velocity influence.

    symmetry_plane : TransformationPlane, optional
        If given, the influence of the quadrilateral filaments is computed as if they were
        mirrored across the given plane.

    out_velocity : (K, 3) array, optional
        Output array to write the computed velocities to. If not given, a new one is
        created.

    n_threads : int, default: 1
        Number of threads to use for computing the velocities.

    Returns
    -------
    (K, 3) array
        Array of velocity vectors at the target positions induced by the quadrilateral
        filaments. If ``out_velocity`` was given, then the reference to it is returned.
    """
    ...

def quad_normal_induction(
    tol: float,
    quad_positions: npt.ArrayLike,
    quad_circulations: npt.ArrayLike,
    target_positions: npt.ArrayLike,
    target_normals: npt.ArrayLike,
    symmetry_plane: TransformationPlane | None = None,
    out_velocity: npt.NDArray[np.double] | None = None,
    n_threads: int = 1,
) -> npt.NDArray[np.double]:
    """Compute the normal velocity induced by the quad filaments at the given positions.

    This is mainly used for computing the influence of the wake, which contains quads,
    which are considered separate (hence no mesh).

    Parameters
    ----------
    tol : float
        Distance at which the induced velocity is set to zero due to being too
        close to the vortex line.

    quad_positions : (M, 4, 3) array
        Array of positions of the corners of the quadrilateral filaments. The first
        dimension corresponds to the filaments, while the second dimension corresponds to
        the corners of each filament.

    quad_circulations : (M,) array
        Array of circulations for each quadrilateral filament.

    target_positions : (K, 3) array
        Array of positions at which to compute the velocity influence.

    target_normals : (K, 3) array
        Array of normal vectors at the target positions. The normal vectors should be
        normalized.

    symmetry_plane : TransformationPlane, optional
        If given, the influence of the quadrilateral filaments is computed as if they were
        mirrored across the given plane.

    out_velocity : (K, 3) array, optional
        Output array to write the computed velocities to. If not given, a new one is
        created.

    n_threads : int, default: 1
        Number of threads to use for computing the velocities.

    Returns
    -------
    (K,) array
        Array of normal components of velocity vectors at the target positions induced by
        the quad filaments. If ``out_velocity`` was given, then the reference to it is
        returned.
    """
    ...
