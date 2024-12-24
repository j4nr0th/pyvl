"""Typing file for C implemented functions/objects."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Self, final

import numpy as np
from numpy import typing as npt

from pydust._typing import VecLike3
from pydust.io_common import HirearchicalMap

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

    # @staticmethod
    # def line_velocity_to_force(
    #     primal_mesh: Mesh,
    #     dual_mesh: Mesh,
    #     surface_circulation: npt.NDArray[np.float64],
    #     line_velocity_force: npt.NDArray[np.float64],
    # ) -> None:
    #     """Compute line forces due to average velocity along it inplace."""
    #     ...

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

class ReferenceFrame:
    """Class which is used to define position and orientation of geometry."""

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
        """Apply transformation to the reference frame from parent with offset."""
        ...

    def from_parent_without_offset(
        self, r: npt.ArrayLike, out: npt.NDArray[np.float64] | None = None, /
    ) -> npt.NDArray[np.float64]:
        """Apply transformation to the reference frame from parent without offset."""
        ...

    def to_parent_with_offset(
        self, r: npt.ArrayLike, out: npt.NDArray[np.float64] | None = None, /
    ) -> npt.NDArray[np.float64]:
        """Reverse transformation from the reference frame to parent with offset."""
        ...

    def to_parent_without_offset(
        self, r: npt.ArrayLike, out: npt.NDArray[np.float64] | None = None, /
    ) -> npt.NDArray[np.float64]:
        """Reverse transformation from the reference frame to parent without offset."""
        ...

    def from_global_with_offset(
        self, r: npt.ArrayLike, out: npt.NDArray[np.float64] | None = None, /
    ) -> npt.NDArray[np.float64]:
        """Apply transformation to the reference frame from global with offset."""
        ...

    def from_global_without_offset(
        self, r: npt.ArrayLike, out: npt.NDArray[np.float64] | None = None, /
    ) -> npt.NDArray[np.float64]:
        """Apply transformation to the reference frame from global without offset."""
        ...

    def to_global_with_offset(
        self, r: npt.ArrayLike, out: npt.NDArray[np.float64] | None = None, /
    ) -> npt.NDArray[np.float64]:
        """Reverse transformation from the reference frame to global with offset."""
        ...

    def to_global_without_offset(
        self, r: npt.ArrayLike, out: npt.NDArray[np.float64] | None = None, /
    ) -> npt.NDArray[np.float64]:
        """Reverse transformation from the reference frame to global without offset."""
        ...

    def rotate_x(self, theta: float) -> ReferenceFrame:
        """Create a copy of the frame rotated around the x-axis."""
        ...

    def rotate_y(self, theta: float) -> ReferenceFrame:
        """Create a copy of the frame rotated around the y-axis."""
        ...

    def rotate_z(self, theta: float) -> ReferenceFrame:
        """Create a copy of the frame rotated around the z-axis."""
        ...

    def with_offset(self, new_offset: npt.ArrayLike) -> ReferenceFrame:
        """Create a copy of the frame with different offset value."""
        ...

    def at_time(self, t: float) -> ReferenceFrame:
        """Compute reference frame at the given time.

        This is useful when the reference frame is moving or rotating in space.

        Parameters
        ----------
        t : float
            Time at which the reference frame is needed.

        Returns
        -------
        ReferenceFrame
            New reference frame.
        """
        ...

    @staticmethod
    def angles_from_rotation(rotation_matrix: npt.ArrayLike) -> npt.NDArray[np.double]:
        """Compute rotation angles from a transformation matrix."""
        ...

    def save(self, group: HirearchicalMap, /) -> None:
        """Serialize the ReferenceFrame into a HDF5 group."""
        ...

    @classmethod
    def load(cls, group: HirearchicalMap, parent: ReferenceFrame | None = None) -> Self:
        """Load the ReferenceFrame from a HDF5 group."""
        ...
