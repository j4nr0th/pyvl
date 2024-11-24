"""Typing file for C implemented functions/objects."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np
import pyvista as pv
from numpy import typing as npt

class GeoID:
    """Class used to refer to topological objects with orientation."""

    def __init__(self, index: int, orientation: object = False) -> None: ...
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

class Line:
    """Class which describes a connection between two points."""

    def __init__(self, begin: int | GeoID, end: int | GeoID) -> None: ...

    begin: int
    """Start point of the line."""

    end: int
    """End point of the line."""

    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...
    def __eq__(self, value) -> bool: ...

class Surface:
    """Surface bound by a set of lines.

    If the end point of a line ``i`` is not the start point of the line
    ``i + 1``, a :class:`ValueError` will be raised.
    """

    def __init__(self, lines: Sequence[Line]) -> None: ...
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

class Mesh:
    """Object describing a discretization of a surface."""

    def __init__(
        self, positions: npt.ArrayLike, lines: Sequence[Line], surfaces: Sequence[Surface]
    ) -> None: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...
    @property
    def n_points(self) -> int:
        """Return the number of points in the mesh."""
        ...

    @property
    def n_lines(self) -> int:
        """Return the number of lines in the mesh."""
        ...

    @property
    def n_surfaces(self) -> int:
        """Return the number of surfaces in the mesh."""
        ...

    @property
    def positions(self) -> npt.NDArray[np.float64]:
        """Return array of positions of mesh points."""
        ...

    @positions.setter
    def positions(self, positions: npt.ArrayLike) -> None:
        """Set positions of mesh points."""
        ...

    @property
    def lines_iterator(self) -> Iterator[Line]:  # TODO: in the future make it ValuesView
        """Return iterator of lines objects in the mesh."""
        ...

    @property
    def surfaces_iterator(
        self,
    ) -> Iterator[Surface]:  # TODO: in the future make it ValuesView
        """Return iterator of lines objects in the mesh."""
        ...

    @classmethod
    def from_element_connectivity(
        cls, positions: npt.ArrayLike, element_indices: Sequence[Sequence[int]]
    ) -> Mesh:
        """Create Mesh from positions and point connectivity."""
        ...

    def compute_dual(self) -> Mesh:
        """Create dual to the current mesh."""
        ...

    def to_polydata(self) -> pv.PolyData: ...
    @classmethod
    def from_polydata(cls, pd: pv.PolyData) -> Mesh: ...
