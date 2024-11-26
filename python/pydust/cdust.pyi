"""Typing file for C implemented functions/objects."""

from __future__ import annotations

from collections.abc import Sequence
from typing import final

import numpy as np
from numpy import typing as npt

@final
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

@final
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

@final
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

@final
class Mesh:
    """Object describing a discretization of a surface."""

    def __init__(
        self, positions: npt.ArrayLike, connectivity: Sequence[Sequence[int]]
    ) -> None: ...
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

    @property
    def positions(self) -> npt.NDArray[np.float64]:
        """Return array of positions of mesh points."""
        ...

    @positions.setter
    def positions(self, positions: npt.ArrayLike) -> None:
        """Positions of mesh points."""
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

    # def strip_invalid(self) -> Mesh:
    #     """Return mesh without any entries with invalid ids."""
    #     ...
