"""Implementation of the meshing for the vortex lattice meshes.

Vortex lattice meshes are used to represent the geometry of the lifting surfaces in the
flow. These have no thickness, but can have a camber and twist.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np
import numpy.typing as npt

from pyvl._typing import Vec3Callable
from pyvl.cvl import Mesh, ReferenceFrame
from pyvl.geometry import Geometry


# Not a frozen dataclass, since it is just a generator for meshes
@dataclass
class VLBlade:
    """A vortex lattice blade.

    This class is used to represent a single blade of a propeller or rotor. It is
    defined by its geometry and the number of panels in the chordwise and spanwise
    directions.

    The blade at the root is defined in the following coordinate system:

    - x-axis goes from the leading edge to the trailing edge of the blade
    - y-axis goes from the root to the tip of the blade
    - z-axis is perpendicular to the plane of the blade, following the right-hand rule


    Attributes
    ----------
    reference_line : Vec3Callable
        A callable which maps the span fraction to a 3D point in space. This defines the
        reference line of the blade.

    reference_chord_fraction : float
        The fraction of the chord length at which the reference line is defined. This
        affects the results of the twist and chord distribution.

    chord_distribution : Callable[[float], float]
        A callable which maps the span fraction to the chord length at that span
        fraction. This defines the chord distribution of the blade.

    twist_distribution : Callable[[float], float], optional
        The angle at which the local chord is rotated about the reference line.
        Accepts a span fraction and returns the twist angle in radians. If not provided,
        the twist angle is assumed to be zero along the entire span of the blade.

    camber_distribution : Callable[[float, float], float], optional
        A callable which maps the span fraction and chord fraction to the camber height
        at that point. This defines the camber distribution of the blade. If not given,
        the blade is assumed to be flat (no camber).
    """

    _reference_line: Vec3Callable
    reference_chord_fraction: float
    _chord_distribution: Callable[[float], float]
    _twist_distribution: Callable[[float], float]
    _camber_distribution: Callable[[float, float], float] | None

    def __init__(
        self,
        reference_line: Vec3Callable | float,
        reference_chord_fraction: float = 0.25,
        chord_distribution: Callable[[float], float] | float = 1.0,
        twist_distribution: Callable[[float], float] | float = 0.0,
        camber_distribution: Callable[[float, float], float] | float | None = None,
    ) -> None:
        self.reference_line = reference_line
        self.reference_chord_fraction = reference_chord_fraction
        self.chord_distribution = chord_distribution
        self.twist_distribution = twist_distribution
        self.camber_distribution = camber_distribution

    @property
    def reference_line(self) -> Vec3Callable:
        """The reference line of the blade."""
        return self._reference_line

    @reference_line.setter
    def reference_line(self, value: Vec3Callable | float) -> None:
        if callable(value):
            # If a callable is provided, use it directly
            self._reference_line = value
        else:
            # If a float is provided, create a callable that goes up to given span
            self._reference_line = lambda s: np.array([0, value * s, 0])

    @property
    def chord_distribution(self) -> Callable[[float], float]:
        """The chord distribution of the blade."""
        return self._chord_distribution

    @chord_distribution.setter
    def chord_distribution(self, value: Callable[[float], float] | float) -> None:
        if callable(value):
            # If a callable is provided, use it directly
            self._chord_distribution = value
        else:
            # If a float is provided, create a callable that returns the constant chord
            self._chord_distribution = lambda _: value

    @property
    def twist_distribution(self) -> Callable[[float], float]:
        """The twist distribution of the blade."""
        return self._twist_distribution

    @twist_distribution.setter
    def twist_distribution(self, value: Callable[[float], float] | float) -> None:
        if callable(value):
            # If a callable is provided, use it directly
            self._twist_distribution = value
        else:
            # If a float is provided, create a callable that returns the constant twist
            self._twist_distribution = lambda _: value

    @property
    def camber_distribution(self) -> Callable[[float, float], float] | None:
        """The camber distribution of the blade."""
        return self._camber_distribution

    @camber_distribution.setter
    def camber_distribution(
        self, value: Callable[[float, float], float] | float | None
    ) -> None:
        if callable(value):
            # If a callable is provided, use it directly
            self._camber_distribution = value
        elif value is None:
            # If None is provided, set the camber distribution to None
            self._camber_distribution = None
        else:
            # If a float is provided, create a callable that returns the constant camber
            self._camber_distribution = lambda _, __: value

    def mesh_geometry(
        self,
        spanwise_positions: int | npt.ArrayLike,
        chordwise_positions: int | npt.ArrayLike,
        label: str = "vl_blade",
        reference_frame: ReferenceFrame = ReferenceFrame(),
    ) -> Geometry:
        """Generate a vortex lattice mesh for the blade.

        Parameters
        ----------
        spanwise_positions : int | npt.ArrayLike
            The number of spanwise positions (panels) to generate along the blade. If an
            integer is provided, the spanwise positions will be evenly spaced along the
            span of the blade. If an array-like object is provided, it should contain the
            spanwise positions (as fractions of the span) at which to generate the panels.

        chordwise_positions : int | npt.ArrayLike
            The number of chordwise positions (panels) to generate along the blade. If an
            integer is provided, the chordwise positions will be evenly spaced along the
            chord of the blade. If an array-like object is provided, it should contain the
            chordwise positions (as fractions of the chord) at which to generate the
            panels.

        label : str, default : "vl_blade"
            A label for the generated :class:`Geometry`.

        reference_frame : ReferenceFrame, default : ReferenceFrame()
            The reference frame in which the generated :class:`Geometry` is defined.

        Returns
        -------
        Geometry
            Geometry object containing the vortex lattice mesh for the blade.
        """
        # Check if we got integers or arrays for the spanwise and chordwise positions
        if isinstance(spanwise_positions, int):
            spanwise_positions = np.linspace(0, 1, spanwise_positions)
        else:
            spanwise_positions = np.asarray(spanwise_positions, np.double)
        if isinstance(chordwise_positions, int):
            chordwise_positions = np.linspace(0, 1, chordwise_positions)
        else:
            chordwise_positions = np.asarray(chordwise_positions, np.double)

        # Twist these around the reference chord
        sections: list[npt.NDArray[np.double]] = list()

        for i, s in enumerate(spanwise_positions):
            chord_length = self._chord_distribution(s)
            chord_positions = (
                chordwise_positions - self.reference_chord_fraction
            ) * chord_length
            pts = np.zeros((chord_positions.size, 3), dtype=np.double)
            pts[:, 0] = chord_positions

            if self._camber_distribution is not None:
                pts[:, 2] = [self._camber_distribution(s, c) for c in chordwise_positions]

            # Rotate the leading edge and trailing edge positions around the reference
            # line by the twist angle
            twist_angle = np.clip(self._twist_distribution(s), -2 * np.pi, 2 * np.pi)
            if twist_angle != 0.0:
                ct = np.cos(twist_angle)
                st = np.sin(twist_angle)
                # Twist is positive CW when looking from the root to the tip
                px = pts[:, 0] * ct - pts[:, 2] * st
                pz = pts[:, 0] * st + pts[:, 2] * ct
                pts[:, 0] = px
                pts[:, 2] = pz

            # Add the reference line position to the points
            line_point = np.asarray(self._reference_line(s), np.double)
            if line_point.shape != (3,):
                raise ValueError("reference_line must return a 3D point")
            pts += line_point

            sections.append(pts)

        s = np.stack(sections, axis=0)

        return Geometry(
            mesh=Mesh.make_quad2d_plane(s.shape[1] - 1, s.shape[0] - 1),
            positions=s.reshape(-1, 3),
            label=label,
            reference_frame=reference_frame,
        )
