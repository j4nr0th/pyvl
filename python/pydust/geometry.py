"""Implementation of Geometry related operations."""

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import ItemsView, KeysView, ValuesView
from warnings import warn

import meshio as mio
import numpy as np
import numpy.typing as npt
import pyvista as pv

from pydust.cdust import INVALID_ID, Mesh, ReferenceFrame


def mesh_from_mesh_io(m: mio.Mesh) -> Mesh:
    """TODO."""
    connections: list[npt.NDArray[np.unsignedinteger]] = []
    c: mio.CellBlock
    for c in m.cells:
        if c.dim != 2:
            warn(
                f"The mesh contains a cell block of type {c.type}, which has "
                "topological dimension not equal to 2, so it will be ignored."
            )
            continue
        for element in c.data:
            connections.append(np.asarray(element, np.uint32))
    return Mesh(m.points, connections)


@dataclass(init=False, frozen=True)
class Geometry:
    """Class which describes a geometry compoenent."""

    label: str
    reference_frame: ReferenceFrame
    msh: Mesh

    def __init__(
        self, label: str, reference_frame: ReferenceFrame, mesh: mio.Mesh | Mesh
    ) -> None:
        if not isinstance(label, str):
            raise TypeError(
                f"label must be a string, instead it was {type(label).__name__}."
            )
        if not isinstance(reference_frame, ReferenceFrame):
            raise TypeError(
                "reference_frame must be a ReferenceFrame, instead it was "
                f"{type(reference_frame).__name__}."
            )
        if not isinstance(mesh, (Mesh, mio.Mesh)):
            raise TypeError(
                f"mesh must be a either {Mesh} or {mio.Mesh}, instead it was "
                f"{type(mesh)}."
            )
        if isinstance(mesh, mio.Mesh):
            mesh = mesh_from_mesh_io(mesh)

        object.__setattr__(self, "label", label)
        object.__setattr__(self, "reference_frame", reference_frame)
        object.__setattr__(self, "msh", mesh)

    def as_polydata(self) -> pv.PolyData:
        """Convert geometry into PyVista's PolyData."""
        positions = self.reference_frame.from_parent_with_offset(self.msh.positions)
        nper_elem, indices = self.msh.to_element_connectivity()
        offsets = np.pad(np.cumsum(nper_elem), (1, 0))
        faces = [indices[offsets[i] : offsets[i + 1]] for i in range(nper_elem.size)]
        pd = pv.PolyData.from_irregular_faces(positions, faces)
        return pd

    @property
    def normals(self) -> npt.NDArray[np.float64]:
        """Compute normals to mesh surfaces."""
        n = self.msh.surface_normals
        return self.reference_frame.from_parent_without_offset(n, n)

    @property
    def centers(self) -> npt.NDArray[np.float64]:
        """Compute centers of mesh sufraces."""
        n = self.msh.surface_centers
        return self.reference_frame.from_parent_with_offset(n, n)


@dataclass(frozen=True)
class GeometryInfo:
    """Class containing information about geometry."""

    rf: ReferenceFrame
    msh: Mesh
    closed: bool
    points: slice
    lines: slice
    surfaces: slice


@dataclass(frozen=True)
class SimulationGeometry(Mapping):
    """Class which is the result of combining multiple geometries together."""

    _info: dict[str, GeometryInfo]
    n_surfaces: int
    n_lines: int
    n_points: int

    def __init__(self, *geometries: Geometry) -> None:
        geos = {g.label: g for g in geometries}
        meshes: list[Mesh] = []
        info: dict[str, GeometryInfo] = {}
        n_points = 0
        n_lines = 0
        n_surfaces = 0
        for g_name in geos:
            g = geos[g_name]
            meshes.append(g.msh)
            if g.label in info:
                raise ValueError(
                    f'Geometries with duplicated label "{g.label}" were found.'
                )
            c_p = g.msh.n_points
            c_l = g.msh.n_lines
            c_s = g.msh.n_surfaces
            dual = g.msh.compute_dual()
            closed = True
            for il in range(dual.n_lines):
                ln = dual.get_line(il)
                if ln.begin == INVALID_ID or ln.end == INVALID_ID:
                    closed = False
                    break
            info[g.label] = GeometryInfo(
                g.reference_frame,
                g.msh,
                closed,
                slice(n_points, n_points + c_p),
                slice(n_lines, n_lines + c_l),
                slice(n_surfaces, n_surfaces + c_s),
            )
            n_points += c_p
            n_lines += c_l
            n_surfaces += c_s

        object.__setattr__(self, "_info", info)
        object.__setattr__(self, "n_points", n_points)
        object.__setattr__(self, "n_lines", n_lines)
        object.__setattr__(self, "n_surfaces", n_surfaces)

    def __getitem__(self, key: str) -> GeometryInfo:
        """Return the geometry corresponding to the key."""
        return self._info[key]

    def __iter__(self) -> Iterator[str]:
        """Return iterator over keys of the simulation geometry."""
        return iter(self._info)

    def __len__(self) -> int:
        """Return the number of meshes in the simulation geometry."""
        return len(self._info)

    def __contains__(self, key: object) -> bool:
        """Check if a geometry with given label is within the simulation geometry."""
        return key in self._info

    def keys(self) -> KeysView[str]:
        """Return the view of the keys."""
        return self._info.keys()

    def items(self) -> ItemsView[str, GeometryInfo]:
        """Return the view of the items."""
        return self._info.items()

    def values(self) -> ValuesView[GeometryInfo]:
        """Return the view of the values."""
        return self._info.values()

    def at_time(self, t: float) -> Mesh:
        """Return the geometry at the specified time."""
        meshes: list[Mesh] = []
        for geo_name in self._info:
            info = self._info[geo_name]
            new_rf = info.rf.at_time(float(t))
            new_pos = new_rf.to_global_with_offset(info.msh.positions)
            meshes.append(info.msh.copy(new_pos))

        return Mesh.merge_meshes(*meshes)

    def at_time_polydata(self, t: float) -> pv.PolyData:
        """Return the geometry at the specified time."""
        mesh = self.at_time(t)
        positions = mesh.positions
        nper_elem, indices = mesh.to_element_connectivity()
        offsets = np.pad(np.cumsum(nper_elem), (1, 0))
        faces = [indices[offsets[i] : offsets[i + 1]] for i in range(nper_elem.size)]
        pd = pv.PolyData.from_irregular_faces(positions, faces)
        return pd


def geometry_show_pyvista(
    geometries: Iterable[Geometry], plt: pv.Plotter | None = None
) -> None:
    """Show the geometry using PyVista."""
    show = plt is None
    if plt is None:
        plt = pv.Plotter(theme=pv.themes.DocumentProTheme())
        plt.theme.show_edges = True
        plt.theme.show_scalar_bar = False

    for geo in geometries:
        plt.add_mesh(geo.as_polydata(), label=geo.label)

    if show:
        plt.show()
