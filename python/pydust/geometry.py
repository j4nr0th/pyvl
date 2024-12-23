"""Implementation of Geometry related operations."""

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import ItemsView, KeysView, Self, ValuesView
from warnings import warn

import h5py
import meshio as mio
import numpy as np
import numpy.typing as npt
import pyvista as pv

from pydust.cdust import INVALID_ID, Mesh, ReferenceFrame


def mesh_from_mesh_io(m: mio.Mesh) -> tuple[npt.NDArray[np.float64], Mesh]:
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
    return (np.asarray(m.points, dtype=np.float64), Mesh(m.points.shape[0], connections))


def mesh_to_polydata_faces(m: Mesh) -> list[npt.NDArray]:
    """Convert mesh into PolyData faces."""
    nper_elem, indices = m.to_element_connectivity()
    offsets = np.pad(np.cumsum(nper_elem), (1, 0))
    faces = [indices[offsets[i] : offsets[i + 1]] for i in range(nper_elem.size)]
    return faces


def mesh_to_hdf5(m: Mesh, group: h5py.Group) -> None:
    """Serialize the mesh into a HDF5 group."""
    group["n_points"] = m.n_points
    n_per_element, flattened_elements = m.to_element_connectivity()
    group["n_per_element"] = n_per_element
    group["flattened_elements"] = flattened_elements


def mesh_from_hdf5(group: h5py.Group) -> Mesh:
    """Deserialize the mesh from HDF5 group."""
    n_points = group["n_points"]
    n_per_element = group["n_per_element"]
    flattened_elements = group["flattened_elements"]
    assert isinstance(n_points, h5py.Dataset)
    assert isinstance(n_per_element, h5py.Dataset)
    assert isinstance(flattened_elements, h5py.Dataset)
    nper_elem = n_per_element[()]
    indices = flattened_elements[()]
    offsets = np.pad(np.cumsum(nper_elem), (1, 0))
    faces = [indices[offsets[i] : offsets[i + 1]] for i in range(nper_elem.size)]
    return Mesh(n_points=n_points[()], connectivity=faces)


def rf_to_hdf5(self: ReferenceFrame, group: h5py.Group) -> None:
    """Serialize the ReferenceFrame into a HDF5 group."""
    rf_type = type(self)
    group["type"] = rf_type.__module__ + "." + rf_type.__qualname__
    data = group.create_group("data")
    self.save(data)
    if self.parent is not None:
        parent = group.create_group("parent")
        rf_to_hdf5(self.parent, parent)


def rf_from_hdf5(group: h5py.Group) -> ReferenceFrame:
    """Load reference frame from a HDF5 group."""
    rf_type_name = group["type"]
    assert isinstance(rf_type_name, h5py.Dataset)
    type_name: str = rf_type_name[()].decode()
    mname, tname = type_name.rsplit(".", 1)
    mod = __import__(mname, fromlist=[tname])
    cls = getattr(mod, tname)
    data = group["data"]
    assert isinstance(data, h5py.Group)
    parent = None
    if "parent" in group:
        parent_group = group["parent"]
        assert isinstance(parent_group, h5py.Group)
        parent = rf_from_hdf5(parent_group)
    return cls.load(group=data, parent=parent)


@dataclass(init=False, frozen=True)
class Geometry:
    """Class which describes a geometry compoenent."""

    label: str
    reference_frame: ReferenceFrame
    positions: npt.NDArray[np.float64]
    msh: Mesh

    def __init__(
        self,
        label: str,
        reference_frame: ReferenceFrame,
        mesh: Mesh,
        positions: npt.ArrayLike,
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
        if not isinstance(mesh, Mesh):
            raise TypeError(
                f"mesh must be a either Mesh object, instead it was " f"{type(mesh)}."
            )
        try:
            pos = np.array(positions, np.float64).reshape((-1, 3))
        except Exception as e:
            raise ValueError("Positions must be a (N, 3) array.") from e

        object.__setattr__(self, "label", label)
        object.__setattr__(self, "reference_frame", reference_frame)
        object.__setattr__(self, "positions", pos)
        object.__setattr__(self, "msh", mesh)

    @classmethod
    def from_meshio(
        cls, label: str, reference_frame: ReferenceFrame, mesh: mio.Mesh
    ) -> Self:
        """Create a Geometry from a MeshIO Mesh object."""
        p, m = mesh_from_mesh_io(mesh)
        return cls(label=label, reference_frame=reference_frame, mesh=m, positions=p)

    @classmethod
    def from_polydata(
        cls, label: str, reference_frame: ReferenceFrame, pd: pv.PolyData
    ) -> Self:
        """Create a Geometry from a PyVista's PolyData object."""
        return cls(
            label=label,
            reference_frame=reference_frame,
            mesh=Mesh(pd.points.shape[0], pd.irregular_faces),
            positions=pd.points,
        )

    def as_polydata(self) -> pv.PolyData:
        """Convert geometry into PyVista's PolyData."""
        positions = self.reference_frame.from_parent_with_offset(self.positions)
        faces = mesh_to_polydata_faces(self.msh)
        pd = pv.PolyData.from_irregular_faces(positions, faces)
        return pd

    @property
    def normals(self) -> npt.NDArray[np.float64]:
        """Compute normals to mesh surfaces."""
        n = self.msh.surface_normal(self.positions)
        return self.reference_frame.from_parent_without_offset(n, n)

    @property
    def centers(self) -> npt.NDArray[np.float64]:
        """Compute centers of mesh sufraces."""
        n = self.msh.surface_average_vec3(self.positions)
        return self.reference_frame.from_parent_with_offset(n, n)

    def save(self, group: h5py.Group) -> None:
        """Save geometry into a HDF5 group."""
        group["positions"] = self.positions
        mesh_grop = group.create_group("mesh")
        mesh_to_hdf5(self.msh, mesh_grop)
        rf_group = group.create_group("reference_frame")
        rf_to_hdf5(self.reference_frame, rf_group)

    @classmethod
    def load(cls, label: str, group: h5py.Group) -> Self:
        """Load the geometry for the HDF5 group."""
        positions = group["positions"]
        mesh_group = group["mesh"]
        rf_group = group["reference_frame"]
        assert isinstance(positions, h5py.Dataset)
        assert isinstance(mesh_group, h5py.Group)
        assert isinstance(rf_group, h5py.Group)

        msh = mesh_from_hdf5(mesh_group)
        rf = rf_from_hdf5(rf_group)

        return cls(label=label, reference_frame=rf, mesh=msh, positions=positions[()])


@dataclass(frozen=True)
class GeometryInfo:
    """Class containing information about geometry."""

    rf: ReferenceFrame
    msh: Mesh
    pos: npt.NDArray[np.float64]
    closed: bool
    points: slice
    lines: slice
    surfaces: slice


@dataclass(frozen=True)
class SimulationGeometry(Mapping):
    """Class which is the result of combining multiple geometries together."""

    _info: dict[str, GeometryInfo]
    mesh: Mesh
    dual: Mesh
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
                g.positions,
                closed,
                slice(n_points, n_points + c_p),
                slice(n_lines, n_lines + c_l),
                slice(n_surfaces, n_surfaces + c_s),
            )
            n_points += c_p
            n_lines += c_l
            n_surfaces += c_s

        object.__setattr__(self, "_info", info)
        object.__setattr__(self, "mesh", Mesh.merge_meshes(*meshes))
        object.__setattr__(self, "dual", self.mesh.compute_dual())
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

    def positions_at_time(self, t: float) -> npt.NDArray[np.float64]:
        """Return the point positions at the specified time."""
        pos = np.empty((self.n_points, 3), np.float64)
        for geo_name in self._info:
            info = self._info[geo_name]
            new_rf = info.rf.at_time(float(t))
            pos[info.points] = new_rf.to_global_with_offset(info.pos)
        return pos

    def polydata_at_time(self, t: float) -> pv.PolyData:
        """Return the geometry as polydata at the specified time."""
        pos = self.positions_at_time(t)
        faces = mesh_to_polydata_faces(self.mesh)
        pd = pv.PolyData.from_irregular_faces(pos, faces)
        return pd

    def te_normal_criterion(self, crit: float) -> npt.NDArray[np.uint]:
        """Identify edges, for which the normals of neighbouring surfaces meet dp crit."""
        normals = np.empty((self.n_surfaces, 3), np.float64)
        for name in self._info:
            info = self._info[name]
            normals[info.points] = info.msh.surface_normal(info.pos)
        return self.dual.dual_normal_criterion(crit, normals)

    def te_free_criterion(self) -> npt.NDArray[np.uint]:
        """Identify edges, which have only one surface attached."""
        return self.dual.dual_free_edges()

    def line_adjecency_information(
        self, lines: Sequence[int] | npt.NDArray[np.integer]
    ) -> tuple[npt.NDArray[np.uint], npt.NDArray[np.uint]]:
        """Return line information in terms of adjacent points and surfaces."""
        bordering_nodes = np.empty((len(lines), 2), np.uint)
        adjacent_surfaces = np.empty((len(lines), 2), np.uint)
        for i, line_id in enumerate(lines):
            primal_line = self.mesh.get_line(int(line_id))
            dual_line = self.dual.get_line(int(line_id))
            bordering_nodes[i, :] = (primal_line.begin, primal_line.end)
            adjacent_surfaces[i, :] = (dual_line.begin, dual_line.end)
        return (bordering_nodes, adjacent_surfaces)

    def save(self, group: h5py.Group) -> None:
        """Save the simulation geometry into a HDF5 group."""
        for geo_name in self._info:
            info = self._info[geo_name]
            geo_group = group.create_group(geo_name)
            Geometry(geo_name, info.rf, info.msh, info.pos).save(geo_group)

    @classmethod
    def load(cls, group: h5py.Group) -> Self:
        """Load the simulation geometry from the HDF5 group."""
        geometries: list[Geometry] = []
        for geo_name in group:
            sub_group = group[geo_name]
            assert isinstance(sub_group, h5py.Group)
            geo = Geometry.load(str(geo_name), sub_group)
            geometries.append(geo)
        return cls(*geometries)


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
