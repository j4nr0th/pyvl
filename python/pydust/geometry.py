"""Implementation of Geometry related operations."""

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from warnings import warn

import meshio as mio
import numpy as np
import numpy.typing as npt
import pyvista as pv

from pydust.cdust import Mesh, ReferenceFrame


def mesh_from_mesh_io(m: mio.Mesh) -> Mesh:
    """TODO."""
    connections: list[npt.NDArray[np.unsignedinteger]] = []
    c: mio.CellBlock
    for c in m.cells:
        if c.dim != 2:
            warn(
                f"The mesh contains a cell block of type {c.type}, which has "
                "topological dimesion not equal to 2, so it will be ignored."
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
    primal: Mesh

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
        object.__setattr__(self, "primal", mesh)

    def as_polydata(self) -> pv.PolyData:
        """Convert geometry into PyVista's PolyData."""
        positions = self.reference_frame.from_parent_with_offset(self.primal.positions)
        nper_elem, indices = self.primal.to_element_connectivity()
        offsets = np.pad(np.cumsum(nper_elem), (1, 0))
        faces = [indices[offsets[i] : offsets[i + 1]] for i in range(nper_elem.size)]
        pd = pv.PolyData.from_irregular_faces(positions, faces)
        return pd

    @property
    def normals(self) -> npt.NDArray[np.float64]:
        """Compute normals to primal mesh surfaces."""
        n = self.primal.surface_normals
        return self.reference_frame.from_parent_without_offset(n, n)

    @property
    def centers(self) -> npt.NDArray[np.float64]:
        """Compute centers of primal mesh sufraces."""
        n = self.primal.surface_centers
        return self.reference_frame.from_parent_with_offset(n, n)

    # def induction_matrix(
    #     self,
    #     tol: float,
    #     control_points: npt.ArrayLike,
    #     out: npt.NDArray[np.float64] | None = None,
    # ):
    #     """Compute the induction matrix of the geometry."""
    #     cpts = np.asarray(control_points, np.float64)
    #     if len(cpts.shape) != 2 or cpts.shape[1] != 3:
    #         raise ValueError("Control points array must have the shape of (N, 3)")
    #     out_shape = (cpts.shape[0], self.primal.n_surfaces, 3)
    #     if out is None:
    #         out = np.empty(out_shape, np.float64)
    #     elif out.shape != out_shape:
    #         raise ValueError(
    #             f"The output array does not have the correct shape of {out_shape}, "
    #             f"instead its shape is {out.shape}."
    #         )
    #     elif out.dtype != np.float64:
    #         raise ValueError(
    #             f"The output array does not have {np.float64} as its dtype, instead it "
    #             f"has {out.dtype}."
    #         )
    #     line_buffer_shape = (cpts.shape[0], self.primal.n_lines, 3)
    #     if self._line_buffer is None or self._line_buffer.shape != line_buffer_shape:
    #         object.__setattr__(
    #             self, "_line_buffer", np.empty(line_buffer_shape, np.float64)
    #         )
    #     return self.primal.induction_matrix(tol, cpts, out, self._line_buffer)


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


@dataclass(frozen=True)
class _GeoInfo:
    name: str
    offset: np.uint32
    msh: Mesh
    rf: ReferenceFrame


class SimulationGeometry:
    """Simulation geometry containing several geometries in different reference frames."""

    msh: Mesh
    info: tuple[_GeoInfo, ...]
    _time: float
    _normals: npt.NDArray[np.float64]
    _cpts: npt.NDArray[np.float64]

    def __init__(self, geo: Mapping[str, Geometry] = {}, /, **kwargs: Geometry) -> None:
        for kw in kwargs:
            if kw in geo:
                raise ValueError(
                    f"Geometry with name {kw} given as both the kwarg and in the mapping."
                )
        position_list: list[npt.NDArray[np.float64]] = []
        element_counts: list[npt.NDArray[np.uint64]] = []
        indices: list[npt.NDArray[np.uint64]] = []
        n_pts = 0
        info_list: list[_GeoInfo] = []
        geo = {**geo, **kwargs}
        for geo_name in geo:
            g = geo[geo_name]
            nelm, conn = g.primal.to_element_connectivity()
            conn += n_pts
            offsets = np.pad(np.cumsum(nelm), (1, 0))
            faces = [conn[offsets[i] : offsets[i + 1]] for i in range(nelm.size)]
            pos = g.primal.positions
            position_list.append(pos)
            element_counts.append(nelm)
            indices.extend(faces)
            info = _GeoInfo(geo_name, np.uint32(n_pts), g.primal, g.reference_frame)
            info_list.append(info)
            n_pts += pos.shape[0]

        joined_mesh = Mesh(np.concatenate(position_list), indices)
        self.msh = joined_mesh
        self.info = tuple(info_list)
        self._time = 0.0
        self._normals = np.empty((joined_mesh.n_surfaces, 3), np.float64)
        self._cpts = np.empty((joined_mesh.n_surfaces, 3), np.float64)
        self.time = 0.0

    @property
    def time(self) -> float:
        """Time at which the geometry is."""
        return self._time

    @time.setter
    def time(self, t: float) -> None:
        if not isinstance(t, float):
            raise TypeError(f"Time is not a float but is instead {type(t)}.")
        self._time = t
        for info in self.info:
            rf = info.rf.at_time(t)
            i_begin = info.offset
            i_end = i_begin + info.msh.n_surfaces
            # Transform in place
            rf.to_parent_with_offset(
                info.msh.positions, self.msh.positions[i_begin:i_end, :]
            )
            rf.to_parent_with_offset(
                info.msh.surface_centers, self._cpts[i_begin:i_end, :]
            )
            rf.to_parent_without_offset(
                info.msh.surface_normals, self._normals[i_begin:i_end, :]
            )

    @property
    def normals(self) -> npt.NDArray[np.float64]:
        """Normal unit vectors for each surface."""
        return self._normals

    @property
    def control_points(self) -> npt.NDArray[np.float64]:
        """Control points for each surface."""
        return self._cpts

    def compute_circulation(
        self,
        velocity_function: Callable[
            [npt.NDArray[np.float64], float], npt.NDArray[np.float64]
        ],
        tol: float = 1e-6,
    ) -> npt.NDArray[np.float64]:
        """Compute circulation resulting from given velocity function."""
        v = velocity_function(self._cpts, self._time)
        rhs = np.sum(self._normals * v, axis=1)
        lhs = np.sum(
            self._normals[:, None, :] * self.msh.induction_matrix(tol, self._cpts), axis=2
        )
        return np.linalg.solve(lhs, rhs)

    def as_polydata(self) -> pv.PolyData:
        """Convert SimulationGeometry into PyVista's PolyData."""
        positions = self.msh.positions
        nper_elem, indices = self.msh.to_element_connectivity()
        offsets = np.pad(np.cumsum(nper_elem), (1, 0))
        faces = [indices[offsets[i] : offsets[i + 1]] for i in range(nper_elem.size)]
        pd = pv.PolyData.from_irregular_faces(positions, faces)
        return pd
