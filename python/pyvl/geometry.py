"""Implementation of Geometry related operations."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import ItemsView, KeysView, Self, ValuesView
from warnings import warn

import meshio as mio
import numpy as np
import numpy.typing as npt
import pyvista as pv

from pyvl._typing import CallableDeserializer, CallableSerializer
from pyvl.cvl import INVALID_ID, GeoID, Mesh, ReferenceFrame
from pyvl.fio.io_common import HirearchicalMap
from pyvl.fio.type_resolution import reference_frame_from_serial


def mesh_from_mesh_io(m: mio.Mesh) -> tuple[npt.NDArray[np.double], Mesh]:
    """Convert a meshio.Mesh object into position array and pyvl.Mesh.

    This function extracts 2D cell blocks from the meshio mesh and converts them
    into the pyvl Mesh format. Only cells with topological dimension 2 (surfaces)
    are included; other cell types are ignored with a warning.

    Parameters
    ----------
    m : meshio.Mesh
        The meshio Mesh object to convert.

    Returns
    -------
    tuple of (N, 3) ndarray, Mesh
        A tuple containing:
        - An array of point positions with shape (N, 3)
        - A Mesh object with the connectivity information
    """
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
    return (np.asarray(m.points, dtype=np.double), Mesh(m.points.shape[0], connections))


def mesh_to_polydata_faces(m: Mesh) -> list[npt.NDArray]:
    """Convert mesh into PolyData faces."""
    nper_elem, indices = m.to_element_connectivity()
    offsets = np.pad(np.cumsum(nper_elem), (1, 0))
    faces = [indices[offsets[i] : offsets[i + 1]] for i in range(nper_elem.size)]
    return faces


def mesh_to_serial(m: Mesh) -> HirearchicalMap:
    """Serialize the mesh into a HirearchicalMap."""
    out = HirearchicalMap()
    out.insert_int("n_points", m.n_points)
    n_per_element, flattened_elements = m.to_element_connectivity()
    out.insert_array("n_per_element", n_per_element)
    out.insert_array("flattened_elements", flattened_elements)
    return out


def mesh_from_serial(group: HirearchicalMap) -> Mesh:
    """Deserialize the mesh from a HirearchicalMap."""
    n_points = group.get_int("n_points")
    n_per_element = group.get_array("n_per_element")
    flattened_elements = np.asarray(group.get_array("flattened_elements"), np.uint32)
    offsets = np.pad(np.cumsum(n_per_element), (1, 0))
    faces = [
        flattened_elements[offsets[i] : offsets[i + 1]] for i in range(n_per_element.size)
    ]
    return Mesh(n_points=n_points, connectivity=faces)


def rf_to_serial(self: ReferenceFrame, serializer: CallableSerializer) -> HirearchicalMap:
    """Serialize the ReferenceFrame into a HirearchicalMap."""
    out = HirearchicalMap()
    self.save(out, serializer)
    print(f"DEBUG: ReferenceFrame keys after save: {list(out.keys())}")
    if self.parent is not None:
        parent = rf_to_serial(self.parent, serializer)
        out.insert_hirearchical_map("parent", parent)
    return out


def rf_from_serial(
    group: HirearchicalMap, deserializer: CallableDeserializer
) -> ReferenceFrame:
    """Load reference frame from a HirearchicalMap.

    Parameters
    ----------
    group : HirearchicalMap
        The serialized reference frame data.

    Returns
    -------
    ReferenceFrame
        The deserialized reference frame.
    """
    parent = None
    if "parent" in group:
        parent_group = group.get_hirearchical_map("parent")
        parent = reference_frame_from_serial(parent_group, deserializer)

    return (
        ReferenceFrame.load(group, deserializer, parent)
        if parent
        else ReferenceFrame.load(group, deserializer)
    )


@dataclass(init=False, frozen=True, eq=False)
class Geometry:
    """Class used to describe a geometry component.

    These objects are intended to be created by calling its class methods,
    such as :meth:`Geometry.from_meshio` or :meth:`Geometry.from_polydata`.

    Parameters
    ----------
    label : str
        Label by which to identify the geometry by in plotting and analysis.

    reference_frame : ReferenceFrame
        The frame of reference in which describes how the geometry's coordinate
        system is oriented in relation to the global coordinate system.

    mesh : Mesh
        The connectivity information about the geometry. Typically generated
        by :func:`classmethod` constructors.

    positions : (N, 3) array_like
        An array of position vectors for coordinates of each point in the mesh.

    Examples
    --------
    As an example, load the data from the ``examples`` submodule.

    .. jupyter-execute::

        >>> from pyvl.examples import example_file_name
        >>> from pyvl import ReferenceFrame, Geometry
        >>> import pyvista as pv
        >>> pv.set_plot_theme("document")
        >>> pv.set_jupyter_backend("html")
        >>> pv.global_theme.show_edges = True

        >>> import meshio as mio
        >>> msh = mio.read(example_file_name("wing1.obj"))
        >>> geo = Geometry.from_meshio(
        ...     "example_wing",
        ...     ReferenceFrame(),
        ...     msh,
        ... )

    The :class:`Geometry` can now be plotted:

    .. jupyter-execute::

        >>> geo.as_polydata().plot(interactive=False)

    """

    label: str
    reference_frame: ReferenceFrame
    positions: npt.NDArray[np.double]
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
                f"mesh must be a either Mesh object, instead it was {type(mesh)}."
            )
        try:
            pos = np.array(positions, np.double).reshape((-1, 3))
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
        """Create a Geometry from a MeshIO Mesh object.

        Parameters
        ----------
        label : str
            Label by which to identify the geometry by in plotting and analysis.

        reference_frame : ReferenceFrame
            The frame of reference in which describes how the geometry's coordinate
            system is oriented in relation to the global coordinate system.

        mesh : meshio.Mesh
            ``Mesh`` object, which contains the geometry to load. From it only
            cell blocks with cells of topological dimension 2 will be loaded.
            Any other type of cells will be ignored and a :class:`UserWarning`
            will be issued.

        Returns
        -------
        Geometry
            Newly created geometry.
        """
        p, m = mesh_from_mesh_io(mesh)
        return cls(label=label, reference_frame=reference_frame, mesh=m, positions=p)

    @classmethod
    def from_polydata(
        cls, label: str, reference_frame: ReferenceFrame, pd: pv.PolyData
    ) -> Self:
        """Create a Geometry from a PyVista's PolyData object.

        Parameters
        ----------
        label : str
            Label by which to identify the geometry by in plotting and analysis.

        reference_frame : ReferenceFrame
            The frame of reference in which describes how the geometry's coordinate
            system is oriented in relation to the global coordinate system.

        mesh : PolyData
            :class:`pyvista.PolyData` object, which contains the geometry to load.
            From it, all the faces will be extracted.

        Returns
        -------
        Geometry
            Newly created geometry.
        """
        return cls(
            label=label,
            reference_frame=reference_frame,
            mesh=Mesh(
                pd.points.shape[0],
                tuple(np.astype(x, np.uint32) for x in pd.irregular_faces),
            ),
            positions=pd.points,
        )

    def as_polydata(self) -> pv.PolyData:
        """Convert geometry into PyVista's PolyData.

        This is inverse to the :meth:`Geometry.from_polydata`, as it takes a
        :class:`Geometry` object and produces a :class:`pyvista.PolyData`.

        Returns
        -------
        pyvista.PolyData
            PolyData, which represents the geometry.
        """
        positions = self.reference_frame.from_parent_position(self.positions)
        faces = mesh_to_polydata_faces(self.msh)
        pd = pv.PolyData.from_irregular_faces(positions, faces)
        return pd

    def _propagate_edges(
        self, seed_edges: Iterable[int], max_angle: float, dual: Mesh
    ) -> npt.NDArray[np.uint]:
        """Propagate edge selection based on angle between direction vectors."""
        selected: set[GeoID] = set()
        edges_to_check = list(GeoID(e) for e in seed_edges) + list(
            GeoID(e, orientation=True) for e in seed_edges
        )

        while edges_to_check:
            curr = edges_to_check.pop()
            if curr in selected:
                # Skip already selected edges
                continue
            selected.add(curr)
            # Get the primal line and compute the direction vector
            pi1, pi2 = self.msh.get_line_points(curr)
            d = self.positions[pi2] - self.positions[pi1]
            norm = np.linalg.norm(d)
            curr_dir = d / norm if norm > 0 else np.zeros(3)
            # Get the neighbors from the dual mesh
            neighbors: list[GeoID] = list()
            neighbors.extend(-e for e in dual.get_surface_lines(pi1))
            neighbors.extend(e for e in dual.get_surface_lines(pi2))
            neighbors.remove(-curr)  # Remove the current edge from neighbors if present

            for neighbor in neighbors:
                if neighbor in selected:
                    # Skip already selected neighbors
                    continue

                # Get the direction vector of the neighbor edge
                # One of these is either p1 or p2
                pi1_n, pi2_n = self.msh.get_line_points(neighbor)
                assert pi1_n == pi2 or pi2_n == pi1, (
                    "Neighbor edge should share a point with the current edge."
                )
                d_n = self.positions[pi2_n] - self.positions[pi1_n]

                # if pi2_n == pi2 or pi1_n == pi1:
                #     # Direction should always go away from the current line,
                #     # so if the edge ends at one of the current line's points,
                #     # the direction should be flipped.
                #     d_n = -d_n

                norm_n = np.linalg.norm(d_n)
                neighbor_dir = d_n / norm_n if norm_n > 0 else np.zeros(3)
                # Absolute value of DP would for opposite directions
                cos_theta = np.dot(curr_dir, neighbor_dir)
                if cos_theta >= np.cos(max_angle):
                    edges_to_check.append(neighbor)

        return np.array(np.unique([edge.index for edge in selected]), dtype=np.uint)

    def detect_trailing_edge_manual(
        self, initial_angle: float = 0.05
    ) -> npt.NDArray[np.uint]:
        """Interactively detect trailing edge by picking seed edge and the max angle.

        Parameters
        ----------
        initial_angle : float, default: 0.05
            Initial value of the maximum angle between two edges which still lets them
            be connected as a part of the trailing edge.

        Returns
        -------
        array
            Indices of the detected trailing edge edges.
        """
        plotter = pv.Plotter()
        plotter.add_mesh(self.as_polydata(), opacity=0.5, label=f"{self.label} surface")

        dual = self.msh.compute_dual()
        pos = self.positions
        lines = self.msh.line_data
        cell = pv.CellArray.from_regular_cells(np.astype(lines, int))
        edge_pd = pv.PolyData(pos, lines=cell)
        # Add the edges as a separate mesh to allow picking them separately
        plotter.add_mesh(
            edge_pd, color="black", line_width=2, label=f"{self.label} edges"
        )

        @dataclass
        class _TESelectionState:
            seed: int | None
            max_angle: float
            selected: npt.NDArray[np.uint]
            selection_mesh: pv.Actor | None

        state = _TESelectionState(
            seed=None,
            max_angle=initial_angle,
            selected=np.array([], dtype=np.uint),
            selection_mesh=None,
        )

        def update_selection():
            if state.seed is None:
                return

            res = self._propagate_edges([state.seed], state.max_angle, dual)
            state.selected = res

            if state.selection_mesh is not None:
                plotter.remove_actor(state.selection_mesh)

            selected_indices = state.selected
            if len(selected_indices) == 0:
                return

            sel_lines = lines[selected_indices, :]

            sel_cell = pv.CellArray.from_regular_cells(np.astype(sel_lines, int))
            sel_pd = pv.PolyData(pos, lines=sel_cell)
            state.selection_mesh = plotter.add_mesh(sel_pd, color="red", line_width=5)
            plotter.render()

        def on_pick(edge_msh: pv.UnstructuredGrid):
            # Get the picked edge index from the mesh with one line
            edge_id = edge_msh.cell_data["vtkOriginalCellIds"][0]
            state.seed = edge_id
            print("Picked edge index:", edge_id)
            update_selection()

        def on_slider(value):
            state.max_angle = value
            update_selection()

        plotter.enable_element_picking(
            callback=on_pick, show_message=True, picker="Volume"
        )
        plotter.add_slider_widget(
            callback=on_slider,
            rng=[0, np.pi / 2],
            value=initial_angle,
            title="Max Angle (rad)",
            pointa=(0.7, 0.1),
            pointb=(0.9, 0.1),
        )
        plotter.add_axes()

        plotter.show()
        return state.selected

    def detect_trailing_edge_automatic(
        self,
        inflow_direction: npt.ArrayLike,
        max_angle: float,
        seed_angle: float = np.radians(80),
    ) -> npt.NDArray[np.uint]:
        """Automatically detect trailing edges based on inflow direction and angle.

        Parameters
        ----------
        inflow_direction : array_like
            Direction vector of the expected inflow.

        max_angle : float
            Maximum allowed angle between neighboring edges during propagation.

        seed_angle : float, optional
            Angle relative to the inflow vector used to identify initial seed edges.
            Defaults to 80 degrees.

        Returns
        -------
        array
            Indices of the detected trailing edge edges.
        """
        inflow_dir = np.asarray(inflow_direction, dtype=np.double).reshape((3,))
        inflow_dir /= np.linalg.norm(inflow_dir)

        seeds = []
        for i_line in range(self.msh.n_lines):
            ip1, ip2 = self.msh.get_line_points(i_line)
            p1 = self.positions[ip1]
            p2 = self.positions[ip2]
            d = p2 - p1
            norm = np.linalg.norm(d)
            if norm > 0:
                d /= norm
            cos_theta = np.abs(np.dot(d, inflow_dir))
            angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))
            if angle >= seed_angle:
                seeds.append(i_line)

        return self._propagate_edges(seeds, max_angle, self.msh.compute_dual())

    def save(self, serializer: CallableSerializer) -> HirearchicalMap:
        """Save geometry into a HirearchicalMap.

        This method is used for serialization of the object.

        Returns
        -------
        HirearchicalMap
            The contents of the class serialized into a :class:`HirearchicalMap`.
        """
        out = HirearchicalMap()
        out.insert_array("positions", self.positions)
        mesh_group = mesh_to_serial(self.msh)
        out.insert_hirearchical_map("mesh", mesh_group)
        rf_group = rf_to_serial(self.reference_frame, serializer)
        out.insert_hirearchical_map("reference_frame", rf_group)
        return out

    @classmethod
    def load(
        cls, label: str, group: HirearchicalMap, deserializer: CallableDeserializer
    ) -> Self:
        """Load the geometry from a HirearchicalMap.

        This method is used for de-serialization.

        Parameters
        ----------
        label : str
            Label by which to identify the geometry by in plotting and analysis.

        group : HirearchicalMap
            A :class:`HirearchicalMap` object, which contains the data created
            by a call to :meth:`Geometry.save`.

        Returns
        -------
        Self
            Geometry object which has been de-serialized.
        """
        positions = group.get_array("positions")
        mesh_group = group.get_hirearchical_map("mesh")
        rf_group = group.get_hirearchical_map("reference_frame")

        msh = mesh_from_serial(mesh_group)
        rf = rf_from_serial(rf_group, deserializer)

        return cls(label=label, reference_frame=rf, mesh=msh, positions=positions)

    def __eq__(self, other) -> bool:
        """Check for equality with another object.

        If the other object is not a :class:`Geometry`, false is returned.
        """
        if not isinstance(other, Geometry):
            return False
        return (
            self.label == other.label
            and self.msh == other.msh
            and np.allclose(self.positions, other.positions)
        )

    @property
    def normals(self) -> npt.NDArray[np.double]:
        r"""Normals to geometry surfaces in the global reference frame.

        This property computes the unit normal vectors to each surface. The normal
        vector is computed as:

        .. math::

            \vec{n} = \sum\limits_{i = 0}^N \left( \vec{r}_{\mod(i + 1, N)} - \vec{r}_{i}
            \right) \times \left( \vec{r}_{i} - \vec{r}_{\mod(i - 1, N)} \right)

        This vector is then normalized. In essence this means, that for any element, which
        is not a triangle, the normal is not guaranteed to be accurate at all points if
        the element is not planar.

        Returns
        -------
        (N, 3) array
            Array of unit normal vectors for surfaces of the geometry in the global
            reference frame.
        """
        n = self.msh.surface_normal(self.positions)
        return self.reference_frame.from_parent_vector(n, out=n)

    @property
    def centers(self) -> npt.NDArray[np.double]:
        r"""Compute centers of geometry sufraces in the global reference frame.

        These are computed by simply finding the average position vector:

        .. math::

            \vec{r}_C = \frac{1}{N} \sum\limits_{i = 0}^N \vec{r}_i

        Returns
        -------
        (N, 3) array
            Array of position vectors of surface centers in  in the global reference
            frame.
        """
        n = self.msh.surface_average_vec3(self.positions)
        return self.reference_frame.from_parent_vector(n, out=n)


@dataclass(frozen=True, eq=False)
class GeometryInfo:
    """Class containing information about geometry."""

    rf: ReferenceFrame
    pos: npt.NDArray[np.double]
    closed: bool
    points: slice
    lines: slice
    surfaces: slice

    def __eq__(self, other) -> bool:
        """Equality check."""
        if not isinstance(other, GeometryInfo):
            return False
        return (
            self.rf == other.rf
            and np.allclose(self.pos, other.pos)
            and self.closed == other.closed
            and self.points == other.points
            and self.lines == other.lines
            and self.surfaces == other.surfaces
        )

    def save(self, serializer: CallableSerializer) -> HirearchicalMap:
        """Save geometry info into a HirearchicalMap."""
        out = HirearchicalMap()
        out.insert_array("pos", self.pos)
        out.insert_hirearchical_map("reference_frame", rf_to_serial(self.rf, serializer))
        out.insert_int("closed", self.closed)
        out.insert_int("points_start", self.points.start)
        out.insert_int("points_stop", self.points.stop)
        out.insert_int("lines_start", self.lines.start)
        out.insert_int("lines_stop", self.lines.stop)
        out.insert_int("surfaces_start", self.surfaces.start)
        out.insert_int("surfaces_stop", self.surfaces.stop)
        return out

    @classmethod
    def load(cls, group: HirearchicalMap, deserializer: CallableDeserializer) -> Self:
        """Load geometry info from a HirearchicalMap."""
        pos = group.get_array("pos")
        rf = rf_from_serial(group.get_hirearchical_map("reference_frame"), deserializer)
        closed = bool(group.get_int("closed"))
        points = slice(
            group.get_int("points_start"),
            group.get_int("points_stop"),
        )
        lines = slice(
            group.get_int("lines_start"),
            group.get_int("lines_stop"),
        )
        surfaces = slice(
            group.get_int("surfaces_start"),
            group.get_int("surfaces_stop"),
        )
        return cls(rf, pos, closed, points, lines, surfaces)


@dataclass(frozen=True)
class SimulationGeometry(Mapping):
    """Class which is the result of combining multiple geometries together.

    Parameters
    ----------
    *geometries: Geometry
        The individual geometries to add in the :class:`SimulationGeometry`.

    Examples
    --------
    This example simply loads the example wing and fuselage, then puts them together
    into a single :class:`SimulationGeometry` object.

    .. jupyter-execute::

        >>> import meshio as mio
        >>> import pyvista as pv
        >>> import pyvl
        >>> from pyvl.examples import example_file_name
        >>>
        >>> pv.set_plot_theme("document")
        >>> pv.set_jupyter_backend("html")
        >>> pv.global_theme.show_edges = True

    First, the wing is loaded:

    .. jupyter-execute::

        >>> m_wing = mio.read(example_file_name("wing1.obj"))
        >>> rf_wing = pyvl.ReferenceFrame((-5, 0, 1.5))
        >>> geo_wing = pyvl.Geometry.from_meshio("wing", rf_wing, m_wing)

    Next, the fuselage:

    .. jupyter-execute::

        >>> m_fus = mio.read(example_file_name("fus1.obj"))
        >>> rf_fus = pyvl.ReferenceFrame()
        >>> geo_fus = pyvl.Geometry.from_meshio("fuselage", rf_fus, m_fus)

    Lastly, the two are combined, and the :class:`SimulationGeometry` is displayed
    afterwards.

    .. jupyter-execute::

        >>> sim_geo = pyvl.SimulationGeometry(geo_wing, geo_fus)
        >>> sim_geo.polydata_at_time(0.0).plot(interactive=False)

    """

    _info: dict[str, GeometryInfo]
    mesh: Mesh
    dual: Mesh

    @classmethod
    def from_geometries(cls, *geometries: Geometry) -> SimulationGeometry:
        """Create a SimulationGeometry from multiple Geometry objects.

        Parameters
        ----------
        *geometries: Geometry
            The individual geometries to add in the :class:`SimulationGeometry`.

        Returns
        -------
        SimulationGeometry
            The created :class:`SimulationGeometry` object.
        """
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
                ln1, ln2 = dual.get_line_points(il)
                if ln1 == INVALID_ID or ln2 == INVALID_ID:
                    closed = False
                    break
            info[g.label] = GeometryInfo(
                g.reference_frame,
                g.positions,
                closed,
                slice(n_points, n_points + c_p),
                slice(n_lines, n_lines + c_l),
                slice(n_surfaces, n_surfaces + c_s),
            )
            n_points += c_p
            n_lines += c_l
            n_surfaces += c_s

        merged_mesh = Mesh.merge_meshes(*meshes)
        return cls(
            _info=info,
            mesh=merged_mesh,
            dual=merged_mesh.compute_dual(),
        )

    def __getitem__(self, key: str) -> GeometryInfo:
        """Return the Geometry corresponding to the key."""
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

    @property
    def n_surfaces(self) -> int:
        """Return the total number of surfaces in the simulation geometry."""
        return self.mesh.n_surfaces

    @property
    def n_lines(self) -> int:
        """Return the total number of lines in the simulation geometry."""
        return self.mesh.n_lines

    @property
    def n_points(self) -> int:
        """Return the total number of points in the simulation geometry."""
        return self.mesh.n_points

    def positions_at_time(
        self, t: float, out_pos: npt.NDArray[np.double] | None = None
    ) -> npt.NDArray[np.double]:
        """Return the point positions at the specified time.

        This uses the different reference frames of the individual :class:`Geometry`
        objects to determine their positions in the global reference frame.

        Parameters
        ----------
        t : float
            Time at which to get the positions and velocities at.

        out_pos : (N, 3) array, optional
            Optional array to store the positions in. If not provided, a new array will
            be created.

        Returns
        -------
        (N, 3) array
            Array of position vectors of individual points of the geometries.
        """
        if out_pos is None:
            pos = np.empty((self.n_points, 3), np.double)
        else:
            pos = out_pos

        for geo_name in self._info:
            info = self._info[geo_name]
            rf = info.rf
            # pos = np.array(info.pos)
            rf.to_global_position(
                x=pos,
                time=t,
                # This output array should be fine, since it should be contiguous
                out=pos[info.points],
            )

        return pos

    def geometry_at_time(
        self,
        t: float,
        out_pos: npt.NDArray[np.double] | None = None,
        out_vel: npt.NDArray[np.double] | None = None,
    ) -> tuple[npt.NDArray[np.double], npt.NDArray[np.double]]:
        """Return the point positions and velocities at the specified time.

        This uses the different reference frames of the individual :class:`Geometry`
        objects to determine their positions and velocities in the global reference frame.

        Parameters
        ----------
        t : float
            Time at which to get the positions and velocities at.

        out_pos : (N, 3) array, optional
            Optional array to store the positions in. If not provided, a new array will
            be created.

        out_vel : (N, 3) array, optional
            Optional array to store the velocities in. If not provided, a new array will
            be created.

        Returns
        -------
        (N, 3) array
            Array of position vectors of individual points of the geometries.
        (N, 3) array
            Array of velocity vectors of individual points of the geometries.
        """
        if out_pos is None:
            pos = np.empty((self.n_points, 3), np.double)
        else:
            pos = out_pos

        if out_vel is None:
            vel = np.empty((self.n_points, 3), np.double)
        else:
            vel = out_vel

        for geo_name in self._info:
            info = self._info[geo_name]
            rf = info.rf
            # pos = np.array(info.pos)
            v = np.zeros_like(vel[info.points])
            rf.to_global_velocity(
                position=pos,
                velocity=v,
                time=t,
                # These output arrays should be fine, since they should be contiguous
                out_position=pos[info.points],
                out_velocity=vel[info.points],
            )

        return pos, vel

    def polydata_at_time(self, t: float) -> pv.PolyData:
        """Return the geometry as polydata at the specified time.

        This uses the :meth:`SimulationGeometry.positions_at_time` to determine
        the positions of individual mesh points.

        Parameters
        ----------
        t : float
            Time at which to get the mesh at.

        Returns
        -------
        pyvista.PolyData
            :class:`pyvista.PolyData` object which represents the surface mesh.
        """
        pos = self.positions_at_time(t)
        faces = mesh_to_polydata_faces(self.mesh)
        pd = pv.PolyData.from_irregular_faces(pos, faces)
        return pd

    def polydata_edges_at_time(self, t: float) -> pv.PolyData:
        """Return the edges of geometry as polydata at the specified time.

        This uses the :meth:`SimulationGeometry.positions_at_time` to determine
        the positions of individual mesh points.

        Parameters
        ----------
        t : float
            Time at which to get the mesh at.

        Returns
        -------
        pyvista.PolyData
            :class:`pyvista.PolyData` object which represents the line mesh.
        """
        pos = self.positions_at_time(t)
        lines = self.mesh.line_data
        cell = pv.CellArray.from_regular_cells(np.astype(lines, int))
        pd = pv.PolyData(pos, lines=cell)
        return pd

    def te_normal_criterion(self, crit: float) -> npt.NDArray[np.uint]:
        """Identify edges, for which the normals of neighbouring surfaces meet dp crit.

        This function returns the array with indices of all mesh edges which meet the
        specific criterion. In this case, this criterion is that the edge should border
        two surfaces, whose unit normals have dot product less than the value specified
        by ``crit``. This can be used to identify closed trailing edges.

        Parameters
        ----------
        crit : float
            Minimum value of dot product of neighbouring surface unit normal vectors
            before an edge is considered.

        Returns
        -------
        array
            Array of sorted indices of edges which meet the criterion.
        """
        normals = self.mesh.surface_normal(
            np.concatenate([info.pos for info in self._info.values()], axis=0)
        )
        return self.dual.dual_normal_criterion(crit, normals)

    def te_free_criterion(self) -> npt.NDArray[np.uint]:
        """Identify edges, which have only one surface attached.

        This function returns the array with indices of all mesh edges which meet the
        specific criterion. In this case, this criterion is that the edge should be a
        part of one and only one surface.

        Returns
        -------
        array
            Array of sorted indices of edges which meet the criterion.
        """
        return self.dual.dual_free_edges()

    def line_adjecency_information(
        self, lines: Sequence[int] | npt.NDArray[np.integer]
    ) -> tuple[npt.NDArray[np.uint], npt.NDArray[np.uint]]:
        """Return the arrays of indices of nodes and surfaces related with the edges.

        The first array returned is the array of bordering nodes for each edge specified.
        Along with it, an array with indices of surfaces bordering the line is returned.

        Parameters
        ----------
        lines : Sequence of N int or (N,) array
            Lines for which the information should be returned.

        Returns
        -------
        (N, 2) array
            Array with indices of nodes which the lines connect.
        (N, 2) array
            Array with indices of surfaces which the lines border.
        """
        bordering_nodes = np.empty((len(lines), 2), np.uint)
        adjacent_surfaces = np.empty((len(lines), 2), np.uint)
        for i, line_id in enumerate(lines):
            primal_line = self.mesh.get_line_points(line_id)
            dual_line = self.dual.get_line_points(line_id)
            bordering_nodes[i, :] = primal_line
            adjacent_surfaces[i, :] = dual_line
        return (bordering_nodes, adjacent_surfaces)

    def save(self, serializer: CallableSerializer) -> HirearchicalMap:
        """Save the simulation geometry into a HirearchicalMap.

        Returns
        -------
        HirearchicalMap
            State serialized into a :class:`HirearchicalMap` object.
        """
        out = HirearchicalMap()
        out_info = HirearchicalMap()
        for geo_name in self._info:
            geo_group = self._info[geo_name].save(serializer)
            out_info.insert_hirearchical_map(geo_name, geo_group)

        mesh_group = mesh_to_serial(self.mesh)
        out.insert_hirearchical_map("mesh", mesh_group)
        out.insert_hirearchical_map("info", out_info)
        return out

    @classmethod
    def load(cls, group: HirearchicalMap, deserializer: CallableDeserializer) -> Self:
        """Load the simulation geometry from a HirearchicalMap.

        Parameters
        ----------
        group : HirearchicalMap
            Serialized state created by a call to :meth:`SimulationGeometry.save`.

        Returns
        -------
        Self
            De-serialized :class:`SimulationGeometry` object.
        """
        info_group = group.get_hirearchical_map("info")
        info: dict[str, GeometryInfo] = {}
        for geo_name in info_group:
            geo_info = GeometryInfo.load(
                info_group.get_hirearchical_map(geo_name), deserializer
            )
            info[geo_name] = geo_info

        mesh_group = group.get_hirearchical_map("mesh")
        mesh = mesh_from_serial(mesh_group)

        return cls(_info=info, mesh=mesh, dual=mesh.compute_dual())

    def __eq__(self, other) -> bool:
        """Check for equality."""
        if not isinstance(other, SimulationGeometry):
            return False

        return (
            self.n_points == other.n_points
            and self.n_lines == other.n_lines
            and self.n_surfaces == other.n_surfaces
            and self._info == other._info
            and self.mesh == other.mesh
            and self.dual == other.dual
        )


def geometry_show_pyvista(
    geometries: Iterable[Geometry], plt: pv.Plotter | None = None
) -> None:
    """Show the geometry using PyVista."""
    show = plt is None
    if plt is None:
        plt = pv.Plotter()  # theme=pv.themes.DocumentProTheme())
        plt.theme.show_edges = True
        plt.theme.show_scalar_bar = False

    for geo in geometries:
        plt.add_mesh(geo.as_polydata(), label=geo.label)

    if show:
        plt.show()
