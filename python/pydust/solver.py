"""Implementation of the flow solver."""

from collections.abc import Iterable
from dataclasses import dataclass
from time import perf_counter

import numpy as np
import numpy.typing as npt
import pyvista as pv
import scipy.linalg as la

from pydust.cdust import INVALID_ID, Mesh, ReferenceFrame
from pydust.geometry import Geometry
from pydust.settings import ModelSettings, SolverSettings


@dataclass(frozen=True)
class SolverResults:
    """Class which contains solver results."""

    circulations: npt.NDArray[np.float64]
    times: npt.NDArray[np.float64]
    point_mapping: dict[str, slice]
    line_mapping: dict[str, slice]
    surface_mapping: dict[str, slice]

    def get_part_solution(self, name: str) -> npt.NDArray[np.float64]:
        """Get part of the solution corresponding to the solution of a part."""
        return self.circulations[:, self.surface_mapping[name]]


@dataclass
class _GeoInfo:
    name: str
    msh: Mesh
    rf: ReferenceFrame
    closed: bool
    updated: bool


class _SimulationGeometryState:
    """Simulation geometry containing several geometries in different reference frames."""

    info: tuple[_GeoInfo, ...]
    msh: Mesh
    time: float
    normals: npt.NDArray[np.float64]
    cpts: npt.NDArray[np.float64]
    positions: npt.NDArray[np.float64]

    def __init__(self, geo: Iterable[Geometry] = {}, /, *args: Geometry) -> None:
        info_list: list[_GeoInfo] = []
        meshes: list[Mesh] = []
        all_geos = (*geo, *args)
        n_surfaces = 0
        n_pts = 0
        for g in all_geos:
            meshes.append(g.msh)
            dual = g.msh.compute_dual()
            closed = True
            for il in range(dual.n_lines):
                ln = dual.get_line(il)
                if ln.begin == INVALID_ID or ln.end == INVALID_ID:
                    closed = False
                    break
            info = _GeoInfo(g.label, g.msh, g.reference_frame, closed, False)
            n_surfaces += g.msh.n_surfaces
            n_pts += g.msh.n_points
            info_list.append(info)

        self.info = tuple(info_list)
        self.time = 0.0
        self.normals = np.empty((n_surfaces, 3), np.float64)
        self.cpts = np.empty((n_surfaces, 3), np.float64)
        self.positions = np.empty((n_pts, 3), np.float64)
        self.msh = Mesh.merge_meshes(*meshes)
        self.time = 0.0

    def set_time(self, t: float) -> None:
        if not isinstance(t, float):
            raise TypeError(f"Time is not a float but is instead {type(t)}.")
        self.time = t
        point_mapping = self.point_slices
        surface_mapping = self.surface_slices
        update_cnt = 0
        for info in self.info:
            rf = info.rf.at_time(t)
            if rf == info.rf:
                info.updated = False
                break
            else:
                info.updated = True
                info.rf = rf
            update_cnt += 1
            # Transform in place
            rf.to_parent_with_offset(
                info.msh.positions, self.positions[point_mapping[info.name], :]
            )
            rf.to_parent_with_offset(
                info.msh.surface_centers, self.cpts[surface_mapping[info.name], :]
            )
            rf.to_parent_without_offset(
                info.msh.surface_normals, self.normals[surface_mapping[info.name], :]
            )
        if update_cnt != 0:
            self.msh.positions = self.positions

    def adjust_circulations(self, circulations: npt.NDArray[np.float64]) -> None:
        """Adjust circulation of closed surfaces in the mesh."""
        offset = 0
        n_surf = 0
        for info in self.info:
            offset += n_surf
            n_surf = info.msh.n_surfaces
            if not info.updated or not info.closed:
                continue
            circulations[offset : offset + n_surf] -= np.mean(
                circulations[offset : offset + n_surf]
            )

    def as_polydata(self) -> pv.PolyData:
        """Convert SimulationGeometry into PyVista's PolyData."""
        positions = self.msh.positions
        nper_elem, indices = self.msh.to_element_connectivity()
        offsets = np.pad(np.cumsum(nper_elem), (1, 0))
        faces = [indices[offsets[i] : offsets[i + 1]] for i in range(nper_elem.size)]
        pd = pv.PolyData.from_irregular_faces(positions, faces)
        return pd

    @property
    def surface_slices(self) -> dict[str, slice]:
        """Return slices which can be used to index into arrays of surfaces."""
        n_surf = 0
        out: dict[str, slice] = dict()
        for i in self.info:
            ns = i.msh.n_surfaces
            out[i.name] = slice(n_surf, n_surf + ns)
            n_surf += ns
        return out

    @property
    def line_slices(self) -> dict[str, slice]:
        """Return slices which can be used to index into arrays of lines."""
        n_line = 0
        out: dict[str, slice] = dict()
        for i in self.info:
            nl = i.msh.n_lines
            out[i.name] = slice(n_line, n_line + nl)
            n_line += nl
        return out

    @property
    def point_slices(self) -> dict[str, slice]:
        """Return slices which can be used to index into arrays of point."""
        n_point = 0
        out: dict[str, slice] = dict()
        for i in self.info:
            np = i.msh.n_points
            out[i.name] = slice(n_point, n_point + np)
            n_point += np
        return out


def run_solver(geometries: Iterable[Geometry], settings: SolverSettings) -> SolverResults:
    """Run the flow solver to obtain specified circulations."""
    geometry = _SimulationGeometryState(geometries)
    del geometries
    times: npt.NDArray[np.float64]
    if settings.time_settings is None:
        times = np.array((0,), np.float64)
    else:
        times = np.arange(settings.time_settings.nt) * settings.time_settings.dt

    n_elements = geometry.msh.n_surfaces
    n_lines = geometry.msh.n_lines
    out_array = np.empty((len(times), n_elements), np.float64)
    system_matrix = np.empty((n_elements, n_elements), np.float64)
    line_buffer = np.empty((n_elements, n_lines, 3), np.float64)
    velocity = np.empty((n_elements, 3), np.float64)
    rhs = np.empty((n_elements), np.float64)

    for iteration, time in enumerate(times):
        iteration_begin_time = perf_counter()
        # Update the mesh
        geometry.set_time(time)
        # Get mesh properties
        control_points = geometry.cpts
        normals = geometry.normals
        # Compute flow velocity (if necessary)
        velocity = settings.flow_conditions.get_velocity(time, control_points, velocity)
        # Compute flow penetration at control points
        rhs = np.vecdot(normals, -velocity, rhs, axis=1)  # type: ignore
        # Compute normal induction
        system_matrix = geometry.msh.induction_matrix3(
            settings.model_settings.vortex_limit,
            control_points,
            normals,
            system_matrix,
            line_buffer,
        )
        # Decompose the system matrix to allow for solving multiple times
        decomp = la.lu_factor(system_matrix, overwrite_a=True)
        # Solve the linear system
        # By setting overwrite_b=True, rhs is where the output is written to
        circulation = la.lu_solve(decomp, rhs, overwrite_b=True)
        # Adjust circulations
        geometry.adjust_circulations(circulation)
        iteration_end_time = perf_counter()
        if (
            settings.time_settings is None
            or (settings.time_settings.output_interval is None)
            or (iteration % settings.time_settings.output_interval == 0)
        ):
            out_array[iteration, :] = circulation
        print(
            f"Finished iteration {iteration} out of {len(times)} in "
            f"{iteration_end_time - iteration_begin_time:g} seconds."
        )

    del system_matrix, line_buffer, velocity, rhs
    return SolverResults(
        out_array,
        times,
        geometry.point_slices,
        geometry.line_slices,
        geometry.surface_slices,
    )


def compute_induced_velocities(
    msh: Mesh,
    circulation: npt.NDArray[np.float64],
    model_setting: ModelSettings,
    positions: npt.NDArray,
) -> npt.NDArray:
    """Compute velocity induced by the mesh with circulation."""
    induction_matrix = msh.induction_matrix(model_setting.vortex_limit, positions)
    return np.linalg.vecdot(induction_matrix, circulation[None, :, None], axis=1)
