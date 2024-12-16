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
    part_mapping: dict[str, slice]

    def get_part_solution(self, name: str) -> npt.NDArray[np.float64]:
        """Get part of the solution corresponding to the solution of a part."""
        return self.circulations[:, self.part_mapping[name]]


@dataclass
class _GeoInfo:
    name: str
    offset: np.uint32
    msh: Mesh
    rf: ReferenceFrame
    closed: bool
    updated: bool


class _SimulationGeometryState:
    """Simulation geometry containing several geometries in different reference frames."""

    msh: Mesh
    info: tuple[_GeoInfo, ...]
    _time: float
    _normals: npt.NDArray[np.float64]
    _cpts: npt.NDArray[np.float64]

    def __init__(self, geo: Iterable[Geometry] = {}, /, *args: Geometry) -> None:
        position_list: list[npt.NDArray[np.float64]] = []
        element_counts: list[npt.NDArray[np.uint64]] = []
        indices: list[npt.NDArray[np.uint64]] = []
        n_pts = 0
        info_list: list[_GeoInfo] = []
        all_geos = (*geo, *args)
        for g in all_geos:
            nelm, conn = g.msh.to_element_connectivity()
            conn += n_pts
            offsets = np.pad(np.cumsum(nelm), (1, 0))
            faces = [conn[offsets[i] : offsets[i + 1]] for i in range(nelm.size)]
            pos = g.msh.positions
            position_list.append(pos)
            element_counts.append(nelm)
            indices.extend(faces)
            dual = g.msh.compute_dual()
            closed = True
            for il in range(dual.n_lines):
                ln = dual.get_line(il)
                if ln.begin == INVALID_ID or ln.end == INVALID_ID:
                    closed = False
                    break
            info = _GeoInfo(
                g.label, np.uint32(n_pts), g.msh, g.reference_frame, closed, False
            )
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
            if rf == info.rf:
                info.updated = False
                break
            else:
                info.updated = True
                info.rf = rf
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

    def adjust_circulations(self, circulations: npt.NDArray[np.float64]) -> None:
        """Adjust circulation of closed surfaces in the mesh."""
        for info in self.info:
            if not info.updated or not info.closed:
                continue
            i_begin = info.offset
            i_end = i_begin + info.msh.n_surfaces
            circulations[i_begin:i_end] -= np.mean(circulations[i_begin:i_end])

    def as_polydata(self) -> pv.PolyData:
        """Convert SimulationGeometry into PyVista's PolyData."""
        positions = self.msh.positions
        nper_elem, indices = self.msh.to_element_connectivity()
        offsets = np.pad(np.cumsum(nper_elem), (1, 0))
        faces = [indices[offsets[i] : offsets[i + 1]] for i in range(nper_elem.size)]
        pd = pv.PolyData.from_irregular_faces(positions, faces)
        return pd


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
        geometry.time = time
        # Get mesh properties
        control_points = geometry._cpts
        normals = geometry._normals
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
    result_mapping = {
        g.name: slice(g.offset, np.uint32(g.offset + g.msh.n_surfaces))
        for g in geometry.info
    }
    return SolverResults(out_array, times, result_mapping)


def compute_induced_velocities(
    msh: Mesh,
    circulation: npt.NDArray[np.float64],
    model_setting: ModelSettings,
    positions: npt.NDArray,
) -> npt.NDArray:
    """Compute velocity induced by the mesh with circulation."""
    induction_matrix = msh.induction_matrix(model_setting.vortex_limit, positions)
    return np.linalg.vecdot(induction_matrix, circulation[None, :, None], axis=1)
