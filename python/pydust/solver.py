"""Implementation of the flow solver."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from time import perf_counter

import h5py
import numpy as np
import numpy.typing as npt
import scipy.linalg as la

from pydust.cdust import ReferenceFrame
from pydust.geometry import SimulationGeometry
from pydust.settings import SolverSettings
from pydust.wake import WakeModel, _load_wake_model, _store_wake_model


class SolverState:
    """State of the solver at a specific moment."""

    positions: npt.NDArray[np.float64]
    normals: npt.NDArray[np.float64]
    control_points: npt.NDArray[np.float64]
    circulation: npt.NDArray[np.float64]
    geometry: SimulationGeometry
    settings: SolverSettings
    wake_model: WakeModel | None
    iteration: int

    def __init__(
        self,
        geometry: SimulationGeometry,
        # settings: SolverSettings,
    ) -> None:
        self.geometry = geometry
        self.positions = np.empty((geometry.n_points, 3), np.float64)
        self.normals = np.empty((geometry.n_surfaces, 3), np.float64)
        self.control_points = np.empty((geometry.n_surfaces, 3), np.float64)
        # self.settings = settings

    def save(self, out_file: h5py.File, comment: str | None = None) -> None:
        """Serialize current state to file."""
        out_file["positions"] = self.positions
        out_file["normals"] = self.normals
        out_file["control_points"] = self.control_points
        out_file["circulation"] = self.circulation
        if self.wake_model is not None:
            wake_group = out_file.create_group("wake_model")
            _store_wake_model(self.wake_model, wake_group)
        out_file["iteration"] = self.iteration
        # TODO: settings
        geometry_group = out_file.create_group("simulation_geometry")
        self.geometry.save(geometry_group)

        if comment:
            out_file["comment"] = comment

    @classmethod
    def load(cls, in_file: h5py.File) -> None:
        """Serialize current state to file."""
        positions = in_file["positions"]
        normals = in_file["normals"]
        control_points = in_file["control_points"]
        circulation = in_file["circulation"]
        geometry_group = in_file["simulation_geometry"]

        assert isinstance(positions, h5py.Dataset)
        assert isinstance(normals, h5py.Dataset)
        assert isinstance(control_points, h5py.Dataset)
        assert isinstance(circulation, h5py.Dataset)
        assert isinstance(geometry_group, h5py.Group)

        # TODO: settings
        geometry = SimulationGeometry.load(geometry_group)
        self: SolverState = cls(geometry)

        self.positions[:] = positions[()]
        self.normals[:] = normals[()]
        self.control_points[:] = control_points[()]
        self.circulation[:] = circulation[()]
        if "wake_model" in in_file:
            wake_group = in_file["wake_model"]
            assert isinstance(wake_group, h5py.Group)
            self.wake_model = _load_wake_model(wake_group)
        else:
            self.wake_model = None
        # TODO: settings


@dataclass(frozen=True)
class SolverResults:
    """Class which contains solver results."""

    circulations: npt.NDArray[np.float64]
    times: npt.NDArray[np.float64]


def run_solver(geometry: SimulationGeometry, settings: SolverSettings) -> SolverResults:
    """Run the flow solver to obtain specified circulations."""
    times: npt.NDArray[np.float64]
    if settings.time_settings is None:
        times = np.array((0,), np.float64)
    else:
        times = np.arange(settings.time_settings.nt) * settings.time_settings.dt

    i_out = 0

    out_circulaiton_array = np.empty((len(times), geometry.n_surfaces), np.float64)

    pos = np.empty((geometry.n_points, 3), np.float64)
    norm = np.empty((geometry.n_surfaces, 3), np.float64)
    cpts = np.empty((geometry.n_surfaces, 3), np.float64)

    previous_rfs: dict[str, None | ReferenceFrame] = {name: None for name in geometry}

    for iteration, time in enumerate(times):
        iteration_begin_time = perf_counter()
        updated = 0
        for geo_name in geometry:
            info = geometry[geo_name]
            new_rf = info.rf.at_time(time)
            if new_rf == previous_rfs[geo_name]:
                # The geometry of that particular part did not change.
                continue
            updated += 1
            previous_rfs[geo_name] = new_rf
            pos[info.points] = new_rf.to_global_with_offset(info.pos)
            cpts[info.surfaces] = new_rf.to_global_with_offset(
                info.msh.surface_average_vec3(info.pos)
            )
            norm[info.surfaces] = new_rf.to_global_without_offset(
                info.msh.surface_normal(info.pos)
            )

        # Compute flow velocity
        element_velocity = settings.flow_conditions.get_velocity(time, cpts)
        # Compute flow penetration at control points
        rhs = np.vecdot(norm, -element_velocity, axis=1)  # type: ignore
        # if updated != 0:
        # Compute normal induction
        system_matrix = geometry.mesh.induction_matrix3(
            settings.model_settings.vortex_limit, pos, cpts, norm
        )

        # Apply the wake model's effect
        if settings.wake_model is not None:
            settings.wake_model.apply_corrections(cpts, norm, system_matrix, rhs)

        # Decompose the system matrix to allow for solving multiple times
        decomp = la.lu_factor(system_matrix, overwrite_a=True)

        # Solve the linear system
        # By setting overwrite_b=True, rhs is where the output is written to
        circulation = la.lu_solve(decomp, rhs, overwrite_b=True)
        # Adjust circulations
        for geo_name in geometry:
            info = geometry[geo_name]
            if not info.closed:
                continue
            circulation[info.surfaces] -= np.mean(circulation[info.surfaces])

        # update the wake model
        if settings.wake_model is not None:
            settings.wake_model.update(
                time, geometry, pos, circulation, settings.flow_conditions
            )

        iteration_end_time = perf_counter()
        if (
            settings.time_settings is None
            or (settings.time_settings.output_interval is None)
            or (iteration % settings.time_settings.output_interval == 0)
        ):
            out_circulaiton_array[i_out, :] = circulation
            i_out += 1
        print(
            f"Finished iteration {iteration} out of {len(times)} in "
            f"{iteration_end_time - iteration_begin_time:g} seconds."
        )

    # del system_matrix, line_buffer, velocity, rhs
    return SolverResults(
        out_circulaiton_array,
        times,
    )


def compute_induced_velocities(
    simulation_geometry: SimulationGeometry,
    settings: SolverSettings,
    results: SolverResults,
    positions: npt.NDArray,
    workers: int | None = None,
) -> npt.NDArray:
    """Compute velocity induced by the mesh with circulation."""
    out_vec = np.empty((len(results.times), positions.shape[0], 3), np.float64)

    def _velocity_compute_function(
        i, time: float, circulation: npt.NDArray[np.float64]
    ) -> None:
        """Compute velocity induction."""
        points = simulation_geometry.positions_at_time(time)
        ind_mat = simulation_geometry.mesh.induction_matrix(
            settings.model_settings.vortex_limit,
            points,
            positions,
        )
        np.vecdot(ind_mat, circulation[None, :, None], out=out_vec[i, ...], axis=1)  # type: ignore

    with ThreadPoolExecutor(workers) as executor:
        executor.map(
            _velocity_compute_function,
            range(results.times.size),
            results.times,
            results.circulations,
        )

    return out_vec
