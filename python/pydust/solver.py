"""Implementation of the flow solver."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Literal, Self

import numpy as np
import numpy.typing as npt
import scipy.linalg as la

from pydust.cdust import ReferenceFrame
from pydust.fio.io_common import HirearchicalMap, SerializationFunction
from pydust.fio.io_hdf5 import serialize_hdf5
from pydust.fio.io_json import serialize_json
from pydust.geometry import SimulationGeometry
from pydust.settings import SolverSettings


class SolverState:
    """State of the solver at a specific moment."""

    positions: npt.NDArray[np.float64]
    normals: npt.NDArray[np.float64]
    control_points: npt.NDArray[np.float64]
    circulation: npt.NDArray[np.float64]
    geometry: SimulationGeometry
    settings: SolverSettings
    iteration: int

    def __init__(
        self,
        geometry: SimulationGeometry,
        settings: SolverSettings,
    ) -> None:
        self.geometry = geometry
        self.positions = np.empty((geometry.n_points, 3), np.float64)
        self.normals = np.empty((geometry.n_surfaces, 3), np.float64)
        self.control_points = np.empty((geometry.n_surfaces, 3), np.float64)
        self.settings = settings

    def save(self) -> HirearchicalMap:
        """Serialize current state to a HirearchicalMap."""
        out = HirearchicalMap()
        out.insert_array("positions", self.positions)
        out.insert_array("normals", self.normals)
        out.insert_array("control_points", self.control_points)
        out.insert_array("circulation", self.circulation)
        out.insert_int("iteration", self.iteration)
        out.insert_hirearchycal_map("solver_settings", self.settings.save())
        out.insert_hirearchycal_map("simulation_geometry", self.geometry.save())
        return out

    @classmethod
    def load(cls, hmap: HirearchicalMap) -> Self:
        """Deserialize current state from a HirearchicalMap."""
        geometry = SimulationGeometry.load(
            hmap.get_hirearchical_map("simulation_geometry")
        )
        settings = SolverSettings.load(hmap.get_hirearchical_map("solver_settings"))
        self = cls(geometry=geometry, settings=settings)
        self.iteration = hmap.get_int("iteration")
        self.positions[:] = hmap.get_array("positions")
        self.normals[:] = hmap.get_array("normals")
        self.control_points[:] = hmap.get_array("control_points")
        self.circulation[:] = hmap.get_array("circulation")

        return self


OutputFileType = Literal["HDF5", "JSON"]
NamingFunction = Callable[[int, float], str]


@dataclass(init=False, eq=False, frozen=True)
class OutputSettings:
    """Settings to control the output from a solver."""

    naming_callback: Callable[[int, float], str]
    serialization_fn: SerializationFunction

    def __init__(
        self, ftype: OutputFileType, naming_callback: Callable[[int, float], str]
    ) -> None:
        serialization_fn: SerializationFunction
        match ftype:
            case "HDF5":
                serialization_fn = serialize_hdf5
            case "JSON":
                serialization_fn = serialize_json
            case _:
                raise ValueError(f"The file type {ftype=} is not valid.")
        object.__setattr__(self, "serialization_fn", serialization_fn)
        object.__setattr__(self, "naming_callback", naming_callback)


@dataclass(frozen=True)
class SolverResults:
    """Class which contains solver results."""

    circulations: npt.NDArray[np.float64]
    times: npt.NDArray[np.float64]


def run_solver(
    geometry: SimulationGeometry,
    settings: SolverSettings,
    output_settings: OutputSettings,
) -> SolverResults:
    """Run the flow solver to obtain specified circulations."""
    times: npt.NDArray[np.float64]
    if settings.time_settings is None:
        times = np.array((0,), np.float64)
    else:
        times = np.arange(settings.time_settings.nt) * settings.time_settings.dt

    i_out = 0

    state = SolverState(geometry, settings)

    out_circulaiton_array = np.empty((len(times), geometry.n_surfaces), np.float64)

    previous_rfs: dict[str, None | ReferenceFrame] = {name: None for name in geometry}

    for iteration, time in enumerate(times):
        state.iteration = iteration
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
            state.positions[info.points] = new_rf.to_global_with_offset(info.pos)
            state.control_points[info.surfaces] = new_rf.to_global_with_offset(
                info.msh.surface_average_vec3(info.pos)
            )
            state.normals[info.surfaces] = new_rf.to_global_without_offset(
                info.msh.surface_normal(info.pos)
            )

        # Compute flow velocity
        element_velocity = settings.flow_conditions.get_velocity(
            time, state.control_points
        )
        # Compute flow penetration at control points
        rhs = np.vecdot(state.normals, -element_velocity, axis=1)  # type: ignore
        # if updated != 0:
        # Compute normal induction
        system_matrix = geometry.mesh.induction_matrix3(
            settings.model_settings.vortex_limit,
            state.positions,
            state.control_points,
            state.normals,
        )

        # Apply the wake model's effect
        if settings.wake_model is not None:
            settings.wake_model.apply_corrections(
                state.control_points, state.normals, system_matrix, rhs
            )

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
                time, geometry, state.positions, circulation, settings.flow_conditions
            )

        iteration_end_time = perf_counter()
        if (
            settings.time_settings is None
            or (settings.time_settings.output_interval is None)
            or (iteration % settings.time_settings.output_interval == 0)
        ):
            output_settings.serialization_fn(
                state.save(), output_settings.naming_callback(iteration, time)
            )
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
