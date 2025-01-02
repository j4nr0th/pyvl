"""Implementation of the flow solver."""

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable, Literal, Self

import numpy as np
import numpy.typing as npt
import scipy.linalg as la

from pyvl.cvl import ReferenceFrame
from pyvl.fio.io_common import HirearchicalMap, SerializationFunction
from pyvl.fio.io_hdf5 import serialize_hdf5
from pyvl.fio.io_json import serialize_json
from pyvl.geometry import SimulationGeometry
from pyvl.settings import SolverSettings
from pyvl.wake import WakeModel


class SolverResults:
    """Class containing results of a solver."""

    geometry: SimulationGeometry
    circulations: npt.NDArray[np.float64]
    wake_models: list[WakeModel | None]
    settings: SolverSettings

    def __init__(self, geo: SimulationGeometry, settings: SolverSettings):
        self.geometry = geo
        self.settings = SolverSettings(
            flow_conditions=settings.flow_conditions,
            model_settings=settings.model_settings,
            time_settings=settings.time_settings,
        )
        self.wake_models = list()
        self.circulations = np.empty(
            (settings.time_settings.output_times.size, geo.n_surfaces), np.float64
        )


class SolverState:
    """State of the solver at a specific moment."""

    positions: npt.NDArray[np.float64]
    normals: npt.NDArray[np.float64]
    control_points: npt.NDArray[np.float64]
    cp_velocity: npt.NDArray[np.float64]
    circulation: npt.NDArray[np.float64]
    geometry: SimulationGeometry
    settings: SolverSettings
    wake_model: WakeModel | None
    iteration: int

    def __init__(
        self,
        geometry: SimulationGeometry,
        settings: SolverSettings,
        wake_model: WakeModel | None,
    ) -> None:
        self.geometry = geometry
        self.positions = np.empty((geometry.n_points, 3), np.float64)
        self.normals = np.empty((geometry.n_surfaces, 3), np.float64)
        self.control_points = np.empty((geometry.n_surfaces, 3), np.float64)
        self.cp_velocity = np.empty((geometry.n_surfaces, 3), np.float64)
        self.circulation = np.empty((geometry.n_surfaces,), np.float64)
        self.settings = settings
        self.wake_model = wake_model

    def save(self) -> HirearchicalMap:
        """Serialize current state to a HirearchicalMap."""
        out = HirearchicalMap()
        out.insert_array("positions", self.positions)
        out.insert_array("normals", self.normals)
        out.insert_array("control_points", self.control_points)
        out.insert_array("cp_velocity", self.cp_velocity)
        out.insert_array("circulation", self.circulation)
        out.insert_int("iteration", self.iteration)
        out.insert_hirearchycal_map("solver_settings", self.settings.save())
        out.insert_hirearchycal_map("simulation_geometry", self.geometry.save())
        if self.wake_model is not None:
            wake_model = HirearchicalMap()
            wake_model.insert_type("type", type(self.wake_model))
            wake_model.insert_hirearchycal_map("data", self.wake_model.save())
            out.insert_hirearchycal_map("wake_model", wake_model)
        return out

    @classmethod
    def load(cls, hmap: HirearchicalMap) -> Self:
        """Deserialize current state from a HirearchicalMap."""
        geometry = SimulationGeometry.load(
            hmap.get_hirearchical_map("simulation_geometry")
        )
        settings = SolverSettings.load(hmap.get_hirearchical_map("solver_settings"))
        wake_model = None
        if "wake_model" in hmap:
            wm_hmap = hmap.get_hirearchical_map("wake_model")
            wm_type: type[WakeModel] = wm_hmap.get_type("type")
            wake_model = wm_type.load(wm_hmap.get_hirearchical_map("data"))
        self = cls(geometry=geometry, settings=settings, wake_model=wake_model)
        self.iteration = hmap.get_int("iteration")
        self.positions[:] = hmap.get_array("positions")
        self.normals[:] = hmap.get_array("normals")
        self.control_points[:] = hmap.get_array("control_points")
        self.circulation[:] = hmap.get_array("circulation")
        self.cp_velocity[:] = hmap.get_array("cp_velocity")

        return self


OutputFileType = Literal["HDF5", "JSON"]


@dataclass(init=False, eq=False, frozen=True)
class OutputSettings:
    """Settings to control the output from a solver.

    Parameters
    ----------
    ftype : "JSON" or "HDF5"
        File format to write the output as.
    naming_callback : (int, float) -> str | Path
        Callback to use to determine the name of the next file
        to write based on the iteration number and the simulation time.
    """

    naming_callback: Callable[[int, float], str | Path]
    serialization_fn: SerializationFunction

    def __init__(
        self, ftype: OutputFileType, naming_callback: Callable[[int, float], str | Path]
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


def run_solver(
    geometry: SimulationGeometry,
    settings: SolverSettings,
    wake_model: WakeModel | None,
    output_settings: OutputSettings | None,
) -> SolverResults:
    """Run the flow solver to obtain specified circulations.

    Parameters
    ----------
    geometry : SimulationGeometry
        Geometry to solver for.
    settings : SolverSettings
        Settings of the solver.
    wake_model : WakeModel, optional
        Model with which to model the wake with.
    outpu_settings : OutputSettings, optional
        Settings related to file IO.
    """
    results = SolverResults(geometry, settings)
    times: npt.NDArray[np.float64]
    if settings.time_settings is None:
        times = np.array((0,), np.float64)
    else:
        times = np.arange(settings.time_settings.nt) * settings.time_settings.dt

    i_out = 0

    state = SolverState(geometry, settings, wake_model)

    for iteration, time in enumerate(times):
        state.iteration = iteration
        iteration_begin_time = perf_counter()
        for geo_name in geometry:
            info = geometry[geo_name]
            new_rf = info.rf.at_time(time)
            positions = np.array(info.pos)
            velocities = np.zeros_like(positions)
            rf: ReferenceFrame | None = new_rf
            while rf is not None:
                # Transform positions to parent
                positions = rf.to_parent_with_offset(positions, positions)
                # Add reference frame velocity
                rf.add_velocity(positions, velocities)
                # Transform velocity to the parent
                velocities = rf.to_parent_without_offset(velocities, velocities)
                # Move to the parent
                rf = rf.parent
            # Update the properties in the global reference frame
            state.positions[info.points] = positions
            state.control_points[info.surfaces] = info.msh.surface_average_vec3(positions)
            state.cp_velocity[info.surfaces] = info.msh.surface_average_vec3(velocities)
            state.normals[info.surfaces] = info.msh.surface_normal(positions)

        # Compute flow velocity
        element_velocity = settings.flow_conditions.get_velocity(
            time, state.control_points
        )
        # Add the control point velocities
        element_velocity -= state.cp_velocity
        # Compute flow penetration at control points
        np.vecdot(state.normals, -element_velocity, out=state.circulation, axis=1)  # type: ignore
        # if updated != 0:
        # Compute normal induction
        system_matrix = geometry.mesh.induction_matrix3(
            settings.model_settings.vortex_limit,
            state.positions,
            state.control_points,
            state.normals,
        )

        # Apply the wake model's effect
        if state.wake_model is not None and iteration > 0:
            state.wake_model.apply_corrections(
                state.control_points, state.normals, system_matrix, state.circulation
            )

        # Decompose the system matrix to allow for solving multiple times
        decomp = la.lu_factor(system_matrix, overwrite_a=True)

        # Solve the linear system
        # By setting overwrite_b=True, rhs is where the output is written to
        circulation = la.lu_solve(decomp, state.circulation, overwrite_b=True)
        # Adjust circulations
        for geo_name in geometry:
            info = geometry[geo_name]
            if not info.closed:
                continue
            circulation[info.surfaces] -= np.mean(circulation[info.surfaces])

        # update the wake model
        if state.wake_model is not None:
            state.wake_model.update(
                time, geometry, state.positions, circulation, settings.flow_conditions
            )

        iteration_end_time = perf_counter()
        if (
            settings.time_settings is None
            or (settings.time_settings.output_interval is None)
            or (iteration % settings.time_settings.output_interval == 0)
        ):
            results.circulations[i_out, :] = circulation
            wm = state.wake_model

            if wm is not None:
                results.wake_models.append(type(wm).load(wm.save()))
            else:
                results.wake_models.append(None)

            if output_settings is not None:
                output_settings.serialization_fn(
                    state.save(), output_settings.naming_callback(iteration, time)
                )
            i_out += 1
        print(
            f"Finished iteration {iteration} out of {len(times)} in "
            f"{iteration_end_time - iteration_begin_time:g} seconds."
        )

    return results


def compute_induced_velocities(
    state: SolverState,
    positions: npt.ArrayLike,
) -> npt.NDArray:
    """Compute velocity induced by the mesh with circulation."""
    points = np.asarray(positions, np.float64).reshape((-1, 3))
    ind_mat = state.geometry.mesh.induction_matrix(
        state.settings.model_settings.vortex_limit,
        points,
        state.positions,
    )
    return np.vecdot(ind_mat, state.circulation[None, :, None], axis=1)  # type: ignore
