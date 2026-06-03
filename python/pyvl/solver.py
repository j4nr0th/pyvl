"""Implementation of the flow solver."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable, Literal, Self

import numpy as np
import numpy.typing as npt
import scipy.linalg as la

from pyvl._typing import CallableDeserializer, CallableSerializer
from pyvl.fio.io_common import HirearchicalMap, PythonSerializer, SerializationFunction
from pyvl.fio.io_hdf5 import serialize_hdf5
from pyvl.fio.io_json import serialize_json
from pyvl.geometry import SimulationGeometry
from pyvl.settings import SolverSettings
from pyvl.wake import WakeState


class SolverResults:
    """Class containing results of a solver."""

    geometry: SimulationGeometry
    circulations: npt.NDArray[np.double]
    wake_states: list[WakeState]
    settings: SolverSettings

    def __init__(self, geo: SimulationGeometry, settings: SolverSettings):
        self.geometry = geo
        self.settings = SolverSettings(
            flow_conditions=settings.flow_conditions,
            model_settings=settings.model_settings,
            time_settings=settings.time_settings,
        )
        self.wake_states = list()
        self.circulations = np.empty(
            (settings.time_settings.output_times.size, geo.n_surfaces), np.double
        )


@dataclass(frozen=True)
class SolverState:
    """State of the solver at a specific moment."""

    time: float
    cp_velocity: npt.NDArray[np.double]
    circulation: npt.NDArray[np.double]
    geometry: SimulationGeometry
    settings: SolverSettings
    wake: WakeState

    def save(self, serializer: CallableSerializer) -> HirearchicalMap:
        """Serialize current state to a HirearchicalMap."""
        out = HirearchicalMap()
        out.insert_scalar("time", self.time)
        out.insert_array("cp_velocity", self.cp_velocity)
        out.insert_array("circulation", self.circulation)
        out.insert_hirearchical_map("solver_settings", self.settings.save(serializer))
        out.insert_hirearchical_map("simulation_geometry", self.geometry.save(serializer))
        out.insert_hirearchical_map("wake", self.wake.save())
        return out

    @classmethod
    def load(
        cls,
        hmap: HirearchicalMap,
        deserializer: CallableDeserializer,
        custom_types: Mapping[str, type] | None = None,
        allow_override: bool = False,
    ) -> Self:
        """Deserialize current state from a HirearchicalMap.

        Parameters
        ----------
        hmap : HirearchicalMap
            Serialized state of the :class:`SolverState` object.
        custom_types : Mapping[str, type], optional
            A mapping of type names to types for custom subclasses of FlowConditions
            and WakeModel.
        allow_override : bool, default: False
            If True, custom types can override built-in types.
        """
        return cls(
            time=hmap.get_scalar("time"),
            geometry=SimulationGeometry.load(
                hmap.get_hirearchical_map("simulation_geometry"), deserializer
            ),
            settings=SolverSettings.load(
                hmap.get_hirearchical_map("solver_settings"),
                deserializer,
                custom_types,
                allow_override,
            ),
            wake=WakeState.load(hmap.get_hirearchical_map("wake")),
            cp_velocity=hmap.get_array("cp_velocity"),
            circulation=hmap.get_array("circulation"),
        )

    @classmethod
    def create_new(
        cls,
        time: float,
        geometry: SimulationGeometry,
        settings: SolverSettings,
    ) -> Self:
        """Create a new :class:`SolverState` object with uninitialized state."""
        return cls(
            time=time,
            geometry=geometry,
            settings=settings,
            wake=WakeState.empty(
                settings.model_settings.wake_settings.wake_element_capacity
            ),
            cp_velocity=np.empty((geometry.n_surfaces, 3), np.double),
            circulation=np.empty(geometry.n_surfaces, np.double),
        )


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
    callable_serializer: CallableSerializer
    callable_deserializer: CallableDeserializer

    def __init__(
        self,
        ftype: OutputFileType,
        naming_callback: Callable[[int, float], str | Path],
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
        callable_serialization = PythonSerializer()
        object.__setattr__(self, "callable_serializer", callable_serialization.serialize)
        object.__setattr__(
            self, "callable_deserializer", callable_serialization.deserialize
        )


def update_simulation_state(
    state: SolverState,
    target_time: float,
    out_state: SolverState | None = None,
) -> SolverState:
    """Update the simulation state by applying the wake model's update method.

    Parameters
    ----------
    state : SolverState
        The current state of the solver.

    target_time : float
        The target simulation time.

    out_state : SolverState, optional
        The state to write the updated values to. If not provided,
        the input state will be updated in-place.

    Returns
    -------
    SolverState
        Updated solver state. If the output state is provided,
        this will be the reference to the same object, otherwise a new
        state object will be returned.
    """
    # Check that we are not going back in time
    if target_time < state.time:
        raise ValueError(
            f"Target time {target_time} is less than current time {state.time}."
        )

    geometry = state.geometry
    settings = state.settings
    tol = settings.model_settings.vortex_limit
    flow_cond = settings.flow_conditions
    wake = state.wake

    # Ensure output state
    if out_state is None:
        out_cp_vel = np.empty_like(state.cp_velocity)
        out_circ = np.empty_like(state.circulation)
        out_wake = WakeState.empty(capacity=state.wake.capacity)
    else:
        out_cp_vel = out_state.cp_velocity
        out_circ = out_state.circulation
        out_wake = out_state.wake

    # Compute the positions and velocities of the geometry at the current time step
    pos, vel = geometry.geometry_at_time(target_time)

    # Compute the positions, normals, and velocities of the control points
    norm = geometry.mesh.surface_normal(pos)
    cp_pos = geometry.mesh.surface_average_vec3(pos)
    cp_vel = geometry.mesh.surface_average_vec3(vel)

    # Compute the wake model's effect
    wake.induced_normal_velocity(
        tol=tol,
        control_pts=cp_pos,
        normals=norm,
        out_velocity=out_circ,
    )

    # Compute flow velocity
    element_velocity = flow_cond.get_velocity(target_time, cp_pos)
    # Add the control point velocities
    element_velocity -= cp_vel
    # Compute flow penetration at control points
    out_circ[:] -= np.sum(norm * element_velocity, axis=1)
    # Compute normal induction
    system_matrix = geometry.mesh.induction_matrix3(
        tol=tol,
        positions=pos,
        control_points=cp_pos,
        normals=norm,
    )

    # Decompose the system matrix to allow for solving multiple times
    decomp = la.lu_factor(system_matrix, overwrite_a=True)

    # Solve the linear system
    # By setting overwrite_b=True, rhs is where the output is written to
    out_circ[:] = np.asarray(la.lu_solve(decomp, out_circ, overwrite_b=True), np.double)
    # Adjust circulations of closed surfaces to have zero mean circulation
    for geo_name in geometry:
        info = geometry[geo_name]
        if not info.closed:
            continue
        out_circ[info.surfaces] -= np.mean(out_circ[info.surfaces])

    # Compute the velocities of wake elements at this time step
    wake_pos = wake.positions
    if wake.quad_count == 0:
        wake_mesh_induction = np.zeros((0, 4, 3), dtype=np.double)
    else:
        # Reshape wake positions to (M*4, 3) to use induction_matrix
        wake_pos_flat = wake_pos.reshape(-1, 3)
        wake_ind_mat_flat = geometry.mesh.induction_matrix(
            tol=tol,
            positions=pos,
            control_points=wake_pos_flat,
        )
        # Reshape back to (M, 4, n_surfaces, 3)
        wake_ind_mat = wake_ind_mat_flat.reshape(wake.quad_count, 4, -1, 3)
        wake_mesh_induction = np.sum(wake_ind_mat * out_circ[None, None, :, None], axis=2)

    wake_self_induction = wake.induced_velocity(
        tol=tol, positions=wake_pos[: wake.quad_count]
    )
    wake_freestream = flow_cond.get_velocity(target_time, wake_pos[: wake.quad_count])

    # update the wake model
    wake.update_wake(
        dt=target_time - state.time,
        velocities=wake_freestream + wake_mesh_induction + wake_self_induction,
        out_state=out_wake,
    )

    # TODO: add new wake elements based on shedding

    if out_state is None:
        return SolverState(
            time=target_time,
            cp_velocity=out_cp_vel,
            circulation=out_circ,
            geometry=state.geometry,
            settings=state.settings,
            wake=out_wake,
        )

    return out_state


def run_solver(
    geometry: SimulationGeometry,
    settings: SolverSettings,
    output_settings: OutputSettings | None,
) -> SolverResults:
    """Run the flow solver to obtain specified circulations.

    Parameters
    ----------
    geometry : SimulationGeometry
        Geometry to solver for.

    settings : SolverSettings
        Settings of the solver.

    output_settings : OutputSettings, optional
        Settings related to file IO.
    """
    results = SolverResults(geometry, settings)
    times: npt.NDArray[np.double]
    if settings.time_settings is None:
        times = np.array((0,), np.double)
    else:
        times = np.astype(
            np.arange(settings.time_settings.nt) * settings.time_settings.dt, np.double
        )

    i_out = 0

    # Main state
    state = SolverState.create_new(times[0], geometry, settings)
    # Output state to avoid unnecessary allocations during the loop
    out_state = SolverState.create_new(times[0], geometry, settings)

    for iteration, time in enumerate(times):
        iteration_begin_time = perf_counter()
        update_simulation_state(state, time, out_state=out_state)
        iteration_end_time = perf_counter()
        # Swap the states for the next iteration
        state, out_state = out_state, state

        if (
            settings.time_settings is None
            or (settings.time_settings.output_interval is None)
            or (iteration % settings.time_settings.output_interval == 0)
        ):
            results.circulations[i_out, :] = state.circulation
            results.wake_states.append(state.wake)
            if output_settings is not None:
                output_settings.serialization_fn(
                    state.save(output_settings.callable_serializer),
                    output_settings.naming_callback(iteration, time),
                )
            i_out += 1
        print(
            f"Finished iteration {iteration} out of {len(times)} in "
            f"{iteration_end_time - iteration_begin_time:g} seconds."
        )

    return results
