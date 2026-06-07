"""Implementation of the flow solver."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable, Literal, Self

import numpy as np
import numpy.typing as npt
import scipy.linalg as la

from pyvl._typing import CallableDeserializer, CallableSerializer
from pyvl.cvl import Mesh
from pyvl.fio.io_common import HirearchicalMap, PythonSerializer, SerializationFunction
from pyvl.fio.io_hdf5 import serialize_hdf5
from pyvl.fio.io_json import serialize_json
from pyvl.flow_conditions import FlowConditions
from pyvl.geometry import SimulationGeometry
from pyvl.settings import SolverSettings, WakeShedderCallback, WakeShedderUniform
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
class PyVLComputeMemory:
    """Memory for the solver to use for intermediate calculations to avoid allocations.

    The sizes of buffers we need to provide:
    - Vectors per point (size n_points x 3)
    - Vectors per surface (size n_surfaces x 3)
    - Scalars per surface (size n_surfaces)
    - Vectors per wake point (size n_wake_capacity x 4 x 3)
    """

    vec_p_0: npt.NDArray[np.double]
    vec_p_1: npt.NDArray[np.double]
    vec_p_2: npt.NDArray[np.double]
    vec_e_0: npt.NDArray[np.double]
    vec_e_1: npt.NDArray[np.double]
    vec_e_2: npt.NDArray[np.double]
    vec_e_3: npt.NDArray[np.double]
    vec_e_4: npt.NDArray[np.double]
    scalar_e_0: npt.NDArray[np.double]
    vec_w_0: npt.NDArray[np.double]
    vec_w_1: npt.NDArray[np.double]

    @classmethod
    def from_solver_settings(
        cls, geometry: SimulationGeometry, settings: SolverSettings
    ) -> PyVLComputeMemory:
        """Create cache from solver settings."""
        return cls(
            vec_p_0=np.empty((geometry.n_points, 3), np.double),
            vec_p_1=np.empty((geometry.n_points, 3), np.double),
            vec_p_2=np.empty((geometry.n_points, 3), np.double),
            vec_e_0=np.empty((geometry.n_surfaces, 3), np.double),
            vec_e_1=np.empty((geometry.n_surfaces, 3), np.double),
            vec_e_2=np.empty((geometry.n_surfaces, 3), np.double),
            vec_e_3=np.empty((geometry.n_surfaces, 3), np.double),
            vec_e_4=np.empty((geometry.n_surfaces, 3), np.double),
            scalar_e_0=np.empty((geometry.n_surfaces,), np.double),
            vec_w_0=np.empty(
                (settings.model_settings.wake_settings.wake_element_capacity, 4, 3),
                np.double,
            ),
            vec_w_1=np.empty(
                (settings.model_settings.wake_settings.wake_element_capacity, 4, 3),
                np.double,
            ),
        )


@dataclass(frozen=True)
class SolverState:
    """State of the solver at a specific moment."""

    time: float
    circulation: npt.NDArray[np.double]
    geometry: SimulationGeometry
    settings: SolverSettings
    wake: WakeState

    def save(self, serializer: CallableSerializer) -> HirearchicalMap:
        """Serialize current state to a HirearchicalMap."""
        out = HirearchicalMap()
        out.insert_scalar("time", self.time)
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


def _compute_induced_velocity(
    time: float,
    tol: float,
    mesh: Mesh,
    positions: npt.NDArray[np.double],
    line_circulation: npt.NDArray[np.double],
    wake: WakeState,
    flow_cond: FlowConditions,
    target: npt.NDArray[np.double],
    out: npt.NDArray[np.double] | None = None,
    tmp: npt.NDArray[np.double] | None = None,
) -> npt.NDArray[np.double]:
    """Compute induced velocity at specified locations.

    Parameters
    ----------
    time : float
        Time at which we are computing this.

    tol : float
        Minimum distance before the induced velocity is clamped to zero.

    mesh : Mesh
        Connectivity information of the mesh.

    positions : array
        Positions of the mesh points.

    line_circulations : array
        Circulations of the mesh lines.

    wake : WakeState
        State of the wake.

    flow_cond : FlowConditions
        Flow conditions of the simulation.

    target : array
        Positions where the wake velocity should be computed.

    out : array, optional
        Array that receives the resulting velocity. If not provided,
        a new one is created.

    tmp : array, optional
        Array used for intermediate results. If not provided,
        a new one is created.

    Returns
    -------
    array
        Velocities at target positions. If ``out`` was provided,
        a reference to it is returned, otherwise a new array is created.
    """
    out_shape = target.shape
    # Checking dims is for losers!
    if out is None:
        out = np.empty_like(target).reshape(-1, 3, copy=False)
    if tmp is None:
        tmp = np.empty_like(target).reshape(-1, 3, copy=False)

    target = target.reshape(-1, 3, copy=False)
    # Compute induction of the mesh
    mesh.induction_velocity(
        tol=tol,
        positions=positions,
        control_points=target,
        line_circulation=line_circulation,
        out=out,
    )
    # Compute wake induction
    wake.induced_velocity(
        tol=tol,
        positions=target,
        out_velocity=tmp,
    )
    # Add wake induction to the mesh induction
    np.add(out, tmp, out=out)
    # Compute flow conditions
    flow_cond.get_velocity(time=time, positions=target, out_array=tmp)
    # Add the flow conditions
    np.add(out, tmp, out=out)
    # Return after we reshape back to the right shape
    return out.reshape(out_shape, copy=False)


def update_simulation_state(
    state: SolverState,
    target_time: float,
    out_state: SolverState | None = None,
    compute_memory: PyVLComputeMemory | None = None,
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

    compute_memory : PyVLComputeMemory, optional
        Optional memory object that can be used to avoid memory allocations.

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
        out_circ = np.empty_like(state.circulation)
        out_wake = WakeState.empty(capacity=state.wake.capacity)
    else:
        out_circ = out_state.circulation
        out_wake = out_state.wake

    if compute_memory is None:
        work_velocity_p = np.empty((geometry.n_points, 3), np.double)
        work_position_p = np.empty((geometry.n_points, 3), np.double)
        work_velocity_p1 = np.empty((geometry.n_points, 3), np.double)
        work_norm_e = np.empty((geometry.n_surfaces, 3), np.double)
        work_position_e = np.empty((geometry.n_surfaces, 3), np.double)
        work_velocity_e = np.empty((geometry.n_surfaces, 3), np.double)
        work_velocity_e1 = np.empty((geometry.n_surfaces, 3), np.double)
        work_flux_e1 = np.empty(geometry.n_surfaces, np.double)
        work_wake_1 = np.empty(
            (settings.model_settings.wake_settings.wake_element_capacity, 4, 3),
            np.double,
        )
        work_wake_2 = np.empty(
            (settings.model_settings.wake_settings.wake_element_capacity, 4, 3),
            np.double,
        )

    else:
        work_velocity_p = compute_memory.vec_p_0
        work_position_p = compute_memory.vec_p_1
        work_velocity_p1 = compute_memory.vec_p_2
        work_norm_e = compute_memory.vec_e_0
        work_position_e = compute_memory.vec_e_1
        work_velocity_e = compute_memory.vec_e_2
        work_velocity_e1 = compute_memory.vec_e_3
        work_flux_e1 = compute_memory.scalar_e_0
        work_wake_1 = compute_memory.vec_w_0
        work_wake_2 = compute_memory.vec_w_1

    # Compute the positions and velocities of the geometry at the current time step
    pos, vel = geometry.geometry_at_time(
        target_time, out_pos=work_position_p, out_vel=work_velocity_p
    )

    # Compute the positions, normals, and velocities of the control points
    norm = geometry.mesh.surface_normal(pos, work_norm_e)
    cp_pos = geometry.mesh.surface_average_vec3(pos, work_position_e)
    cp_vel = geometry.mesh.surface_average_vec3(vel, work_velocity_e)

    # Compute the wake model's effect
    wake.induced_normal_velocity(
        tol=tol,
        control_pts=cp_pos,
        normals=norm,
        out_velocity=out_circ,
    )

    # Compute flow velocity
    element_velocity = flow_cond.get_velocity(
        target_time, cp_pos, out_array=work_velocity_e1
    )
    # Compute the normal velocity on each CP due to flow conditions
    # (initializes the array)
    np.vecdot(norm, element_velocity, out=work_flux_e1)

    # Add the induced wake velocity to the flux from flow conditions
    np.add(out_circ, work_flux_e1, out=out_circ)

    # Compute the normal contribution from control point velocities
    np.vecdot(norm, cp_vel, out=work_flux_e1)

    # Required induction is movement flux minus induction + flow
    np.subtract(work_flux_e1, out_circ, out=out_circ)

    # Compute normal induction matrix
    system_matrix = geometry.mesh.induction_matrix3(
        tol=tol,
        positions=pos,
        control_points=cp_pos,
        normals=norm,
        # TODO: add memory for line buffer and output
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
    dt = target_time - state.time
    line_circulations = geometry.mesh.line_circulations(surface_circulations=out_circ)
    if wake.quad_count > 0:
        # Get wake induced velocity
        wake_ind_vel = _compute_induced_velocity(
            time=target_time,
            tol=tol,
            mesh=geometry.mesh,
            positions=pos,
            line_circulation=line_circulations,
            wake=wake,
            flow_cond=flow_cond,
            target=wake.positions,
            out=work_wake_1,
            tmp=work_wake_2,
        )

    else:
        wake_ind_vel = None

    induced_vel: npt.NDArray[np.double] | None = None
    match state.settings.model_settings.wake_settings.wake_shedder:
        case WakeShedderUniform() as uniform_shedder:
            shedding_lines = uniform_shedder.indices

        case WakeShedderCallback() as callback_shedder:
            # Compute the (relative) point velocities
            induced_vel = _compute_induced_velocity(
                time=target_time,
                tol=tol,
                mesh=geometry.mesh,
                positions=pos,
                line_circulation=line_circulations,
                wake=wake,
                flow_cond=flow_cond,
                target=pos,
                out=work_velocity_p1,
            )
            relative_vel = np.subtract(induced_vel, vel, out=vel)

            shedding_lines = np.asarray(
                callback_shedder.shedder(
                    geometry=state.geometry,
                    positions=pos,
                    velocities=relative_vel,
                    time=target_time,
                ),
                np.intp,
            )
            if shedding_lines.ndim != 1:
                raise ValueError(
                    "Shedder callback must return a 1D array of line indices."
                )

        case _:
            raise TypeError("Invalid wake shedder type.")

    if np.any(shedding_lines >= state.geometry.n_lines) or np.any(shedding_lines < 0):
        raise ValueError("Shed line index is out of bounds for the geometry.")

    # Points each line should take its velocity from
    shedding_points = np.array(
        [geometry.mesh.get_line_points(ln) for ln in shedding_lines], np.uint
    )

    shed_pos = pos[shedding_points.reshape(-1, copy=False), :].reshape(
        -1, 2, 3, copy=False
    )

    if induced_vel is None:
        # We do do not have computed point velocities, so we compute it for only required
        induced_vel = _compute_induced_velocity(
            time=target_time,
            tol=tol,
            mesh=geometry.mesh,
            positions=pos,
            line_circulation=line_circulations,
            wake=wake,
            flow_cond=flow_cond,
            target=shed_pos,
        )
    else:
        # Velocity is computed, we just need to select it
        induced_vel = induced_vel[shedding_points.reshape(-1, copy=False), :].reshape(
            -1, 2, 3, copy=False
        )

    # With line velocities, we can now shed elements in that direction from each line
    new_quads = np.empty((shedding_lines.size, 4, 3), np.double)
    # First two points for each quad can already be set based on shedding points
    new_quads[:, 0, :] = pos[shedding_points[:, 0], :]
    new_quads[:, 1, :] = pos[shedding_points[:, 1], :]
    # Remaining two are computed by the velocity at the shedding points
    new_quads[:, 2, :] = new_quads[:, 1, :] + dt * induced_vel[1, :]
    new_quads[:, 3, :] = new_quads[:, 0, :] + dt * induced_vel[0, :]

    # Finally, update the wake if we can
    if wake_ind_vel is not None:
        # update the wake model
        wake = wake.update_wake(dt=dt, velocities=wake_ind_vel, out_state=out_wake)

    # Add wake quads
    wake.add_quads(
        new_positions=new_quads,
        new_circulations=line_circulations,
        out_state=out_wake,
    )

    if out_state is None:
        return SolverState(
            time=target_time,
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
    cache = PyVLComputeMemory.from_solver_settings(geometry, settings)

    for iteration, time in enumerate(times):
        iteration_begin_time = perf_counter()
        update_simulation_state(state, time, out_state=out_state, compute_memory=cache)
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
