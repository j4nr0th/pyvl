"""Implementation of the flow solver."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Literal, Protocol, Self, overload

import numpy as np
import numpy.typing as npt
import scipy.linalg as la

from pyvl._typing import CallableDeserializer, CallableSerializer
from pyvl.cvl import Mesh
from pyvl.fio.io_common import HirearchicalMap, PythonSerializer, SerializationFunction
from pyvl.fio.io_hdf5 import serialize_hdf5
from pyvl.fio.io_json import serialize_json
from pyvl.flow_conditions import FlowConditions
from pyvl.geometry import Geometry, SimulationGeometry
from pyvl.settings import SolverSettings, WakeShedderCallback, WakeShedderUniform
from pyvl.wake import WakeState


class SolverSystem:
    """Type used to hold solver state."""

    # Mapping of geometry objects for easy access.
    _geometry: dict[str, Geometry]
    # Normal induction matrices, which are the LU decomposition terms used for
    # Gaussian elimination.
    _normal_induction_matrices: dict[tuple[str, str], npt.NDArray[np.double]]
    # Order of geometries in the inverse based on their labels. Essentially serves
    # as the permutation matrix.
    _part_order: list[str]
    # Self-induction blocks, which are on the diagonal. As long as part does not deform,
    # these do not need to ever be recomputed
    _self_induction_diags: dict[str, npt.NDArray[np.double]]
    # Decompositions for the diagonal terms
    _diag_decomposes: dict[str, Any]
    # Time at which we have the state
    time: float
    # Distance below which the induction of a horse shoe vortex is set to zero.
    vortex_tol: float

    def _compute_induction_matrix(
        self, source: str, target: str, t: float, tol: float
    ) -> npt.NDArray[np.double]:
        """Compute the induction matrix of the source on the target.

        Parameters
        ----------
        source : str
            Label of the source geometry.

        target : str
            Label of the target geometry.

        t : float
            Time at which to compute the induction matrices.

        tol : float
            Tolerance used when computing the induction matrices.

        Returns
        -------
        array
            Induction matrix that was just computed.
        """
        source_geo = self._geometry[source]
        target_geo = self._geometry[target]

        # TODO: cache these in part_positions and part_cpts
        source_pos = source_geo.reference_frame.to_global_position(
            source_geo.positions, time=t
        )
        target_cpts = target_geo.reference_frame.to_global_position(
            target_geo.centers, time=t
        )
        target_normals = target_geo.reference_frame.to_global_vector(
            target_geo.normals, time=t
        )
        ind_mat = source_geo.msh.induction_matrix3(
            tol=tol,
            positions=source_pos,
            control_points=target_cpts,
            normals=target_normals,
            out=self._normal_induction_matrices[(source, target)],
        )

        return ind_mat

    def update(self, t_new: float) -> None:
        """Check if induction matrices must be updated and compute them if needed.

        Parameters
        ----------
        t_end : float
            New time at which we need the matrices

        tol : float
            Tolerance used when computing the induction matrices.
        """
        if self.time == t_new:
            # Done, no need to do anything :)
            return

        if len(self._geometry) == 1:
            # A single part never moves with respect to itself.
            return

        # Groups together parts which do not move relative to each other
        static_groups: list[list[str]] = list()

        for i, part_1_name in enumerate(self._part_order):
            part_1 = self._geometry[part_1_name]
            for part_2_name in self._part_order[i + 1 :]:
                part_2 = self._geometry[part_2_name]
                # Check movement
                moved = part_1.reference_frame.moved_relative_to(
                    part_2.reference_frame,
                    t_start=self.time,
                    t_end=t_new,
                    tol=self.vortex_tol,
                )
                if not moved:
                    # The reference frames did not move relative to one another.

                    g1 = [g for g in static_groups if part_1_name in g]
                    g2 = [g for g in static_groups if part_2_name in g]
                    if not len(g1):
                        g1 = None
                    else:
                        assert len(g1) == 1
                        g1 = g1[0]
                    if not len(g2):
                        g2 = None
                    else:
                        assert len(g2) == 1
                        g2 = g2[0]

                    if g1 is None:
                        if g2 is None:
                            static_groups.append([part_1_name, part_2_name])
                        else:
                            g2.append(part_1_name)
                    else:
                        if g2 is None:
                            g1.append(part_2_name)
                        else:
                            static_groups.remove(g2)
                            g1.extend(g2)
                    continue

                # They did move, recompute the induction matrices
                self._compute_induction_matrix(
                    source=part_1_name, target=part_2_name, t=t_new, tol=self.vortex_tol
                )
                self._compute_induction_matrix(
                    source=part_2_name, target=part_1_name, t=t_new, tol=self.vortex_tol
                )

        new_order = self._part_order
        preserved = 0
        if len(static_groups) != 0:
            # We had some parts that did not move relative to one another
            # Sort each of the groups based on the number of elements
            static_groups = [
                sorted(g, key=lambda g: self._geometry[g].msh.n_surfaces, reverse=True)
                for g in static_groups
            ]
            # Sort the static groups based on the total element count
            static_groups = sorted(
                static_groups,
                key=lambda g: sum([self._geometry[n].msh.n_surfaces for n in g]),
                reverse=True,
            )
            preserved = len(static_groups[0])
            # Create a new order, while preserving existing relative order within groups
            new_order: list[str] = list()
            for group in static_groups:
                new_order.extend(sorted(group, key=lambda s: self._part_order.index(s)))

            # Add the parts that were moving with respect to all others
            new_order.extend(
                sorted(
                    [name for name in self._part_order if name not in new_order],
                    key=lambda g: self._geometry[g].msh.n_surfaces,
                    reverse=True,
                )
            )

        # Update the current inverse
        self._update_inverse(new_order, max_preserved=preserved)
        # Update the time
        self.time = t_new

    def _update_inverse(self, new_order: list[str], max_preserved: int) -> None:
        """Update the current system inverse based on the new order.

        Parameters
        ----------
        new_order : list of str
            The new order of geometries.

        max_preserved : int
            Number of the rows which we do not need to recompute if their order stays
            the same.
        """
        # Check how many we still have from the current state (if allowed)
        start = 0
        n = len(new_order)
        assert set(new_order) == set(self._part_order)
        assert 0 <= max_preserved <= n

        # We can reuse the previously computed parts
        for old, new in zip(self._part_order, new_order, strict=True):
            if old != new or start >= max_preserved:
                break
            start += 1

        # Perform (unpivoted LU) block by block
        for i in range(start, n):
            part = new_order[i]
            # Copy the self-induction matrix
            self._normal_induction_matrices[(part, part)] = self._self_induction_diags[
                part
            ].copy()

            # Now clear up the row via Gaussian elimination
            for j in range(0, i):
                other = new_order[j]
                # A_{i,j} = induction of part_j on part_i
                self._normal_induction_matrices[(other, part)][:] = la.lu_solve(
                    self._diag_decomposes[other],
                    self._normal_induction_matrices[(other, part)],
                    overwrite_b=True,
                )
                # Apply this to the other entries of the row (part, ...) after the
                # element (part, other)
                for k in range(j + 1, n):
                    t = new_order[k]
                    np.subtract(
                        self._normal_induction_matrices[(t, part)],
                        self._normal_induction_matrices[(other, part)]
                        @ self._normal_induction_matrices[(t, other)],
                        out=self._normal_induction_matrices[(t, part)],
                    )

            # Every entry was eliminated, so we can add the diagonal decomposition now
            self._diag_decomposes[part] = la.lu_factor(
                self._normal_induction_matrices[(part, part)]
            )

        # Done with the Gaussian elimination, now we can update the order
        self._part_order = new_order

    def __init__(self, time: float, tol: float, geo: Iterable[Geometry]) -> None:
        self._diag_decomposes = dict()
        geo = tuple(geo)
        sizes = np.array([g.msh.n_surfaces for g in geo])
        order = np.argsort(-sizes)
        # Sort by number of elements
        self._part_order = [geo[i].label for i in order]
        self._geometry = {g.label: g for g in geo}

        # Compute the self-induction decompositions for all the elements
        self._self_induction_diags = {
            g.label: g.msh.induction_matrix3(
                tol=tol,
                positions=g.positions,
                control_points=g.centers,
                normals=g.normals,
            )
            for g in geo
        }

        # Allocate the memory and compute the induction matrices
        self._normal_induction_matrices = dict()
        for g1 in geo:
            for g2 in geo:
                mat = np.empty((g2.msh.n_surfaces, g1.msh.n_surfaces), np.double)
                self._normal_induction_matrices[(g1.label, g2.label)] = mat
                self._compute_induction_matrix(g1.label, g2.label, t=time, tol=tol)

        # Now we can compute the whole inverse
        self._update_inverse(new_order=self._part_order, max_preserved=0)
        self.time = time
        self.vortex_tol = tol

    def solve_inverse(self, x: dict[str, npt.NDArray[np.double]]) -> None:
        """Solve compute the system inverse with the current state.

        Uses the fully pivoted LU decomposition that is computed before to
        solve the system.

        Parameters
        ----------
        x : dict of str -> array
            Mapping of normal flux that should be induced for each element for each
            geometry. The solution is written back to these same vectors.
        """
        # First, check the vectors have the right sizes
        if set(x.keys()) != set(self._part_order):
            raise ValueError("Parts in the system and the input vector are not the same.")

        for part_name in x:
            if x[part_name].shape != (self._geometry[part_name].msh.n_surfaces,):
                raise ValueError(
                    f'The vector for geometry"{part_name}" had the shape '
                    f"{x[part_name].shape}, which should instead be a vector with "
                    f"{self._geometry[part_name].msh.n_surfaces} elements."
                )

        # Perform the elimination step with the lower half of the matrix
        n = len(self._part_order)
        # We need to solve A * u = x
        # With the block LU decomposition, the lower matrix is:
        # 1  0 ...
        # L_21 1 ...
        # L_31 L_32 1 ...
        # where L_ij is what we stored in self.normal_induction_matrices[(i, j)]
        # after elimination.
        # Actually, in update_inverse:
        # 1. A_pt = A_pt - A_po * A_ot
        # Where A_po = A_po * inv(A_oo)

        # Step 1: Forward substitution with lower triangular matrix
        # L_{i,j} is stored in normal_induction_matrices[(part_j, part_i)]
        for i in range(1, n):
            row_name = self._part_order[i]
            target_vec = x[row_name]
            for j in range(0, i):
                col_name = self._part_order[j]
                np.subtract(
                    target_vec,
                    self._normal_induction_matrices[(col_name, row_name)] @ x[col_name],
                    out=target_vec,
                )

        # Step 2: Backward substitution with upper triangular matrix
        # U_{i,j} for j > i is the original A_{i,j} (storage is (part_j, part_i))
        # (unchanged by elimination since only A_{i,j} for j < i are overwritten)
        for i in reversed(range(0, n)):
            row_name = self._part_order[i]
            target_vec = x[row_name]
            for j in range(i + 1, n):
                col_name = self._part_order[j]
                np.subtract(
                    target_vec,
                    self._normal_induction_matrices[(col_name, row_name)] @ x[col_name],
                    out=target_vec,
                )
            # Now apply the diagonal inverse A_ii^-1 * target_vec
            target_vec[:] = la.lu_solve(
                self._diag_decomposes[row_name], target_vec, overwrite_b=True
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
    scalar_e_4: npt.NDArray[np.double]
    scalar_e_0: npt.NDArray[np.double]
    vec_w_0: npt.NDArray[np.double]
    vec_w_1: npt.NDArray[np.double]

    # Specific memory buffers
    line_buffer: npt.NDArray[np.double]
    mat_buffer: npt.NDArray[np.double]

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
            scalar_e_4=np.empty((geometry.n_surfaces,), np.double),
            scalar_e_0=np.empty((geometry.n_surfaces,), np.double),
            vec_w_0=np.empty(
                (settings.wake_settings.wake_element_capacity, 4, 3), np.double
            ),
            vec_w_1=np.empty(
                (settings.wake_settings.wake_element_capacity, 4, 3), np.double
            ),
            line_buffer=np.empty((geometry.n_lines, geometry.n_surfaces, 3)),
            mat_buffer=np.empty((geometry.n_surfaces, geometry.n_surfaces)),
        )


@dataclass(frozen=True)
class SolverState:
    """State of the solver at a specific moment."""

    time: float
    circulation: npt.NDArray[np.double]
    geometry: SimulationGeometry
    settings: SolverSettings
    wake: WakeState
    shed_lines: npt.NDArray[np.intp]

    def save(self, serializer: CallableSerializer) -> HirearchicalMap:
        """Serialize current state to a HirearchicalMap."""
        out = HirearchicalMap()
        out.insert_scalar("time", self.time)
        out.insert_array("circulation", self.circulation)
        out.insert_hirearchical_map("solver_settings", self.settings.save(serializer))
        out.insert_hirearchical_map("simulation_geometry", self.geometry.save(serializer))
        out.insert_hirearchical_map("wake", self.wake.save())
        out.insert_array("shed_lines", self.shed_lines)
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
            shed_lines=hmap.get_array("shed_lines"),
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
            wake=WakeState.empty(settings.wake_settings.wake_element_capacity),
            circulation=np.zeros(
                sum([geometry.geometries[geo].msh.n_lines for geo in geometry.geometries])
            ),
            shed_lines=np.array((), np.intp),
        )

    def compute_velocity(
        self,
        positions: npt.ArrayLike,
        induced_only: bool = False,
        out: npt.NDArray[np.double] | None = None,
        n_threads: int = 1,
    ):
        """Compute velocity for this solver state."""
        return _compute_induced_velocity(
            time=self.time,
            tol=self.settings.model_settings.vortex_limit,
            line_circulation=self.circulation,
            mesh=self.geometry.mesh_joined,
            positions=self.geometry.positions_at_time(self.time),
            wake=self.wake,
            flow_cond=None if induced_only else self.settings.flow_conditions,
            target=np.asarray(positions, np.double),
            out=out,
            n_threads=n_threads,
        )


OutputFileType = Literal["HDF5", "JSON"]


class SavePredicate(Protocol):
    """Predicate used to determine if the solver state should be saved at step."""

    def __call__(self, step: int, time: float, state: SolverState) -> bool:
        """Determine if the solver state should be saved at step.

        Parameters
        ----------
        step : int
            Current step of the solver.

        time : float
            Current time of the solver.

        state : SolverState
            Current state of the solver.

        Returns
        -------
        bool
            True if the solver state should be saved, False otherwise.
        """
        ...


@dataclass(eq=False, frozen=True)
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
    output_predicate: SavePredicate | None = None

    @classmethod
    def simple_python(
        cls,
        ftype: OutputFileType,
        naming_callback: Callable[[int, float], str | Path],
        output_predicate: SavePredicate | None = None,
    ) -> Self:
        """Create a simple output settings object that uses Python serialization.

        Such serializer will not be able to load callables which it did not serialize
        itself, so it cannot be used for custom callbacks.

        Parameters
        ----------
        ftype : "JSON" or "HDF5"
            File format to write the output as.

        naming_callback : (int, float) -> str | Path
            Callback to use to determine the name of the next file
            to write based on the iteration number and the simulation time.

        output_predicate : (int, float) -> bool, optional
            Predicate to determine if the solver state should be saved at step.
            If not provided, all steps will be saved.

        Returns
        -------
        OutputSettings
            OutputSettings, which use Python serialization and the specified file type.
        """
        serialization_fn: SerializationFunction
        match ftype:
            case "HDF5":
                serialization_fn = serialize_hdf5
            case "JSON":
                serialization_fn = serialize_json
            case _:
                raise ValueError(f"The file type {ftype=} is not valid.")
        callable_serialization = PythonSerializer()
        return cls(
            serialization_fn=serialization_fn,
            naming_callback=naming_callback,
            output_predicate=output_predicate,
            callable_serializer=callable_serialization.serialize,
            callable_deserializer=callable_serialization.deserialize,
        )


def _save_output_if_needed(
    settings: OutputSettings, state: SolverState, step: int
) -> None:
    """Save the solver state if the output predicate is satisfied.

    Parameters
    ----------
    settings : OutputSettings
        Settings to control the output from a solver.

    state : SolverState
        Current state of the solver.

    step : int
        Current step of the solver.
    """
    if settings.output_predicate is not None and not settings.output_predicate(
        step, state.time, state
    ):
        return

    # Save the output
    filename = settings.naming_callback(step, state.time)
    hmap = state.save(settings.callable_serializer)
    settings.serialization_fn(hmap, filename)


def _compute_induced_velocity(
    time: float,
    tol: float,
    mesh: Mesh,
    positions: npt.NDArray[np.double],
    line_circulation: npt.NDArray[np.double],
    wake: WakeState,
    flow_cond: FlowConditions | None,
    target: npt.NDArray[np.double],
    out: npt.NDArray[np.double] | None = None,
    tmp: npt.NDArray[np.double] | None = None,
    n_threads: int = 1,
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

    flow_cond : FlowConditions or None
        Flow conditions of the simulation. If set to ``None``,
        no contribution is added.

    target : array
        Positions where the wake velocity should be computed.

    out : array, optional
        Array that receives the resulting velocity. If not provided,
        a new one is created.

    tmp : array, optional
        Array used for intermediate results. If not provided,
        a new one is created.

    n_threads : int, default: 1
        Number of threads to use for computing the induction.

    Returns
    -------
    array
        Velocities at target positions. If ``out`` was provided,
        a reference to it is returned, otherwise a new array is created.
    """
    out_shape = target.shape
    # Checking dims is for losers!
    if out is None:
        out = np.empty_like(target)
    out = out.reshape(-1, 3, copy=False)

    if tmp is None:
        tmp = np.empty_like(target)
    tmp = tmp.reshape(-1, 3, copy=False)

    target = target.reshape(-1, 3, copy=False)
    # Compute induction of the mesh
    mesh.induction_velocity(
        tol=tol,
        positions=positions,
        control_points=target,
        line_circulation=line_circulation,
        out=out,
        n_threads=n_threads,
    )
    # Compute wake induction
    wake.induced_velocity(
        tol=tol, positions=target, out_velocity=tmp, n_threads=n_threads
    )
    # Add wake induction to the mesh induction
    np.add(out, tmp, out=out)
    if flow_cond is not None:
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
    system: SolverSystem | None = None,
    n_threads: int = 1,
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

    system : SolverSystem, optional
        System solver used for calculating the solution. Once initialized, it
        can really speed up calculations, especially for cases with static geometry.

    n_threads : int, default: 1
        Number of threads to use.

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
        work_flux_e2 = np.empty(geometry.n_surfaces, np.double)
        work_flux_e1 = np.empty(geometry.n_surfaces, np.double)
        work_wake_1 = np.empty(
            (settings.wake_settings.wake_element_capacity, 4, 3),
            np.double,
        )
        work_wake_2 = np.empty(
            (settings.wake_settings.wake_element_capacity, 4, 3),
            np.double,
        )
        line_buffer = np.empty((geometry.n_lines, geometry.n_surfaces, 3))
        mat_buffer = np.empty((geometry.n_surfaces, geometry.n_surfaces))

    else:
        work_velocity_p = compute_memory.vec_p_0
        work_position_p = compute_memory.vec_p_1
        work_velocity_p1 = compute_memory.vec_p_2
        work_norm_e = compute_memory.vec_e_0
        work_position_e = compute_memory.vec_e_1
        work_velocity_e = compute_memory.vec_e_2
        work_velocity_e1 = compute_memory.vec_e_3
        work_flux_e1 = compute_memory.scalar_e_0
        work_flux_e2 = compute_memory.scalar_e_4
        work_wake_1 = compute_memory.vec_w_0
        work_wake_2 = compute_memory.vec_w_1
        line_buffer = compute_memory.line_buffer
        mat_buffer = compute_memory.mat_buffer

    # Compute the positions and velocities of the geometry at the current time step
    pos, vel = geometry.geometry_at_time(
        target_time, out_pos=work_position_p, out_vel=work_velocity_p
    )

    # Compute the positions, normals, and velocities of the control points
    norm = geometry.mesh_joined.surface_normal(pos, work_norm_e)
    cp_pos = geometry.mesh_joined.surface_average_vec3(pos, work_position_e)
    cp_vel = geometry.mesh_joined.surface_average_vec3(vel, work_velocity_e)

    # Compute the wake model's effect
    cp_flux = wake.induced_normal_velocity(
        tol=tol,
        control_pts=cp_pos,
        normals=norm,
        out_velocity=work_flux_e2,
        n_threads=n_threads,
    )

    # Compute flow velocity
    element_velocity = flow_cond.get_velocity(
        target_time, cp_pos, out_array=work_velocity_e1
    )
    # Compute the normal velocity on each CP due to flow conditions
    # (initializes the array)
    np.vecdot(norm, element_velocity, out=work_flux_e1)

    # Add the induced wake velocity to the flux from flow conditions
    np.add(cp_flux, work_flux_e1, out=cp_flux)

    # Compute the normal contribution from control point velocities
    np.vecdot(norm, cp_vel, out=work_flux_e1)

    # Required induction is movement flux minus induction + flow
    np.subtract(work_flux_e1, cp_flux, out=cp_flux)

    if system is None:
        # Compute normal induction matrix
        system_matrix = geometry.mesh_joined.induction_matrix3(
            tol=tol,
            positions=pos,
            control_points=cp_pos,
            normals=norm,
            out=mat_buffer,
            line_buffer=line_buffer,
            thread_count=n_threads,
        )

        # Decompose the system matrix to allow for solving multiple times
        decomp = la.lu_factor(system_matrix, overwrite_a=True)

        # Solve the linear system
        # By setting overwrite_b=True, rhs is where the output is written to
        surf_circ = np.asarray(la.lu_solve(decomp, cp_flux, overwrite_b=True), np.double)
        geometry.mesh_joined.line_circulations(
            circulation=surf_circ, out=out_circ, n_threads=n_threads
        )

    else:
        # Reuse as much as possible here
        system.update(t_new=target_time)
        part_circ = {
            part_name: cp_flux[geometry[part_name].surfaces] for part_name in geometry
        }
        system.solve_inverse(part_circ)
        geometry.mesh_joined.line_circulations(
            circulation=cp_flux, out=out_circ, n_threads=n_threads
        )

    # Compute the velocities of wake elements at this time step
    dt = target_time - state.time
    if wake.quad_count > 0:
        # Get wake induced velocity
        wake_ind_vel = _compute_induced_velocity(
            time=target_time,
            tol=tol,
            mesh=geometry.mesh_joined,
            positions=pos,
            line_circulation=out_circ,
            wake=wake,
            flow_cond=flow_cond,
            target=wake.positions,
            out=work_wake_1[: wake.quad_count],
            tmp=work_wake_2[: wake.quad_count],
            n_threads=n_threads,
        )

    else:
        wake_ind_vel = None

    induced_vel: npt.NDArray[np.double] | None = None
    match state.settings.wake_settings.wake_shedder:
        case WakeShedderUniform() as uniform_shedder:
            shedding_lines = uniform_shedder.indices

        case WakeShedderCallback() as callback_shedder:
            # Compute the (relative) point velocities
            induced_vel = _compute_induced_velocity(
                time=target_time,
                tol=tol,
                mesh=geometry.mesh_joined,
                positions=pos,
                line_circulation=out_circ,
                wake=wake,
                flow_cond=flow_cond,
                target=pos,
                out=work_velocity_p1,
                n_threads=n_threads,
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

        case None:
            shedding_lines = np.array([])

        case _:
            raise TypeError("Invalid wake shedder type.")

    if np.any(shedding_lines >= state.geometry.n_lines) or np.any(shedding_lines < 0):
        raise ValueError("Shed line index is out of bounds for the geometry.")

    if len(shedding_lines) and dt != 0:
        # Points each line should take its velocity from
        shedding_points = np.array(
            [geometry.mesh_joined.get_line_points(ln) for ln in shedding_lines], np.uint
        )

        shed_pos = pos[shedding_points.reshape(-1, copy=False), :].reshape(
            -1, 2, 3, copy=False
        )
        new_quads = np.empty((shedding_lines.size, 4, 3), np.double)

        if induced_vel is None:
            # We do do not have computed point velocities, so compute it for only required
            induced_vel = _compute_induced_velocity(
                time=target_time,
                tol=tol,
                mesh=geometry.mesh_joined,
                positions=pos,
                line_circulation=out_circ,
                wake=wake,
                flow_cond=flow_cond,
                target=shed_pos,
                n_threads=n_threads,
            )
        else:
            # Velocity is computed, we just need to select it
            induced_vel = induced_vel[shedding_points.reshape(-1, copy=False), :].reshape(
                -1, 2, 3, copy=False
            )

        # With line velocities, we can now shed elements in that direction from each line
        # First two points for each quad can already be set based on shedding points
        new_quads[:, 0, :] = pos[shedding_points[:, 0], :]
        new_quads[:, 1, :] = pos[shedding_points[:, 1], :]
        # Remaining two are computed by the velocity at the shedding points
        new_quads[:, 2, :] = new_quads[:, 1, :] + dt * induced_vel[:, 1, :]
        new_quads[:, 3, :] = new_quads[:, 0, :] + dt * induced_vel[:, 0, :]

        # Finally, update the wake if we can
        if wake_ind_vel is not None:
            # update the wake model
            out_wake = wake = wake.update_wake(
                dt=dt, velocities=wake_ind_vel, out_state=out_wake
            )

        # Add wake quads
        out_wake = wake.add_quads(
            new_positions=new_quads,
            # Circulation must have an opposite sign!
            new_circulations=-out_circ[shedding_lines],
            out_state=out_wake,
        )

    elif wake_ind_vel is not None and dt != 0:
        # update the wake model
        out_wake = wake.update_wake(dt=dt, velocities=wake_ind_vel, out_state=out_wake)

    return SolverState(
        time=target_time,
        circulation=out_circ,
        geometry=state.geometry,
        settings=state.settings,
        wake=out_wake,
        shed_lines=np.asarray(shedding_lines, np.intp),
    )


class TimeStepFunction(Protocol):
    """Function used to determine the time step size at each step."""

    def __call__(self, step: int, time: float, state: SolverState) -> float | None:
        """Determine the time step size at each step.

        Parameters
        ----------
        step : int
            Current step of the solver.

        time : float
            Current time of the solver.

        state : SolverState
            Current state of the solver.

        Returns
        -------
        float or None
            Time step size to use for the next step. If None is returned,
            the simulation is finished and the solver will stop.
        """
        ...


@overload
def run_solver(
    geometry: SimulationGeometry,
    settings: SolverSettings,
    times: Iterable[float] | TimeStepFunction,
    initial_time: float,
    save_predicate: SavePredicate,
    output_settings: OutputSettings | None = None,
    n_threads: int = 1,
) -> tuple[SolverState, ...]: ...
@overload
def run_solver(
    geometry: SimulationGeometry,
    settings: SolverSettings,
    times: Iterable[float] | TimeStepFunction,
    initial_time: float,
    save_predicate: Literal[True],
    output_settings: OutputSettings | None = None,
    n_threads: int = 1,
) -> tuple[SolverState, ...]: ...
@overload
def run_solver(
    geometry: SimulationGeometry,
    settings: SolverSettings,
    times: Iterable[float] | TimeStepFunction,
    initial_time: float,
    save_predicate: Literal[False],
    output_settings: OutputSettings | None = None,
    n_threads: int = 1,
) -> SolverState: ...


def run_solver(
    geometry: SimulationGeometry,
    settings: SolverSettings,
    times: Iterable[float] | TimeStepFunction,
    initial_time: float = 0,
    save_predicate: SavePredicate | bool = True,
    output_settings: OutputSettings | None = None,
    n_threads: int = 1,
) -> tuple[SolverState, ...] | SolverState:
    """Run the flow solver to obtain specified circulations.

    Parameters
    ----------
    geometry : SimulationGeometry
        Geometry to solver for.

    settings : SolverSettings
        Settings of the solver.

    times : Iterable of float or TimeStepFunction
        If an iterable of floats is provided, these are the time steps at which the
        solver will be evaluated.

        If a TimeStepFunction is provided, it will be called
        at each step with the current step, time, and state to determine the next time
        step. If the function returns None, the simulation is finished and the solver
        will stop.

    save_predicate : SavePredicate or bool, default: True
        Predicate to determine if the solver state should be saved at step.
        If set to ``True``, all steps will be saved. If set to ``False``, just the last
        step will be saved. If a callable is provided, it will be called with the
        current step, time, and state to determine if the state should be saved.

    output_settings : OutputSettings, optional
        Settings related to file IO.

    n_threads : int, default: 1
        Number of threads to use.

    Returns
    -------
    tuple of SolverState
        The solver states at each output time step.
    """
    if not isinstance(save_predicate, bool) and not callable(save_predicate):
        raise TypeError(
            f"save_predicate must be a bool or a callable, got {type(save_predicate)}."
        )

    if not callable(times):
        simulation_times = tuple(times)

        def _advance_time_step(
            step: int, time: float, state: SolverState
        ) -> float | None:
            """Advance time step based on the provided iterable of times."""
            del time, state
            if step >= len(simulation_times):
                return None
            return simulation_times[step]

        times = _advance_time_step

    results: list[SolverState] = list()

    # Main state
    state = SolverState.create_new(initial_time, geometry, settings)
    state.circulation[:] = 0  # Zero out the circulation for the first time step
    # Output state to avoid unnecessary allocations during the loop
    out_state = SolverState.create_new(initial_time, geometry, settings)
    compute_mem = PyVLComputeMemory.from_solver_settings(geometry, settings)
    system = SolverSystem(
        time=initial_time,
        tol=settings.model_settings.vortex_limit,
        geo=[geometry.geometries[name] for name in geometry.geometries],
    )

    iteration = 0
    time = initial_time
    while time is not None:
        iteration_begin_time = perf_counter()
        out_state = update_simulation_state(
            state,
            time,
            out_state=out_state,
            compute_memory=compute_mem,
            n_threads=n_threads,
            system=system,
        )
        iteration_end_time = perf_counter()
        # Swap the states for the next iteration
        state, out_state = out_state, state

        # Save the state if needed
        if save_predicate is True or (
            callable(save_predicate) and save_predicate(iteration, time, state)
        ):
            results.append(state)
            # Need new output state
            out_state = SolverState.create_new(time, geometry, settings)

        # Save output to file if needed
        if output_settings is not None:
            _save_output_if_needed(output_settings, state, iteration)

        print(
            f"Finished iteration {iteration + 1:d} at {time:=g} in "
            f"{iteration_end_time - iteration_begin_time:g} seconds."
        )
        iteration += 1
        time = times(iteration, time, state)

    if save_predicate is False:
        return state

    return tuple(results)


def run_solver_steady_state(
    geometry: SimulationGeometry,
    settings: SolverSettings,
    dt: float,
    initial_time: float = 0,
    atol: float = 1e-8,
    rtol: float = 1e-5,
    max_steps: int | None = None,
    output_settings: OutputSettings | None = None,
    n_threads: int = 1,
) -> SolverState:
    """Run the flow solver to obtain specified circulations at steady state.

    The solver is first always ran until the wake is fully developed. After that
    the convergence is checked.

    The steady state is determined by checking the max norm of the circulation change
    for each time step. If either the absolute or the relative tolerance is satisfied,
    the simulation is considered converged and the solver will stop.

    Parameters
    ----------
    geometry : SimulationGeometry
        Geometry to solver for.

    settings : SolverSettings
        Settings of the solver.

    dt : float
        Time step size to use for advancing the simulation.

    initial_time : float, default: 0
        Initial time of the simulation.

    atol : float, default: 1e-8
        Absolute tolerance for convergence.

    rtol : float, default: 1e-5
        Relative tolerance for convergence.

    max_steps : int or None, default: None
        Maximum number of steps to run before stopping the simulation. If None, there is
        no limit on the number of steps.

    output_settings : OutputSettings, optional
        Settings related to file IO.

    n_threads : int, default: 1
        Number of threads to use.

    Returns
    -------
    SolverState
        The final solver state at convergence or after reaching the maximum number of
        steps.
    """
    old_circulations = np.empty(geometry.mesh_joined.n_lines, np.double)

    def _steady_state_time_step(
        step: int, time: float, state: SolverState
    ) -> float | None:
        """Advance time step for steady state solver."""
        if max_steps is not None and step >= max_steps:
            # End if max steps reached
            return None

        if state.wake.quad_count < state.wake.capacity:
            # Keep going until wake is full
            return time + dt

        # Check for convergence based on the max circulation change
        max_change = np.max(np.abs(state.circulation - old_circulations))
        if (
            max_change <= atol
            or max_change
            <= rtol * np.max(np.abs(state.circulation + old_circulations)) / 2
        ):
            # Converged, end the simulation
            return None

        # Update old circulations
        old_circulations[:] = state.circulation
        return time + dt

    return run_solver(
        geometry=geometry,
        settings=settings,
        times=_steady_state_time_step,
        initial_time=initial_time,
        save_predicate=False,
        output_settings=output_settings,
        n_threads=n_threads,
    )
