"""Aerodynamic element support."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

import numpy as np
import numpy.typing as npt

from pyvl.cvl import Mesh, ReferenceFrame
from pyvl.geometry import SimulationGeometry


@dataclass(frozen=True)
class TransformInfo:
    """Information about what transformations to apply to which points and elements."""

    rf: ReferenceFrame
    point_indices: slice
    element_indices: slice


@dataclass(frozen=True)
class ImplicitElements:
    """State of the solver at a specific moment."""

    primal: Mesh
    dual: Mesh
    transform_info: tuple[TransformInfo, ...]
    positions: npt.NDArray[np.double]
    normals: npt.NDArray[np.double]
    control_points: npt.NDArray[np.double]

    @classmethod
    def from_geometry(
        cls,
        geometry: SimulationGeometry,
    ) -> Self:
        """Create a implicit elements from simulation geometry.

        Parameters
        ----------
        geometry : SimulationGeometry
            The simulation geometry from which to create the implicit elements.

        Returns
        -------
        Self
            The created implicit elements.
        """
        geometry = geometry
        positions = np.empty((geometry.n_points, 3), np.float64)
        t_info: list[TransformInfo] = []
        meshes: list[Mesh] = []
        for part in geometry.values():
            # Add TransformInfo
            t_info.append(
                TransformInfo(
                    rf=part.rf, point_indices=part.points, element_indices=part.surfaces
                )
            )
            # Fill in the positions
            positions[part.points] = part.pos
            # Add the mesh
            meshes.append(part.msh)

        # Combine meshes
        primal = Mesh.merge_meshes(*meshes)
        dual = primal.compute_dual()

        # Compute normals
        normals = primal.surface_normal(positions)
        # Compute control points (centroids)
        control_points = primal.surface_average_vec3(positions)
        return cls(primal, dual, tuple(t_info), positions, normals, control_points)

    def at_time(
        self,
        t: float,
        out: ImplicitElements | None = None,
        out_v: npt.NDArray[np.double] | None = None,
    ) -> tuple[ImplicitElements, npt.NDArray[np.double]]:
        """Get the implicit elements at a specific time.

        Parameters
        ----------
        t : float
            The time at which to get the implicit elements.

        out : ImplicitElements, optional
            An optional ImplicitElements object to store the result in. If None, a new
            object will be created.

        out_v : npt.NDArray[np.double], optional
            An optional array to store the control point velocities in. If None, a new
            array will be created. If provided, the control point velocities will be
            computed and stored in this array.

        Returns
        -------
        ImplicitElements
            The implicit elements at the specified time.

        array
            Velocity of the control points at the specified time.
        """
        out_pos: npt.NDArray[np.double]
        out_normals: npt.NDArray[np.double]
        out_control_points: npt.NDArray[np.double]
        if out is None:
            out_pos = np.empty_like(self.positions)
            out_normals = np.empty_like(self.normals)
            out_control_points = np.empty_like(self.control_points)
        else:
            if not isinstance(out, ImplicitElements):
                raise TypeError("out must be an ImplicitElements object or None.")
            if out.primal != self.primal or out.dual != self.dual:
                raise ValueError("out must have the same primal and dual meshes.")
            if out.transform_info != self.transform_info:
                raise ValueError("out must have the same transform_info.")
            if (
                out.normals.shape != self.normals.shape
                or out.normals.dtype != self.normals.dtype
            ):
                raise ValueError(
                    "out.normals must have the same shape and dtype as self.normals."
                )
            if (
                out.control_points.shape != self.control_points.shape
                or out.control_points.dtype != self.control_points.dtype
            ):
                raise ValueError(
                    "out.control_points must have the same shape and dtype as "
                    "self.control_points."
                )
            out_pos = out.positions
            out_normals = out.normals
            out_control_points = out.control_points

        if out_v is not None:
            if not isinstance(out_v, np.ndarray):
                raise TypeError("out_v must be a numpy array or None.")
            if (
                out_v.shape != self.control_points.shape
                or out_v.dtype != self.control_points.dtype
            ):
                raise ValueError(
                    "out_v must have the same shape and dtype as self.control_points."
                )
        else:
            out_v = np.empty_like(self.control_points)

        for t_info in self.transform_info:
            # Apply transformations to the positions, normals, and control points
            t_info.rf.to_global_position(
                self.positions[t_info.point_indices],
                t,
                # Should be fine, since this should be contiguous
                out=out_pos[t_info.point_indices],
            )
            t_info.rf.to_global_position(
                self.normals[t_info.element_indices],
                t,
                # Should be fine, since this should be contiguous
                out=out_normals[t_info.element_indices],
            )
            v_slice = out_v[t_info.element_indices]
            v_slice.fill(0)
            t_info.rf.to_global_velocity(
                position=self.control_points[t_info.element_indices],
                velocity=v_slice,
                time=t,
                # Should be fine, since this should be contiguous
                out_position=out_control_points[t_info.element_indices],
                out_velocity=out_v[t_info.element_indices],
            )

        if out:
            # Arrays were backed by the out object, return it.
            return out, out_v

        # Return the arrays as a part of the new ImplicitElements.
        return ImplicitElements(
            primal=self.primal,
            dual=self.dual,
            transform_info=self.transform_info,
            positions=out_pos,
            normals=out_normals,
            control_points=out_control_points,
        ), out_v

    def cp_velocity(
        self, t: float, out: npt.NDArray[np.double] | None = None
    ) -> npt.NDArray[np.double]:
        """Get the velocity of the control points at a specific time.

        Parameters
        ----------
        t : float
            The time at which to get the control point velocities.

        out : npt.NDArray[np.double], optional
            An optional array to store the result in. If None, a new array will be
            created.

        Returns
        -------
        array
            The velocity of the control points at the specified time.
        """
        out_cp_vel: npt.NDArray[np.double]
        if out is None:
            out_cp_vel = np.empty_like(self.control_points)
        else:
            if not isinstance(out, np.ndarray):
                raise TypeError("out must be a numpy array or None.")
            if (
                out.shape != self.control_points.shape
                or out.dtype != self.control_points.dtype
            ):
                raise ValueError(
                    "out must have the same shape and dtype as self.control_points."
                )
            out_cp_vel = out

        for t_info in self.transform_info:
            out_section = out_cp_vel[t_info.element_indices]
            out_section.fill(0)
            # Apply transformations to the control points
            t_info.rf.to_global_velocity(
                self.control_points[t_info.element_indices],
                out_section,
                t,
                # Should be fine, since this should be contiguous
                out_velocity=out_section,
            )

        return out_cp_vel

    def self_induction(
        self,
        vortex_tol: float = 1e-6,
        out: npt.NDArray[np.double] | None = None,
        line_buffer: npt.NDArray[np.double] | None = None,
        *,
        threads: int = 1,
    ) -> npt.NDArray[np.double]:
        """Get the self-induction of the control points.

        Parameters
        ----------
        vortex_tol : float, default: 1e-6
            Minimum distance for vortex induction, below which the induction is
            cut to zero.

        out : npt.NDArray[np.double], optional
            An optional array to store the result in. If None, a new array will be
            created.

        line_buffer : array, optional
            An optional array to use as a buffer for line segments during the induction
            computation. If None, a new array will be created internally.

        threads : int, default: 1
            The number of threads to use for the induction computation.

        Returns
        -------
        array
            The self-induction of the control points.
        """
        out_self_ind: npt.NDArray[np.double]
        n_elem = self.primal.n_surfaces
        n_lines = self.primal.n_lines
        if out is None:
            out_self_ind = np.empty((n_elem, n_elem), dtype=np.double)
        else:
            if not isinstance(out, np.ndarray):
                raise TypeError("out must be a numpy array or None.")
            if out.shape != (n_elem, n_elem) or out.dtype != np.double:
                raise ValueError(
                    "out must have the same shape and dtype as the self-induction matrix."
                )
            out_self_ind = out
        if line_buffer is not None:
            if not isinstance(line_buffer, np.ndarray):
                raise TypeError("line_buffer must be a numpy array or None.")
            if line_buffer.shape != (n_lines, 3) or line_buffer.dtype != np.double:
                raise ValueError(
                    "line_buffer must have the same shape and dtype as the line buffer."
                )

        # Compute self-induction using the primal mesh and the control points.
        return self.primal.induction_matrix3(
            tol=vortex_tol,
            positions=self.positions,
            control_points=self.control_points,
            normals=self.normals,
            out=out_self_ind,
            thread_count=threads,
            line_buffer=line_buffer,
        )
